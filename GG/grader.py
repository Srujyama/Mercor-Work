#!/usr/bin/env python3
"""
Dual Autograder for Gemini Responses (GPT-5 + Gemini 2.5 Pro)
-------------------------------------------------------------

- Reads Gemini's generated responses from:
    "Gemini 3.0 model responses"

- Uses 'Consolidated Prompt - 10/25' as the original prompt.

- For each record, writes four fields on the same table:

    Gemini Autorater - Gemini 3.0 Response Score        (0-100)
    Gemini Autorater - Gemini 3.0 Response Summary      (JSON summary)

    GPT5 Autorater - Gemini 3.0 Response Score          (0-100)
    GPT5 Autorater - Gemini 3.0 Response Summary        (JSON summary)

- Uses:
    - GPT-5 (OpenAI) via GPT_API_KEY
    - Gemini 2.5 Pro via GOOGLE_API_KEY

- Still uses the Rubric JSON per record.
"""

import asyncio
import json
import logging
import os
import random
import re

from dotenv import load_dotenv
from google import genai
from google.genai import types
from openai import AsyncOpenAI
from pyairtable import Api

# ----------------------- Load .env secrets -----------------------
load_dotenv()

AIRTABLE_API_KEY = os.getenv("AIRTABLE_API_KEY")
AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID", "appgeueGlH9mCUTvu")
AIRTABLE_TABLE_ID = os.getenv("AIRTABLE_TABLE", "tblfy3EPxl1PHvKV7")
AIRTABLE_VIEW_ID = os.getenv("AIRTABLE_VIEW", "viwvAyBaCl2YgsFD6")

if not AIRTABLE_API_KEY:
    raise RuntimeError("Missing AIRTABLE_API_KEY in .env")

GPT_API_KEY = os.getenv("GPT_API_KEY")
if not GPT_API_KEY:
    raise RuntimeError("Missing GPT_API_KEY in .env")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("Missing GOOGLE_API_KEY in .env")

# Tunables
# GPT-5 model (can override via OPENAI_MODEL env var)
GPT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5")
# Gemini 2.5 Pro model (can override via GEMINI_MODEL env var)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
PER_KEY_PARALLEL = int(os.getenv("PER_KEY_PARALLEL", "4"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "4"))

# Field names
INPUT_FIELD = "Gemini 3.0 model responses"


def gpt_score_field() -> str:
    return "GPT5 Autorater - Gemini 3.0 Response Score"


def gpt_summary_field() -> str:
    return "GPT5 Autorater - Gemini 3.0 Response Summary"


def gemini_score_field() -> str:
    return "Gemini Autorater - Gemini 3.0 Response Score"


def gemini_summary_field() -> str:
    return "Gemini Autorater - Gemini 3.0 Response Summary"


# ----------------------- Logging -----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# ----------------------- Small helpers -----------------------
async def _with_retries(coro_factory, *, retries=MAX_RETRIES, base=0.5, jitter=0.25):
    """
    Exponential backoff for async ops.
    Treat invalid-parameter 4xx as fatal (no retries).
    """
    for attempt in range(retries):
        try:
            return await coro_factory()
        except Exception as e:
            msg = str(e)
            is_param_error = (
                "invalid_request_error" in msg
                or "unsupported_value" in msg
                or "'param':" in msg
                or "HTTP/1.1 400" in msg
            )
            if is_param_error or attempt == retries - 1:
                raise
            sleep_time = base * (2**attempt) + random.uniform(0, jitter)
            logger.warning(f"Transient error ({e}), retrying in {sleep_time:.2f}s")
            await asyncio.sleep(sleep_time)


def _safe_json_loads(s: str):
    try:
        return json.loads(s)
    except Exception:
        return None


def _parse_boolean(x):
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        return x.strip().lower() == "true"
    return False


# ----------------------- Main Class -----------------------
class GeminiResponseDualAutograder:
    def __init__(self):
        # Airtable
        self.api = Api(AIRTABLE_API_KEY)
        self.table = self.api.table(AIRTABLE_BASE_ID, AIRTABLE_TABLE_ID)

        # Stats
        self.stats = {"processed": 0, "graded": 0, "failed": 0}

        # GPT client + concurrency
        self.gpt_client = AsyncOpenAI(api_key=GPT_API_KEY)
        self.gpt_semaphore = asyncio.Semaphore(max(1, PER_KEY_PARALLEL))

        # Gemini client (sync, used via executor)
        self.gemini_client = genai.Client(
            api_key=GOOGLE_API_KEY,
            http_options=types.HttpOptions(api_version="v1beta"),
        )
        self.gemini_semaphore = asyncio.Semaphore(max(1, PER_KEY_PARALLEL))

        # Global API concurrency cap across both backends
        self.global_api_semaphore = asyncio.Semaphore(max(4, PER_KEY_PARALLEL * 2))

        # System prompts (reused for both GPT and Gemini)
        self.system_prompt_batched = (
            "You are an expert grader evaluating solutions against specific criteria.\n"
            "For EACH criterion, do two things:\n"
            "1) Produce EXACTLY 10 sentences of reasoning explaining your evaluation.\n"
            "2) Output a boolean decision whether the solution meets the criterion.\n\n"
            "IMPORTANT OUTPUT FORMAT:\n"
            "Return ONLY a single JSON object of the shape:\n"
            "{\n"
            '  "<criterion_key>": {\n'
            '    "decision": true|false,\n'
            '    "reasoning": "Exactly 10 sentences."\n'
            "  }, ...\n"
            "}\n"
            "No extra keys, no markdown, no code fences."
        )

    # ------------------- Record discovery -------------------
    def get_records(self):
        """
        Return records from the single configured view where the input field
        is non-empty and at least one of the four output fields is missing.
        """
        try:
            fields_to_fetch = [
                "Rubric JSON",
                "Consolidated Prompt - 10/25",
                INPUT_FIELD,
                gpt_score_field(),
                gpt_summary_field(),
                gemini_score_field(),
                gemini_summary_field(),
            ]
            records = self.table.all(view=AIRTABLE_VIEW_ID, fields=fields_to_fetch)
            logger.info(
                f"[Main] Fetched {len(records)} total records from view {AIRTABLE_VIEW_ID}"
            )
            needing = []
            for rec in records:
                f = rec.get("fields", {})
                inp = (f.get(INPUT_FIELD) or "").strip()
                if not inp:
                    continue
                if (
                    f.get(gpt_score_field()) is None
                    or f.get(gpt_summary_field()) is None
                    or f.get(gemini_score_field()) is None
                    or f.get(gemini_summary_field()) is None
                ):
                    needing.append(rec)
            logger.info(f"[Main] Found {len(needing)} records needing grading")
            return needing
        except Exception as e:
            logger.error(f"[Main] Error fetching records: {e}")
            return []

    # ------------------- GPT chat helper (JSON) -------------------
    async def _gpt_chat_json(self, messages):
        """
        GPT-5 JSON grading call.
        """

        async def _call_json():
            async with self.global_api_semaphore, self.gpt_semaphore:
                return await self.gpt_client.chat.completions.create(
                    model=GPT_MODEL,
                    messages=messages,
                    response_format={"type": "json_object"},
                )

        try:
            resp = await _with_retries(_call_json)
            content = resp.choices[0].message.content or "{}"
            data = _safe_json_loads(content)
            if data is None:
                m = re.search(r"\{.*\}", content, re.DOTALL)
                data = _safe_json_loads(m.group(0)) if m else None
            return data if isinstance(data, dict) else None
        except Exception as e:
            logger.warning(
                f"GPT JSON mode failed ({e}); retrying without response_format."
            )

            async def _call_text():
                async with self.global_api_semaphore, self.gpt_semaphore:
                    return await self.gpt_client.chat.completions.create(
                        model=GPT_MODEL,
                        messages=messages,
                    )

            resp = await _with_retries(_call_text)
            return resp.choices[0].message.content or ""

    # ------------------- Gemini chat helper (JSON via prompt) -------------------
    async def _gemini_chat_json(self, messages):
        """
        Gemini 2.5 Pro grading call.
        We stringify the messages into a single prompt and send to Gemini.
        """
        # Turn messages into a plain text conversation
        pieces = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            pieces.append(f"{role.upper()}:\n{content}\n")
        full_prompt = "\n".join(pieces)

        loop = asyncio.get_running_loop()

        async def _call():
            async with self.global_api_semaphore, self.gemini_semaphore:

                def sync_call():
                    return self.gemini_client.models.generate_content(
                        model=GEMINI_MODEL,
                        contents=full_prompt,
                    )

                return await loop.run_in_executor(None, sync_call)

        try:
            resp = await _with_retries(_call)
            text = getattr(resp, "text", None)
            if not text:
                cands = getattr(resp, "candidates", [])
                if cands and getattr(cands[0], "content", None):
                    parts_out = cands[0].content.parts or []
                    text = "\n".join(
                        [
                            getattr(p, "text", "")
                            for p in parts_out
                            if getattr(p, "text", "")
                        ]
                    )
            content = (text or "").strip()
            data = _safe_json_loads(content)
            if data is None:
                m = re.search(r"\{.*\}", content, re.DOTALL)
                data = _safe_json_loads(m.group(0)) if m else None
            return data if isinstance(data, dict) else None
        except Exception as e:
            logger.error(f"Gemini JSON grading failed: {e}")
            return None

    # ------------------- Shared rubric grading logic -------------------
    async def grade_with_backend(
        self, backend: str, solution: str, rubric: list, prompt: str
    ) -> dict:
        """
        Grade a solution with either 'gpt' (GPT-5) or 'gemini' (Gemini 2.5 Pro).

        Returns:
            {
                "percentage": float,
                "summary": "<JSON string of per-criterion breakdown>"
            }
        """
        # Build {criterion_key: description}
        crit_map = {}
        for c in rubric:
            key = list(c.keys())[0]
            desc = (c[key] or {}).get("description", "")
            crit_map[key] = desc

        user_prompt = (
            "Evaluate the SOLUTION against each CRITERION.\n"
            "For each criterion, return `decision` (true/false) and `reasoning` (exactly 10 sentences).\n\n"
            f"ORIGINAL PROMPT:\n{prompt}\n\n"
            f"SOLUTION (Gemini 3.0 response):\n{solution}\n\n"
            f"CRITERIA (JSON):\n{json.dumps(crit_map, ensure_ascii=False)}\n"
        )

        messages = [
            {"role": "system", "content": self.system_prompt_batched},
            {"role": "user", "content": user_prompt},
        ]

        if backend == "gpt":
            result = await self._gpt_chat_json(messages)
        elif backend == "gemini":
            result = await self._gemini_chat_json(messages)
        else:
            raise ValueError(f"Unknown backend: {backend}")

        decisions = {}
        if isinstance(result, dict):
            for key, obj in result.items():
                decision = _parse_boolean((obj or {}).get("decision"))
                reasoning = (obj or {}).get("reasoning", "") or ""
                decisions[key] = {"decision": decision, "reasoning": reasoning.strip()}
        else:
            # If we don't get JSON at all, mark all as false with empty reasoning
            for c in rubric:
                key = list(c.keys())[0]
                decisions[key] = {"decision": False, "reasoning": ""}

        graded, total = [], 0
        for c in rubric:
            key = list(c.keys())[0]
            meta = c[key]
            d = decisions.get(key, {"decision": False, "reasoning": ""})
            ok = bool(d["decision"])
            graded.append(
                {
                    "autorating": ok,
                    "description": meta.get("description", ""),
                    "weight": meta.get("weight", ""),
                    "criterion_type": meta.get("criterion_type", []),
                    "dependent_criteria": meta.get("dependent_criteria", []),
                    "justification": meta.get("justification", ""),
                    "sources": meta.get("sources", ""),
                    "human_rating": meta.get("human_rating", ""),
                    "reasoning": d.get("reasoning", ""),
                }
            )
            total += int(ok)
        pct = (total / len(rubric) * 100) if rubric else 0.0
        return {
            "percentage": pct,
            "summary": json.dumps(graded, separators=(",", ":")),
        }

    # ------------------- Per-record processing -------------------
    async def process_record(self, record: dict) -> bool:
        fields = record.get("fields", {})
        record_id = record["id"]

        rubric_json = fields.get("Rubric JSON", "")
        # ONLY use the consolidated prompt
        prompt = str(fields.get("Consolidated Prompt - 10/25") or "")

        if not rubric_json:
            logger.warning(f"[Main] No rubric found for {record_id}")
            return False
        try:
            rubric = json.loads(rubric_json)
        except json.JSONDecodeError as e:
            logger.error(f"[Main] Invalid rubric JSON for {record_id}: {e}")
            return False

        solution = (fields.get(INPUT_FIELD) or "").strip()
        if not solution:
            return False

        # If all outputs already filled, skip
        if (
            fields.get(gpt_score_field()) is not None
            and fields.get(gpt_summary_field()) is not None
            and fields.get(gemini_score_field()) is not None
            and fields.get(gemini_summary_field()) is not None
        ):
            return False

        logger.info(f"[Main] Grading record {record_id} (Gemini response)")

        # Grade with both backends in parallel: GPT-5 + Gemini 2.5 Pro
        tasks = [
            self.grade_with_backend("gpt", solution, rubric, prompt),
            self.grade_with_backend("gemini", solution, rubric, prompt),
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        updates = {}
        gpt_res, gem_res = results

        if isinstance(gpt_res, Exception):
            logger.error(f"[Main] GPT grading failed for {record_id}: {gpt_res}")
        else:
            updates[gpt_score_field()] = gpt_res["percentage"]
            updates[gpt_summary_field()] = gpt_res["summary"]

        if isinstance(gem_res, Exception):
            logger.error(f"[Main] Gemini grading failed for {record_id}: {gem_res}")
        else:
            updates[gemini_score_field()] = gem_res["percentage"]
            updates[gemini_summary_field()] = gem_res["summary"]

        if not updates:
            return False

        loop = asyncio.get_running_loop()
        try:
            await _with_retries(
                lambda: loop.run_in_executor(
                    None, self.table.update, record_id, updates
                )
            )
            logger.info(f"[Main] ✅ Updated record {record_id}")
            return True
        except Exception as e:
            logger.error(f"[Main] Airtable update failed for {record_id}: {e}")
            return False

    # ------------------- Runner -------------------
    async def run(self):
        logger.info("🤖 Starting Dual Autograder (GPT-5 + Gemini 2.5 Pro)")

        records = self.get_records()
        if not records:
            logger.info("✅ No records need grading in the configured view")
            return

        tasks = []
        per_view_sem = asyncio.Semaphore(8)

        async def process_with_sem(rec):
            async with per_view_sem:
                ok = await self.process_record(rec)
                self.stats["processed"] += 1
                if ok:
                    self.stats["graded"] += 1
                else:
                    self.stats["failed"] += 1

        for r in records:
            tasks.append(process_with_sem(r))

        await asyncio.gather(*tasks)
        logger.info(
            f"🎉 Done. Processed={self.stats['processed']} "
            f"Graded={self.stats['graded']} Failed={self.stats['failed']}"
        )


# ------------------- Entrypoint -------------------
async def main():
    grader = GeminiResponseDualAutograder()
    await grader.run()


if __name__ == "__main__":
    asyncio.run(main())
