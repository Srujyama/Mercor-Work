#!/usr/bin/env python3
"""
Dual Autograder Rerun for Gemini Responses (GPT-5 + Gemini)
-----------------------------------------------------------

- Looks for records where the checkbox column "Errored" is checked.
- For those records ONLY, it re-grades the field:
    "Gemini 3.0 model responses"

- Overwrites (for those records ONLY) the GENERAL autorater fields:

    Gemini Autorater - Gemini 3.0 Response Score
    Gemini Autorater - Gemini 3.0 Response Summary
    GPT5 Autorater - Gemini 3.0 Response Score
    GPT5 Autorater - Gemini 3.0 Response Summary

- Uses:
    - GPT-5 (OpenAI) via GPT_API_KEY
    - Gemini via GOOGLE_API_KEY

- Still uses the Rubric JSON + MT Prompt per record.
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

if not AIRTABLE_API_KEY:
    raise RuntimeError("Missing AIRTABLE_API_KEY in .env")

GPT_API_KEY = os.getenv("GPT_API_KEY")
if not GPT_API_KEY:
    raise RuntimeError("Missing GPT_API_KEY in .env")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("Missing GOOGLE_API_KEY in .env")

# Single view config: we’ll use the General view
GENERAL_VIEW_ID = os.getenv("AIRTABLE_VIEW_General")
if not GENERAL_VIEW_ID:
    raise RuntimeError("Missing AIRTABLE_VIEW_General in .env")

VIEW_NAME = "General"
INPUT_FIELD = "Gemini 3.0 model responses"
ERRORED_FIELD = "Errored"  # Checkbox column

# Tunables
GPT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3-pro-preview")
PER_KEY_PARALLEL = int(os.getenv("PER_KEY_PARALLEL", "4"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "4"))

# ----------------------- Logging -----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# ----------------------- Output field names (GENERAL) -----------------------
GPT_SCORE_FIELD = "GPT5 Autorater - Gemini 3.0 Response Score"
GPT_SUMMARY_FIELD = "GPT5 Autorater - Gemini 3.0 Response Summary"
GEMINI_SCORE_FIELD = "Gemini Autorater - Gemini 3.0 Response Score"
GEMINI_SUMMARY_FIELD = "Gemini Autorater - Gemini 3.0 Response Summary"


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
class GeminiResponseDualAutograderRerun:
    """
    Reruns autograding ONLY for records where `Errored` checkbox is checked.
    Uses the unified "Gemini 3.0 model responses" field as the solution,
    and writes to the GENERAL autorater fields (no per-view suffix).
    """

    def __init__(self):
        # Airtable
        self.api = Api(AIRTABLE_API_KEY)
        self.table = self.api.table(AIRTABLE_BASE_ID, AIRTABLE_TABLE_ID)

        # Stats
        self.stats = {"processed": 0, "graded": 0, "failed": 0, "skipped": 0}

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
            '  \"<criterion_key>\": {\n'
            '    \"decision\": true|false,\n'
            '    \"reasoning\": \"Exactly 10 sentences.\"\n'
            "  }, ...\n"
            "}\n"
            "No extra keys, no markdown, no code fences."
        )

    # ------------------- Record discovery -------------------
    def get_errored_records(self):
        """
        Return records from the General view where:
          - Errored checkbox is True
          - Gemini 3.0 model responses is non-empty
        We will regrade ONLY these records, overwriting autorater fields.
        """
        try:
            fields_to_fetch = [
                "Rubric JSON",
                "Consolidated Prompt - 10/25",
                INPUT_FIELD,
                ERRORED_FIELD,
                GPT_SCORE_FIELD,
                GPT_SUMMARY_FIELD,
                GEMINI_SCORE_FIELD,
                GEMINI_SUMMARY_FIELD,
            ]
            records = self.table.all(view=GENERAL_VIEW_ID, fields=fields_to_fetch)
            logger.info(
                f"[{VIEW_NAME}] Fetched {len(records)} total records from view {GENERAL_VIEW_ID}"
            )
            needing = []
            for rec in records:
                f = rec.get("fields", {})
                errored = bool(f.get(ERRORED_FIELD, False))
                solution = (f.get(INPUT_FIELD) or "").strip()
                if not errored:
                    continue
                if not solution:
                    continue
                needing.append(rec)
            logger.info(
                f"[{VIEW_NAME}] Found {len(needing)} records with Errored = True and non-empty solution"
            )
            return needing
        except Exception as e:
            logger.error(f"[{VIEW_NAME}] Error fetching records: {e}")
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
        Gemini grading call.
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
        Grade a solution with either 'gpt' or 'gemini'.

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

        errored = bool(fields.get(ERRORED_FIELD, False))
        if not errored:
            self.stats["skipped"] += 1
            return False

        rubric_json = fields.get("Rubric JSON", "")
        prompt = str(fields.get("Consolidated Prompt - 10/25", "") or "")

        if not rubric_json:
            logger.warning(f"[{VIEW_NAME}] No rubric found for {record_id}")
            return False
        try:
            rubric = json.loads(rubric_json)
        except json.JSONDecodeError as e:
            logger.error(f"[{VIEW_NAME}] Invalid rubric JSON for {record_id}: {e}")
            return False

        solution = (fields.get(INPUT_FIELD) or "").strip()
        if not solution:
            logger.warning(f"[{VIEW_NAME}] No solution in '{INPUT_FIELD}' for {record_id}")
            return False

        logger.info(f"[{VIEW_NAME}] Re-grading record {record_id} (Errored = True)")

        # Grade with both backends in parallel
        tasks = [
            self.grade_with_backend("gpt", solution, rubric, prompt),
            self.grade_with_backend("gemini", solution, rubric, prompt),
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        updates = {}
        gpt_res, gem_res = results

        if isinstance(gpt_res, Exception):
            logger.error(f"[{VIEW_NAME}] GPT grading failed for {record_id}: {gpt_res}")
        else:
            updates[GPT_SCORE_FIELD] = gpt_res["percentage"]
            updates[GPT_SUMMARY_FIELD] = gpt_res["summary"]

        if isinstance(gem_res, Exception):
            logger.error(
                f"[{VIEW_NAME}] Gemini grading failed for {record_id}: {gem_res}"
            )
        else:
            updates[GEMINI_SCORE_FIELD] = gem_res["percentage"]
            updates[GEMINI_SUMMARY_FIELD] = gem_res["summary"]

        if not updates:
            return False

        # NOTE: we intentionally do NOT clear the Errored flag here
        # (you can add that if you want: updates[ERRORED_FIELD] = False)
        loop = asyncio.get_running_loop()
        try:
            await _with_retries(
                lambda: loop.run_in_executor(
                    None, self.table.update, record_id, updates
                )
            )
            logger.info(f"[{VIEW_NAME}] ✅ Updated record {record_id}")
            return True
        except Exception as e:
            logger.error(f"[{VIEW_NAME}] Airtable update failed for {record_id}: {e}")
            return False

    # ------------------- Runner -------------------
    async def run(self):
        logger.info(
            "🤖 Starting Dual Autograder RERUN for Gemini responses (GPT-5 + Gemini, Errored only)"
        )

        records = self.get_errored_records()
        if not records:
            logger.info("✅ No records with Errored = True and non-empty solution")
            return

        per_view_sem = asyncio.Semaphore(8)
        tasks = []

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
            f"Graded={self.stats['graded']} Failed={self.stats['failed']} "
            f"Skipped(non-errored)={self.stats['skipped']}"
        )


# ------------------- Entrypoint -------------------
async def main():
    grader = GeminiResponseDualAutograderRerun()
    await grader.run()


if __name__ == "__main__":
    asyncio.run(main())
