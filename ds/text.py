#!/usr/bin/env python3
"""
Dual Autograder for Combined View
---------------------------------

View: AIRTABLE_VIEW_COMBINED

Columns:
- Prompt:               "Consolidated Prompt - 10/25"
- Interaction type:     "Interaction Type"
- Rubric:               "Rubric JSON"

Model responses:
- Gemini 2.5 response:  "Consolidated Gemini Response - 10/25"
- Gemini 3.0 response:  "Gemini 3.0 model responses"
- GPT-5 response:       "GPT5 Response"

Autorater outputs:

Gemini 2.5 response:
- Gemini Autorater score:  "Gemini Autorater - Gemini Response Score"
- Gemini Autorater summary:"[Gemini graded] Gemini Response Scoring Summary"
- GPT5 Autorater score:    "GPT5 Autorater - Gemini Response Score"
- GPT5 Autorater summary:  "[GPT5 graded] Gemini Response Scoring Summary"

Gemini 3.0 response:
- Gemini Autorater score:  "Gemini Autorater - Gemini 3.0 Response Score"
- Gemini Autorater summary:"Gemini Autorater - Gemini 3.0 Response Summary"
- GPT5 Autorater score:    "GPT5 Autorater - Gemini 3.0 Response Score"
- GPT5 Autorater summary:  "GPT5 Autorater - Gemini 3.0 Response Summary"

GPT-5 response:
- Gemini Autorater score:  "Gemini Autorater - GPT5 Response Score"
- Gemini Autorater summary:"[Gemini graded] GPT5 Response Scoring Summary"
- GPT5 Autorater score:    "GPT5 Autorater - GPT5 Response Score"
- GPT5 Autorater summary:  "[GPT5 graded] GPT5 Response Scoring Summary"

Asks:
- Run Gemini 3.0 as autorater on Gemini 2.5, Gemini 3.0, and GPT-5 responses.
- Run GPT-5 as autorater on Gemini 2.5, Gemini 3.0, and GPT-5 responses.
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
AIRTABLE_VIEW_COMBINED = os.getenv("AIRTABLE_VIEW_COMBINED")

if not AIRTABLE_API_KEY:
    raise RuntimeError("Missing AIRTABLE_API_KEY in .env")
if not AIRTABLE_VIEW_COMBINED:
    raise RuntimeError("Missing AIRTABLE_VIEW_COMBINED in .env")

GPT_API_KEY = os.getenv("GPT_API_KEY")
if not GPT_API_KEY:
    raise RuntimeError("Missing GPT_API_KEY in .env")

# Use the first Gemini key by default; you can rotate if you want
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY_1")
if not GOOGLE_API_KEY:
    raise RuntimeError("Missing GOOGLE_API_KEY or GOOGLE_API_KEY_1 in .env")

# Tunables / models
GPT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3-pro-preview")

# Concurrency
PER_KEY_PARALLEL = int(os.getenv("GPT_KEY_CONCURRENCY", "5"))
GEMINI_KEY_CONCURRENCY = int(os.getenv("GEMINI_KEY_CONCURRENCY", "3"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "4"))

# ----------------------- Logging -----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# ----------------------- Helpers -----------------------
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
                or "INVALID_ARGUMENT" in msg
            )
            if is_param_error or attempt == retries - 1:
                raise
            sleep_time = base * (2**attempt) + random.uniform(0, jitter)
            logger.warning(f"Transient error ({e}), retrying in {sleep_time:.2f}s")
            await asyncio.sleep(sleep_time)


# ----------------------- Column names -----------------------
PROMPT_FIELD = "Consolidated Prompt - 10/25"
RUBRIC_FIELD = "Rubric JSON"

# Input responses
RESP_GEM25_FIELD = "Consolidated Gemini Response - 10/25"
RESP_GEM30_FIELD = "Gemini 3.0 model responses"
RESP_GPT5_FIELD = "GPT5 Response"

# Autorater outputs per response type
RESP_TYPES = [
    {
        "name": "Gemini 2.5",
        "input_field": RESP_GEM25_FIELD,
        # GPT autorater on Gemini 2.5
        "gpt_score_field": "GPT5 Autorater - Gemini Response Score",
        "gpt_summary_field": "[GPT5 graded] Gemini Response Scoring Summary",
        # Gemini autorater on Gemini 2.5
        "gemini_score_field": "Gemini Autorater - Gemini Response Score",
        "gemini_summary_field": "[Gemini graded] Gemini Response Scoring Summary",
    },
    {
        "name": "Gemini 3.0",
        "input_field": RESP_GEM30_FIELD,
        # GPT autorater on Gemini 3.0
        "gpt_score_field": "GPT5 Autorater - Gemini 3.0 Response Score",
        "gpt_summary_field": "GPT5 Autorater - Gemini 3.0 Response Summary",
        # Gemini autorater on Gemini 3.0
        "gemini_score_field": "Gemini Autorater - Gemini 3.0 Response Score",
        "gemini_summary_field": "Gemini Autorater - Gemini 3.0 Response Summary",
    },
    {
        "name": "GPT-5",
        "input_field": RESP_GPT5_FIELD,
        # GPT autorater on GPT-5 response
        "gpt_score_field": "GPT5 Autorater - GPT5 Response Score",
        "gpt_summary_field": "[GPT5 graded] GPT5 Response Scoring Summary",
        # Gemini autorater on GPT-5 response
        "gemini_score_field": "Gemini Autorater - GPT5 Response Score",
        "gemini_summary_field": "[Gemini graded] GPT5 Response Scoring Summary",
    },
]


# ----------------------- Main Class -----------------------
class CombinedViewDualAutograder:
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
        self.gemini_semaphore = asyncio.Semaphore(max(1, GEMINI_KEY_CONCURRENCY))

        # Global API concurrency cap across both backends
        self.global_api_semaphore = asyncio.Semaphore(
            max(4, PER_KEY_PARALLEL + GEMINI_KEY_CONCURRENCY)
        )

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

    # ------------------- Fetch records from combined view -------------------
    def get_records(self):
        """
        Fetch all records from the combined view that have a rubric & at least one response
        where any autorater field is missing.
        """
        # Fields we need to read
        fields_to_fetch = [
            PROMPT_FIELD,
            RUBRIC_FIELD,
            RESP_GEM25_FIELD,
            RESP_GEM30_FIELD,
            RESP_GPT5_FIELD,
        ]
        for cfg in RESP_TYPES:
            fields_to_fetch.extend(
                [
                    cfg["gpt_score_field"],
                    cfg["gpt_summary_field"],
                    cfg["gemini_score_field"],
                    cfg["gemini_summary_field"],
                ]
            )

        try:
            records = self.table.all(
                view=AIRTABLE_VIEW_COMBINED,
                fields=list(set(fields_to_fetch)),
            )
            logger.info(
                f"[Combined] Fetched {len(records)} total records from view {AIRTABLE_VIEW_COMBINED}"
            )
        except Exception as e:
            logger.error(f"[Combined] Error fetching records: {e}")
            return []

        needing = []
        for rec in records:
            f = rec.get("fields", {})
            rubric_json = (f.get(RUBRIC_FIELD) or "").strip()
            if not rubric_json:
                # No rubric -> skip
                continue

            has_any_response = False
            needs_any_grading = False

            for cfg in RESP_TYPES:
                sol = (f.get(cfg["input_field"]) or "").strip()
                if not sol:
                    continue
                has_any_response = True

                # If any autorater outputs for this response type are missing, we need grading
                gpt_score = f.get(cfg["gpt_score_field"])
                gpt_summary = f.get(cfg["gpt_summary_field"])
                gem_score = f.get(cfg["gemini_score_field"])
                gem_summary = f.get(cfg["gemini_summary_field"])

                if (
                    gpt_score is None
                    or gpt_summary is None
                    or gem_score is None
                    or gem_summary is None
                ):
                    needs_any_grading = True

            if has_any_response and needs_any_grading:
                needing.append(rec)

        logger.info(f"[Combined] Found {len(needing)} records needing grading")
        return needing

    # ------------------- GPT chat helper (JSON) -------------------
    async def _gpt_chat_json(self, messages):
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
            if not isinstance(c, dict) or not c:
                continue
            key = list(c.keys())[0]
            desc = (c.get(key) or {}).get("description", "")
            crit_map[key] = desc

        user_prompt = (
            "Evaluate the SOLUTION against each CRITERION.\n"
            "For each criterion, return `decision` (true/false) and `reasoning` (exactly 10 sentences).\n\n"
            f"ORIGINAL PROMPT:\n{prompt}\n\n"
            f"SOLUTION (model response):\n{solution}\n\n"
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
                if not isinstance(c, dict) or not c:
                    continue
                key = list(c.keys())[0]
                decisions[key] = {"decision": False, "reasoning": ""}

        graded, total = [], 0
        for c in rubric:
            if not isinstance(c, dict) or not c:
                continue
            key = list(c.keys())[0]
            meta = c.get(key, {}) or {}
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
    async def process_record(self, record: dict):
        self.stats["processed"] += 1
        fields = record.get("fields", {})
        record_id = record["id"]

        rubric_json = (fields.get(RUBRIC_FIELD) or "").strip()
        if not rubric_json:
            logger.info(f"[Combined] {record_id}: No rubric, skipping")
            self.stats["skipped"] += 1
            return

        try:
            rubric = json.loads(rubric_json)
        except json.JSONDecodeError as e:
            logger.error(f"[Combined] {record_id}: Invalid rubric JSON: {e}")
            self.stats["skipped"] += 1
            return

        prompt = str(fields.get(PROMPT_FIELD, "") or "")

        updates = {}
        any_graded = False

        # For each response type (Gem 2.5, Gem 3.0, GPT-5)
        for cfg in RESP_TYPES:
            name = cfg["name"]
            solution = (fields.get(cfg["input_field"]) or "").strip()
            if not solution:
                continue

            gpt_score = fields.get(cfg["gpt_score_field"])
            gpt_summary = fields.get(cfg["gpt_summary_field"])
            gem_score = fields.get(cfg["gemini_score_field"])
            gem_summary = fields.get(cfg["gemini_summary_field"])

            # If all four outputs exist, skip this response type
            if (
                gpt_score is not None
                and gpt_summary is not None
                and gem_score is not None
                and gem_summary is not None
            ):
                continue

            logger.info(
                f"[Combined] Grading record {record_id} for {name} response with GPT-5 + Gemini"
            )

            # Grade with both backends in parallel (if they are missing)
            tasks = []
            need_gpt = gpt_score is None or gpt_summary is None
            need_gem = gem_score is None or gem_summary is None

            if need_gpt:
                tasks.append(
                    ("gpt", self.grade_with_backend("gpt", solution, rubric, prompt))
                )
            if need_gem:
                tasks.append(
                    ("gem", self.grade_with_backend("gemini", solution, rubric, prompt))
                )

            if not tasks:
                continue

            # Run tasks
            results = await asyncio.gather(
                *[t[1] for t in tasks], return_exceptions=True
            )

            for (backend, _), res in zip(tasks, results):
                if isinstance(res, Exception):
                    logger.error(
                        f"[Combined] {record_id}: {backend} grading failed for {name}: {res}"
                    )
                    self.stats["failed"] += 1
                    continue

                if backend == "gpt":
                    updates[cfg["gpt_score_field"]] = res["percentage"]
                    updates[cfg["gpt_summary_field"]] = res["summary"]
                    any_graded = True
                elif backend == "gem":
                    updates[cfg["gemini_score_field"]] = res["percentage"]
                    updates[cfg["gemini_summary_field"]] = res["summary"]
                    any_graded = True

        if not any_graded or not updates:
            self.stats["skipped"] += 1
            return

        try:
            self.table.update(record_id, updates)
            logger.info(f"[Combined] ✅ Updated record {record_id}")
            self.stats["graded"] += 1
        except Exception as e:
            logger.error(f"[Combined] Airtable update failed for {record_id}: {e}")
            self.stats["failed"] += 1

    # ------------------- Runner -------------------
    async def run(self):
        logger.info("🤖 Starting Combined View Dual Autograder")
        records = self.get_records()
        if not records:
            logger.info("✅ No records need grading in combined view")
            return

        tasks = []
        per_view_sem = asyncio.Semaphore(8)

        async def process_with_sem(rec):
            async with per_view_sem:
                await self.process_record(rec)

        for r in records:
            tasks.append(process_with_sem(r))

        await asyncio.gather(*tasks)

        logger.info(
            f"🎉 Done. Processed={self.stats['processed']} "
            f"Graded={self.stats['graded']} Failed={self.stats['failed']} "
            f"Skipped={self.stats['skipped']}"
        )


# ------------------- Entrypoint -------------------
async def main():
    grader = CombinedViewDualAutograder()
    await grader.run()


if __name__ == "__main__":
    asyncio.run(main())
