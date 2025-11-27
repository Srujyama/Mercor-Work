#!/usr/bin/env python3
"""
Dual Autograder (GPT-5) for FILE/IMAGE outputs ONLY
---------------------------------------------------

Rules:
- Only grade tasks where Requested Outputs contains "file" or "image".
- Only use:  "Gemini 3.0 Pro Response (File Output) - Gemini App"
- If that field has no attachments:
    → Set "No Image/file generated" = True
    → Skip grading
- Never use "Gemini 3.0 model responses".

Behavior:
- Uses GPT-5 multimodal (image_url) to actually look at the attached files.
- Grades according to Rubric JSON + Consolidated Prompt - 10/25.
- Writes the same score/summary into both:
    * GPT5 Autorater - Gemini 3.0 Response Score
    * GPT5 Autorater - Gemini 3.0 Response Summary
    * Gemini Autorater - Gemini 3.0 Response Score
    * Gemini Autorater - Gemini 3.0 Response Summary
"""

import asyncio
import json
import logging
import os
import random
import re
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import AsyncOpenAI
from pyairtable import Api

# ----------------------- Load ENV -----------------------
load_dotenv()

AIRTABLE_API_KEY = os.getenv("AIRTABLE_API_KEY")
AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID")
AIRTABLE_TABLE_ID = os.getenv("AIRTABLE_TABLE")
GENERAL_VIEW_ID = os.getenv("AIRTABLE_VIEW_General")

GPT_API_KEY = os.getenv("GPT_API_KEY")

if not AIRTABLE_API_KEY or not AIRTABLE_BASE_ID or not AIRTABLE_TABLE_ID:
    raise RuntimeError("Missing Airtable config in .env")
if not GPT_API_KEY:
    raise RuntimeError("Missing GPT_API_KEY in .env")
if not GENERAL_VIEW_ID:
    raise RuntimeError("Missing AIRTABLE_VIEW_General in .env")

# Fields
FILE_OUTPUT_FIELD = "Gemini 3.0 Pro Response (File Output) - Gemini App"
REQUESTED_OUTPUTS_FIELD = "Requested Outputs"
NO_FILE_FIELD = "No Image/file generated"

RUBRIC_FIELD = "Rubric JSON"
PROMPT_FIELD = "Consolidated Prompt - 10/25"

# Autorater output fields (general)
GPT_SCORE_FIELD = "GPT5 Autorater - Gemini 3.0 Response Score"
GPT_SUMMARY_FIELD = "GPT5 Autorater - Gemini 3.0 Response Summary"
GEM_SCORE_FIELD = "Gemini Autorater - Gemini 3.0 Response Score"
GEM_SUMMARY_FIELD = "Gemini Autorater - Gemini 3.0 Response Summary"

GPT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5")
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "4"))
PER_KEY_PARALLEL = int(os.getenv("PER_KEY_PARALLEL", "4"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ----------------------- Helpers -----------------------
def _has_file_or_image(req: Any) -> bool:
    """Return True if Requested Outputs contains 'file' or 'image'."""

    def match(s: str) -> bool:
        s = s.lower()
        return "file" in s or "image" in s

    if req is None:
        return False

    if isinstance(req, str):
        return match(req)

    if isinstance(req, list):
        for item in req:
            if isinstance(item, str) and match(item):
                return True
            if isinstance(item, dict):
                name = item.get("name")
                if isinstance(name, str) and match(name):
                    return True
        return False

    if isinstance(req, dict):
        name = req.get("name")
        if isinstance(name, str) and match(name):
            return True
        for v in req.values():
            if isinstance(v, str) and match(v):
                return True

    return False


def _safe_json(s: str):
    try:
        return json.loads(s)
    except Exception:
        return None


async def _with_retries(
    fn, retries: int = MAX_RETRIES, base: float = 0.5, jitter: float = 0.25
):
    for attempt in range(retries):
        try:
            return await fn()
        except Exception as e:
            if attempt == retries - 1:
                raise
            sleep = base * (2**attempt) + random.uniform(0, jitter)
            logger.warning(f"Transient error: {e} — retrying in {sleep:.2f}s")
            await asyncio.sleep(sleep)


# ----------------------- Grader Class -----------------------
class DualAutograder:
    def __init__(self):
        self.air = Api(AIRTABLE_API_KEY).table(AIRTABLE_BASE_ID, AIRTABLE_TABLE_ID)

        self.gpt_client = AsyncOpenAI(api_key=GPT_API_KEY)
        self.global_sem = asyncio.Semaphore(max(4, PER_KEY_PARALLEL * 2))
        self.gpt_sem = asyncio.Semaphore(PER_KEY_PARALLEL)

        self.stats = {"processed": 0, "graded": 0, "failed": 0, "skipped": 0}

        # System prompt used for GPT-5
        self.system_prompt = (
            "You are an expert grader evaluating solutions against specific criteria.\n"
            "You can see the images/files provided to you.\n"
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

    # ---------------- fetch records ----------------
    def fetch_records(self) -> List[Dict[str, Any]]:
        """Return tasks that request file/image output."""
        fields = [
            FILE_OUTPUT_FIELD,
            REQUESTED_OUTPUTS_FIELD,
            RUBRIC_FIELD,
            PROMPT_FIELD,
            NO_FILE_FIELD,
            GPT_SCORE_FIELD,
            GPT_SUMMARY_FIELD,
            GEM_SCORE_FIELD,
            GEM_SUMMARY_FIELD,
        ]
        recs = self.air.all(view=GENERAL_VIEW_ID, fields=fields)

        filtered = []
        for r in recs:
            f = r.get("fields", {})
            if _has_file_or_image(f.get(REQUESTED_OUTPUTS_FIELD)):
                filtered.append(r)

        logger.info(f"Found {len(filtered)} records requesting file/image output.")
        return filtered

    # ---------------- GPT-5 chat helper (multimodal) ----------------
    async def grade_with_gpt(
        self,
        prompt: str,
        rubric: list,
        attachments: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Use GPT-5 to grade based on the actual files/images.
        We pass the file URLs as image_url parts so the model can see them.
        """
        # Build {criterion_key: description}
        crit_map = {}
        for c in rubric:
            key = list(c.keys())[0]
            desc = (c[key] or {}).get("description", "")
            crit_map[key] = desc

        text_part = (
            "You are grading a model's FILE/IMAGE-based output.\n"
            "You CAN see the attached images/files in this conversation.\n\n"
            "Evaluate the SOLUTION against each CRITERION.\n"
            "For each criterion, return `decision` (true/false) and `reasoning` (exactly 10 sentences).\n\n"
            f"ORIGINAL PROMPT:\n{prompt}\n\n"
            "The attached image(s)/file(s) are the model's solution.\n\n"
            f"CRITERIA (JSON):\n{json.dumps(crit_map, ensure_ascii=False)}\n"
        )

        # Build multimodal content: text + all attachment URLs as images
        user_content: List[Dict[str, Any]] = [{"type": "text", "text": text_part}]
        for att in attachments:
            if not isinstance(att, dict):
                continue
            url = att.get("url")
            if not url:
                continue
            # Let GPT-5 fetch the image/file via URL
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": url},
                }
            )

        messages = [
            {
                "role": "system",
                "content": self.system_prompt,
            },
            {
                "role": "user",
                "content": user_content,
            },
        ]

        async def _call():
            async with self.global_sem, self.gpt_sem:
                return await self.gpt_client.chat.completions.create(
                    model=GPT_MODEL,
                    messages=messages,
                    response_format={"type": "json_object"},
                )

        resp = await _with_retries(_call)
        content = resp.choices[0].message.content or "{}"
        data = _safe_json(content)
        if data is None:
            m = re.search(r"\{.*\}", content, re.DOTALL)
            data = _safe_json(m.group(0)) if m else None

        # If still not JSON, fail with a structured default
        if not isinstance(data, dict):
            data = {}

        # Convert decisions into rubric summary
        decisions = {}
        for key, obj in data.items():
            decision = bool((obj or {}).get("decision"))
            reasoning = (obj or {}).get("reasoning", "") or ""
            decisions[key] = {"decision": decision, "reasoning": reasoning.strip()}

        graded = []
        total_true = 0
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
            total_true += int(ok)

        pct = (total_true / len(rubric) * 100) if rubric else 0.0
        return {
            "percentage": pct,
            "summary": json.dumps(graded, separators=(",", ":")),
        }

    # ---------------- per-record ----------------
    async def process_record(self, rec: Dict[str, Any]):
        self.stats["processed"] += 1
        f = rec.get("fields", {})
        rec_id = rec["id"]

        requested = f.get(REQUESTED_OUTPUTS_FIELD)
        if not _has_file_or_image(requested):
            logger.info(f"{rec_id}: Not a file/image request → skipping.")
            self.stats["skipped"] += 1
            return

        attachments = f.get(FILE_OUTPUT_FIELD)
        if not isinstance(attachments, list) or not attachments:
            logger.info(
                f"{rec_id}: File/image requested but no attachments present → marking {NO_FILE_FIELD}."
            )
            try:
                self.air.update(rec_id, {NO_FILE_FIELD: True})
            except Exception as e:
                logger.error(f"{rec_id}: Failed to mark {NO_FILE_FIELD}: {e}")
            self.stats["skipped"] += 1
            return

        rubric_raw = f.get(RUBRIC_FIELD)
        if not rubric_raw:
            logger.warning(f"{rec_id}: Missing rubric → skipping.")
            self.stats["skipped"] += 1
            return

        rubric = _safe_json(rubric_raw)
        if not isinstance(rubric, list):
            logger.warning(f"{rec_id}: Invalid rubric JSON → skipping.")
            self.stats["skipped"] += 1
            return

        prompt = f.get(PROMPT_FIELD) or ""

        logger.info(f"{rec_id}: Grading file/image output with GPT-5 multimodal...")

        try:
            gpt_res = await self.grade_with_gpt(prompt, rubric, attachments)
        except Exception as e:
            logger.error(f"{rec_id}: GPT-5 grading failed: {e}")
            self.stats["failed"] += 1
            return

        updates = {
            GPT_SCORE_FIELD: gpt_res["percentage"],
            GPT_SUMMARY_FIELD: gpt_res["summary"],
            # Mirror GPT-5 judgement into Gemini autorater columns for file/image tasks
            GEM_SCORE_FIELD: gpt_res["percentage"],
            GEM_SUMMARY_FIELD: gpt_res["summary"],
        }

        try:
            self.air.update(rec_id, updates)
            logger.info(f"{rec_id}: ✅ Updated autorater fields.")
            self.stats["graded"] += 1
        except Exception as e:
            logger.error(f"{rec_id}: Airtable update failed: {e}")
            self.stats["failed"] += 1

    # ---------------- runner ----------------
    async def run(self):
        records = self.fetch_records()
        if not records:
            logger.info("No file/image tasks found.")
            return

        sem = asyncio.Semaphore(6)
        tasks = []

        async def wrapped(r):
            async with sem:
                await self.process_record(r)

        for r in records:
            tasks.append(wrapped(r))

        await asyncio.gather(*tasks)

        logger.info(
            f"🎉 DONE. Processed={self.stats['processed']} "
            f"Graded={self.stats['graded']} Failed={self.stats['failed']} "
            f"Skipped={self.stats['skipped']}"
        )


# ------------------ Entrypoint ------------------
async def main():
    await DualAutograder().run()


if __name__ == "__main__":
    asyncio.run(main())
