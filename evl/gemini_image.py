#!/usr/bin/env python3
"""
Gemini-Only Autograder for FILE/IMAGE outputs
---------------------------------------------

Rules:
- Only grade tasks where Requested Outputs contains "file" or "image".
- Only use:  "Gemini 3.0 Pro Response (File Output) - Gemini App"
- If that field has no attachments:
    → Set "No Image/file generated" = True
    → Skip grading
- Never use "Gemini 3.0 model responses".
- Grade using ONLY GEMINI (no GPT).
"""

import asyncio
import json
import logging
import os
import random
import re
from typing import Any, Dict, List

from dotenv import load_dotenv
from pyairtable import Api

# Google Gemini client
from google import genai
from google.genai import types


# ----------------------- Load ENV -----------------------
load_dotenv()

AIRTABLE_API_KEY = os.getenv("AIRTABLE_API_KEY")
AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID")
AIRTABLE_TABLE_ID = os.getenv("AIRTABLE_TABLE")
GENERAL_VIEW_ID = os.getenv("AIRTABLE_VIEW_General")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not AIRTABLE_API_KEY or not AIRTABLE_BASE_ID or not AIRTABLE_TABLE_ID:
    raise RuntimeError("Missing Airtable config in .env")
if not GOOGLE_API_KEY:
    raise RuntimeError("Missing GOOGLE_API_KEY in .env")
if not GENERAL_VIEW_ID:
    raise RuntimeError("Missing AIRTABLE_VIEW_General in .env")

# Fields
FILE_OUTPUT_FIELD = "Gemini 3.0 Pro Response (File Output) - Gemini App"
REQUESTED_OUTPUTS_FIELD = "Requested Outputs"
NO_FILE_FIELD = "No Image/file generated"

RUBRIC_FIELD = "Rubric JSON"
PROMPT_FIELD = "Consolidated Prompt - 10/25"

# Gemini autorater output fields
GEM_SCORE_FIELD = "Gemini Autorater - Gemini 3.0 Response Score"
GEM_SUMMARY_FIELD = "Gemini Autorater - Gemini 3.0 Response Summary"

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3-pro-preview")
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "4"))
PARALLELISM = int(os.getenv("PER_KEY_PARALLEL", "4"))

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
            if isinstance(item, dict) and isinstance(item.get("name"), str) and match(
                item["name"]
            ):
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
            logger.warning(f"Transient Gemini error: {e} — retrying in {sleep:.2f}s")
            await asyncio.sleep(sleep)


# ----------------------- Main Autograder -----------------------
class GeminiAutograder:
    def __init__(self):
        self.air = Api(AIRTABLE_API_KEY).table(AIRTABLE_BASE_ID, AIRTABLE_TABLE_ID)

        self.gem_client = genai.Client(
            api_key=GOOGLE_API_KEY,
            http_options=types.HttpOptions(api_version="v1beta"),
        )

        self.sem = asyncio.Semaphore(PARALLELISM)
        self.stats = {"processed": 0, "graded": 0, "failed": 0, "skipped": 0}

        self.system_prompt = (
            "You are an expert grader evaluating images/files.\n"
            "You CAN see the attached images/files.\n"
            "For each criterion:\n"
            "1) Give EXACTLY 10 sentences of reasoning.\n"
            "2) Decide true/false.\n\n"
            "Output ONLY this JSON shape:\n"
            "{\n"
            '  "<criterion_key>": {\n'
            '    "decision": true|false,\n'
            '    "reasoning": "Exactly 10 sentences."\n'
            "  }, ...\n"
            "}"
        )

    # ---------------- Airtable fetch ----------------
    def fetch_records(self):
        fields = [
            FILE_OUTPUT_FIELD,
            REQUESTED_OUTPUTS_FIELD,
            RUBRIC_FIELD,
            PROMPT_FIELD,
            NO_FILE_FIELD,
            GEM_SCORE_FIELD,
            GEM_SUMMARY_FIELD,
        ]
        recs = self.air.all(view=GENERAL_VIEW_ID, fields=fields)

        filtered = []
        for r in recs:
            if _has_file_or_image(r["fields"].get(REQUESTED_OUTPUTS_FIELD)):
                filtered.append(r)

        logger.info(f"Found {len(filtered)} file/image tasks.")
        return filtered

    # ---------------- Gemini grading ----------------
    async def grade_with_gemini(
        self, prompt: str, rubric: list, attachments: List[Dict[str, Any]]
    ) -> Dict[str, Any]:

        # Build criteria map
        crit_map = {list(c.keys())[0]: c[list(c.keys())[0]].get("description", "") for c in rubric}

        # Build text prompt
        text_prompt = (
            "You are grading images/files.\n"
            "Carefully analyze the attached files.\n\n"
            "ORIGINAL PROMPT:\n"
            f"{prompt}\n\n"
            "CRITERIA:\n"
            f"{json.dumps(crit_map, ensure_ascii=False)}\n\n"
            "Return ONLY the required JSON format."
        )

        # Build multimodal contents for Gemini
        parts = [{"text": text_prompt}]
        for att in attachments:
            url = att.get("url")
            mime = att.get("type", "application/octet-stream")
            if url:
                parts.append(
                    {
                        "fileData": {
                            "fileUri": url,
                            "mimeType": mime,
                        }
                    }
                )

        async def _call():
            async with self.sem:
                return self.gem_client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=[{"role": "user", "parts": parts}],
                )

        resp = await _with_retries(_call)

        text = getattr(resp, "text", None)
        if not text:
            # try candidates
            cands = getattr(resp, "candidates", [])
            if cands:
                parts_out = cands[0].content.parts or []
                text = "\n".join(getattr(p, "text", "") for p in parts_out if getattr(p, "text", ""))
        if not text:
            text = "{}"

        data = _safe_json(text)
        if not isinstance(data, dict):
            # Attempt bracket extraction
            m = re.search(r"\{.*\}", text, re.DOTALL)
            data = _safe_json(m.group(0)) if m else {}

        # Convert into rubric summary
        graded = []
        true_count = 0
        for c in rubric:
            key = list(c.keys())[0]
            meta = c[key]
            obj = data.get(key, {})
            decision = bool(obj.get("decision"))
            reasoning = obj.get("reasoning", "")

            graded.append(
                {
                    "autorating": decision,
                    "description": meta.get("description", ""),
                    "weight": meta.get("weight", ""),
                    "criterion_type": meta.get("criterion_type", []),
                    "dependent_criteria": meta.get("dependent_criteria", []),
                    "justification": meta.get("justification", ""),
                    "sources": meta.get("sources", ""),
                    "human_rating": meta.get("human_rating", ""),
                    "reasoning": reasoning,
                }
            )
            if decision:
                true_count += 1

        pct = (true_count / len(graded) * 100) if graded else 0
        summary_json = json.dumps(graded, separators=(",", ":"))

        return {"percentage": pct, "summary": summary_json}

    # ---------------- per record ----------------
    async def process_record(self, rec):
        f = rec["fields"]
        rec_id = rec["id"]
        self.stats["processed"] += 1

        attachments = f.get(FILE_OUTPUT_FIELD)
        if not attachments:
            logger.info(f"{rec_id}: No attachments → marking {NO_FILE_FIELD}.")
            self.air.update(rec_id, {NO_FILE_FIELD: True})
            self.stats["skipped"] += 1
            return

        rubric_raw = f.get(RUBRIC_FIELD)
        rubric = _safe_json(rubric_raw or "")
        if not isinstance(rubric, list):
            logger.warning(f"{rec_id}: Bad rubric → skipping.")
            self.stats["skipped"] += 1
            return

        prompt = f.get(PROMPT_FIELD) or ""

        logger.info(f"{rec_id}: Grading with Gemini...")

        try:
            gem = await self.grade_with_gemini(prompt, rubric, attachments)
        except Exception as e:
            logger.error(f"{rec_id}: Gemini grading failed: {e}")
            self.stats["failed"] += 1
            return

        updates = {
            GEM_SCORE_FIELD: gem["percentage"],
            GEM_SUMMARY_FIELD: gem["summary"],
        }

        try:
            self.air.update(rec_id, updates)
            logger.info(f"{rec_id}: ✅ Updated Gemini autorater fields.")
            self.stats["graded"] += 1
        except Exception as e:
            logger.error(f"{rec_id}: Airtable update failed: {e}")
            self.stats["failed"] += 1

    # ---------------- runner ----------------
    async def run(self):
        records = self.fetch_records()
        if not records:
            logger.info("No eligible tasks.")
            return

        tasks = []
        for r in records:
            tasks.append(self.process_record(r))

        await asyncio.gather(*tasks)

        logger.info(
            f"🎉 DONE. Processed={self.stats['processed']} "
            f"Graded={self.stats['graded']} Failed={self.stats['failed']} "
            f"Skipped={self.stats['skipped']}"
        )


# ---------------- Entrypoint ----------------
async def main():
    await GeminiAutograder().run()


if __name__ == "__main__":
    asyncio.run(main())
