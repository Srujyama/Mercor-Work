#!/usr/bin/env python3
"""
Gemini 2.5 Dual Autograder (Text + File/Image) for Eval View Only
-----------------------------------------------------------------

- Uses ONLY Gemini (no GPT).
- Works only on the Eval view (AIRTABLE_VIEW_EVALSET).

Inputs:
    - View: AIRTABLE_VIEW_EVALSET (Eval)
    - Rubric: "Rubric JSON"
    - Prompt: "Consolidated Prompt - 10/25"
    - Text solution: "Consolidated Gemini Response - 10/25"
    - File/Image solution: "Gemini 2.5 Pro Response (File Output)"

Outputs (Gemini-only autorater):
    - "Gemini Autorater - Gemini 2.5 Response Score"   (0–100)
    - "Gemini Autorater - Gemini 2.5 Response Summary" (JSON summary)
"""

import asyncio
import functools
import hashlib
import json
import logging
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pyairtable import Api

# ---------------------------------------------------------------------
# ENV & CONSTANTS
# ---------------------------------------------------------------------

load_dotenv()

AIRTABLE_API_KEY = os.getenv("AIRTABLE_API_KEY")
AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID")
AIRTABLE_TABLE_ID = os.getenv("AIRTABLE_TABLE")
EVAL_VIEW_ID = os.getenv("AIRTABLE_VIEW_EVALSET")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "models/gemini-2.5-pro")

if not AIRTABLE_API_KEY:
    raise RuntimeError("Missing AIRTABLE_API_KEY")
if not AIRTABLE_BASE_ID or not AIRTABLE_TABLE_ID:
    raise RuntimeError("Missing Airtable config")
if not EVAL_VIEW_ID:
    raise RuntimeError("Missing AIRTABLE_VIEW_EVALSET")
if not GOOGLE_API_KEY:
    raise RuntimeError("Missing GOOGLE_API_KEY")

# Input fields
RUBRIC_FIELD = "Rubric JSON"
PROMPT_FIELD = "Consolidated Prompt - 10/25"
TEXT_SOLUTION_FIELD = "Consolidated Gemini Response - 10/25"
FILE_SOLUTION_FIELD = "Gemini 2.5 Pro Response (File Output)"

# Output fields (Gemini-only autorater)
GEM_SCORE_FIELD = "Gemini Autorater - Gemini 2.5 Response Score"
GEM_SUMMARY_FIELD = "Gemini Autorater - Gemini 2.5 Response Summary"

MAX_RETRIES = 4
PER_KEY_PARALLEL = 4

# Cache directory for Gemini file uploads
CACHE_DIR = Path("./gemini_cache_2_5")
CACHE_DIR.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


async def run_in_thread(fn, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, functools.partial(fn, *args, **kwargs))


def safe_json(s):
    try:
        return json.loads(s)
    except Exception:
        return None


async def with_retries(fn):
    for attempt in range(MAX_RETRIES):
        try:
            return await fn()
        except Exception as e:
            msg = str(e)
            if "INVALID_ARGUMENT" in msg or "HTTP/1.1 400" in msg:
                raise
            if attempt == MAX_RETRIES - 1:
                raise
            delay = 0.5 * (2**attempt) + random.random() * 0.25
            logger.warning(f"Retrying after error: {e} (waiting {delay}s)")
            await asyncio.sleep(delay)


def attachment_cache_key(att):
    if "id" in att:
        return att["id"]
    h = hashlib.sha1()
    h.update((att.get("url", "") + att.get("filename", "")).encode())
    return h.hexdigest()


def attachment_cache_path(att):
    key = attachment_cache_key(att)
    ext = Path(att.get("filename", "file")).suffix
    return CACHE_DIR / f"{key}{ext}"


def download_attachment(att, max_bytes=20_000_000):
    url = att.get("url")
    if not url:
        return None

    path = attachment_cache_path(att)
    if path.exists() and path.stat().st_size > 0:
        return path

    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        content = r.content
        if len(content) > max_bytes:
            return None
        path.write_bytes(content)
        return path
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return None


async def download_attachment_async(att):
    return await run_in_thread(download_attachment, att)


def describe_attachments(att_list):
    lines = []
    for a in att_list:
        lines.append(f"- {a.get('filename')} | {a.get('size')} bytes | {a.get('type')}")
    return "\n".join(lines)


# ---------------------------------------------------------------------
# Autograder class
# ---------------------------------------------------------------------


class Gemini25Autograder:
    def __init__(self):
        self.air = Api(AIRTABLE_API_KEY).table(AIRTABLE_BASE_ID, AIRTABLE_TABLE_ID)

        self.gem_client = genai.Client(
            api_key=GOOGLE_API_KEY,
            http_options=types.HttpOptions(api_version="v1beta"),
        )

        self.global_sem = asyncio.Semaphore(6)
        self.gem_sem = asyncio.Semaphore(4)

        self.stats = {"processed": 0, "graded": 0, "failed": 0, "skipped": 0}

        self.system_prompt = (
            "You are an expert grader evaluating the model’s output (text + files).\n"
            "For EACH rubric criterion:\n"
            "1. Produce EXACTLY 10 sentences of reasoning.\n"
            "2. Output a boolean decision.\n"
            "FORMAT:\n"
            "{\n"
            '  "<criterion_key>": {\n'
            '    "decision": true/false,\n'
            '    "reasoning": "10 sentences."\n'
            "  }, ...\n"
            "}"
        )

    # ---------------- FETCH ----------------

    async def fetch_records(self):
        fields = [
            RUBRIC_FIELD,
            PROMPT_FIELD,
            TEXT_SOLUTION_FIELD,
            FILE_SOLUTION_FIELD,
            GEM_SCORE_FIELD,
            GEM_SUMMARY_FIELD,
        ]

        recs = await run_in_thread(self.air.all, view=EVAL_VIEW_ID, fields=fields)

        todo = []
        for r in recs:
            f = r["fields"]
            if not f.get(RUBRIC_FIELD):
                continue
            if not f.get(TEXT_SOLUTION_FIELD) and not f.get(FILE_SOLUTION_FIELD):
                continue
            if (
                f.get(GEM_SCORE_FIELD) is not None
                and f.get(GEM_SUMMARY_FIELD) is not None
            ):
                continue
            todo.append(r)

        logger.info(f"Found {len(todo)} Eval records needing grading.")
        return todo

    # ---------------- GEMINI GRADING ----------------

    async def grade(self, prompt, rubric, text_solution, attachments):
        # Safely build a criterion -> description map
        crit_map: Dict[str, str] = {}
        for c in rubric:
            # Skip completely invalid rubric items
            if not isinstance(c, dict) or not c:
                continue

            # Get the single key for this rubric entry
            key = next(iter(c.keys()))
            meta = c.get(key) or {}  # handle None / missing meta

            if isinstance(meta, dict):
                desc = meta.get("description", "")
            else:
                desc = ""

            crit_map[key] = desc

        attachment_meta = describe_attachments(attachments)

        text_section = f"SOLUTION (TEXT):\n{text_solution}\n\n" if text_solution else ""

        user_prompt = (
            "Grade the model's response.\n\n"
            f"ORIGINAL PROMPT:\n{prompt}\n\n"
            f"{text_section}"
            f"CRITERIA:\n{json.dumps(crit_map)}\n\n"
            f"ATTACHMENTS (metadata):\n{attachment_meta}\n\n"
            "Return ONLY the required JSON."
        )

        # ---- upload attachments (unchanged) ----
        uploaded_parts = []

        async def upload_one(att):
            local = await download_attachment_async(att)
            if not local:
                return None
            mime = att.get("type", "application/octet-stream")
            async with self.gem_sem:
                try:
                    obj = await run_in_thread(
                        self.gem_client.files.upload,
                        file=str(local),
                        config={"mime_type": mime},
                    )
                    return {"fileData": {"fileUri": obj.uri, "mimeType": mime}}
                except Exception as e:
                    logger.error(f"Upload failed: {e}")
                    return None

        tasks = [upload_one(a) for a in attachments]
        for c in asyncio.as_completed(tasks):
            part = await c
            if part:
                uploaded_parts.append(part)

        parts = [{"text": user_prompt}] + uploaded_parts

        async def call_gemini():
            async with self.global_sem:
                return await run_in_thread(
                    self.gem_client.models.generate_content,
                    model=GEMINI_MODEL,
                    contents=[{"role": "user", "parts": parts}],
                    config={
                        "system_instruction": {"parts": [{"text": self.system_prompt}]}
                    },
                )

        resp = await with_retries(call_gemini)

        # ---- robust JSON parsing ----
        output = resp.text or "{}"
        parsed = safe_json(output)

        if not isinstance(parsed, dict):
            # Try to extract a JSON object from the text
            m = re.search(r"\{.*\}", output, re.DOTALL)
            parsed = safe_json(m.group(0)) if m else {}

        if not isinstance(parsed, dict):
            # Last-ditch fallback to avoid NoneType errors
            logger.warning(
                f"Gemini output not dict; falling back to empty dict. raw={output!r}"
            )
            parsed = {}

        # ---- build graded summary ----
        graded: List[Dict[str, Any]] = []
        count = 0

        for c in rubric:
            if not isinstance(c, dict) or not c:
                # Skip weird items instead of crashing
                continue

            key = next(iter(c.keys()))

            meta = c.get(key) or {}
            if not isinstance(meta, dict):
                meta = {}

            # Parsed criterion result from Gemini; always a dict now
            datum = parsed.get(key) or {}
            if not isinstance(datum, dict):
                datum = {}

            decision = bool(datum.get("decision"))
            reasoning = (datum.get("reasoning") or "").strip()

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
                count += 1

        pct = (count / len(graded) * 100) if graded else 0.0

        return pct, json.dumps(graded, separators=(",", ":"))

    # ---------------- PROCESS EACH RECORD ----------------

    async def process_record(self, rec):
        self.stats["processed"] += 1

        f = rec["fields"]
        rec_id = rec["id"]

        rubric = safe_json(f.get(RUBRIC_FIELD))
        if not isinstance(rubric, list):
            self.stats["skipped"] += 1
            return

        text_solution = (f.get(TEXT_SOLUTION_FIELD) or "").strip()
        files = f.get(FILE_SOLUTION_FIELD) or []
        prompt = f.get(PROMPT_FIELD) or ""

        try:
            pct, summary = await self.grade(prompt, rubric, text_solution, files)
        except Exception as e:
            logger.error(f"{rec_id}: grading failed: {e}")
            self.stats["failed"] += 1
            return

        try:
            await run_in_thread(
                self.air.update,
                rec_id,
                {
                    GEM_SCORE_FIELD: pct,
                    GEM_SUMMARY_FIELD: summary,
                },
            )
            self.stats["graded"] += 1
            logger.info(f"{rec_id}: ✔ updated")
        except Exception as e:
            logger.error(f"{rec_id}: update failed: {e}")
            self.stats["failed"] += 1

    # ---------------- RUN ----------------

    async def run(self):
        recs = await self.fetch_records()
        sem = asyncio.Semaphore(5)
        tasks = []

        async def wrap(r):
            async with sem:
                await self.process_record(r)

        for r in recs:
            tasks.append(wrap(r))

        await asyncio.gather(*tasks)

        logger.info(f"DONE → {self.stats}")


# ---------------------------------------------------------------------
# ENTRYPOINT
# ---------------------------------------------------------------------


async def main():
    grader = Gemini25Autograder()
    await grader.run()


if __name__ == "__main__":
    asyncio.run(main())
