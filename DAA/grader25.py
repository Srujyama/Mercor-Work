#!/usr/bin/env python3
"""
ChatGPT-5.1 Dual Autograder (Text + File/Image) for Eval View Only
------------------------------------------------------------------

Replaces Gemini autograder with ChatGPT-5.1.

Inputs:
    - View: AIRTABLE_VIEW_EVALSET (Eval)
    - Rubric: "Rubric JSON"
    - Prompt: "Consolidated Prompt - 10/25"
    - Text solution: "Consolidated Gemini Response - 10/25"
    - File/Image solution: "Gemini 2.5 Pro Response (File Output)"

Outputs:
    - "Gemini Autorater - Gemini 2.5 Response Score"
    - "Gemini Autorater - Gemini 2.5 Response Summary"
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
from typing import Any, Dict, List

import requests
from dotenv import load_dotenv
from openai import OpenAI
from pyairtable import Api

# ---------------------------------------------------------------------
# ENV & CONSTANTS
# ---------------------------------------------------------------------

load_dotenv()

AIRTABLE_API_KEY = os.getenv("AIRTABLE_API_KEY")
AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID")
AIRTABLE_TABLE_ID = os.getenv("AIRTABLE_TABLE")
EVAL_VIEW_ID = os.getenv("AIRTABLE_VIEW_EVALSET")

GPT_API_KEY = os.getenv("GPT_API_KEY")
CHATGPT_MODEL = os.getenv("CHATGPT_MODEL", "gpt-5.1")  # default gpt-5.1

if not AIRTABLE_API_KEY:
    raise RuntimeError("Missing AIRTABLE_API_KEY")
if not AIRTABLE_BASE_ID or not AIRTABLE_TABLE_ID:
    raise RuntimeError("Missing Airtable config")
if not EVAL_VIEW_ID:
    raise RuntimeError("Missing AIRTABLE_VIEW_EVALSET")
if not GPT_API_KEY:
    raise RuntimeError("Missing GPT_API_KEY")

# Airtable fields
RUBRIC_FIELD = "Rubric JSON"
PROMPT_FIELD = "Consolidated Prompt - 10/25"
TEXT_SOLUTION_FIELD = "Consolidated Gemini Response - 10/25"
FILE_SOLUTION_FIELD = "Gemini 2.5 Pro Response (File Output)"

# Output fields (still same names unless you want to rename)
GEM_SCORE_FIELD = "Gemini Autorater - Gemini 2.5 Response Score"
GEM_SUMMARY_FIELD = "Gemini Autorater - Gemini 2.5 Response Summary"

MAX_RETRIES = 4
CACHE_DIR = Path("./chatgpt_autograder_cache")
CACHE_DIR.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------


async def run_in_thread(fn, *args, **kwargs):
    """Run sync functions in executor"""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, functools.partial(fn, *args, **kwargs))


def safe_json(s):
    try:
        return json.loads(s)
    except Exception:
        return None


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
        logger.error(f"Failed to download attachment: {e}")
        return None


async def download_attachment_async(att):
    return await run_in_thread(download_attachment, att)


def describe_attachments(att_list):
    return "\n".join(
        f"- {a.get('filename')} | {a.get('size')} bytes | {a.get('type')}"
        for a in att_list
    )


async def with_retries(fn):
    for attempt in range(MAX_RETRIES):
        try:
            return await fn()
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise
            delay = 0.5 * (2**attempt) + random.random() * 0.25
            logger.warning(f"Retry error: {e} → retrying in {delay:.2f}s")
            await asyncio.sleep(delay)


# ---------------------------------------------------------------------
# Autograder
# ---------------------------------------------------------------------


class ChatGPTAutograder:
    def __init__(self):
        self.air = Api(AIRTABLE_API_KEY).table(AIRTABLE_BASE_ID, AIRTABLE_TABLE_ID)
        self.client = OpenAI(api_key=GPT_API_KEY)

        self.stats = {"processed": 0, "graded": 0, "failed": 0, "skipped": 0}

        # identical to your Gemini system prompt
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

    # ---------------- FETCH RECORDS ----------------

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

        needs = []
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
            needs.append(r)

        logger.info(f"Records requiring grading: {len(needs)}")
        return needs

    # ---------------- GRADING ----------------

    async def grade(self, prompt, rubric, text_solution, attachments):
        # parse rubric into map
        crit_map = {}
        for c in rubric:
            if not isinstance(c, dict):
                continue
            key = next(iter(c.keys()))
            meta = c.get(key) or {}
            crit_map[key] = meta.get("description", "")

        attach_desc = describe_attachments(attachments)
        text_section = f"SOLUTION (TEXT):\n{text_solution}\n\n" if text_solution else ""

        user_prompt = (
            "Grade the model's response.\n\n"
            f"ORIGINAL PROMPT:\n{prompt}\n\n"
            f"{text_section}"
            f"CRITERIA:\n{json.dumps(crit_map)}\n\n"
            f"ATTACHMENTS (metadata):\n{attach_desc}\n\n"
            "Return ONLY the required JSON."
        )

        # upload attachments to OpenAI (files.create)
        uploaded_file_ids = []

        async def upload_one(att):
            local = await download_attachment_async(att)
            if not local:
                return None
            try:
                obj = await run_in_thread(
                    self.client.files.create,
                    file=open(local, "rb"),
                    purpose="assistants",
                )
                return obj.id
            except Exception as e:
                logger.error(f"Upload failed: {e}")
                return None

        upload_tasks = [upload_one(a) for a in attachments]
        for t in asyncio.as_completed(upload_tasks):
            fid = await t
            if fid:
                uploaded_file_ids.append(fid)

        async def call_chatgpt():
            return await run_in_thread(
                self.client.chat.completions.create,
                model=CHATGPT_MODEL,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                # attach files via "file_ids"
                file_ids=uploaded_file_ids,
                temperature=0.0,
            )

        resp = await with_retries(call_chatgpt)

        raw_output = resp.choices[0].message["content"]
        parsed = safe_json(raw_output)

        if not isinstance(parsed, dict):
            m = re.search(r"\{.*\}", raw_output, re.DOTALL)
            parsed = safe_json(m.group(0)) if m else {}

        if not isinstance(parsed, dict):
            parsed = {}

        graded = []
        count = 0

        for c in rubric:
            if not isinstance(c, dict):
                continue

            key = next(iter(c.keys()))
            meta = c.get(key) or {}
            datum = parsed.get(key) or {}

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
            logger.error(f"{rec_id}: Airtable update failed: {e}")
            self.stats["failed"] += 1

    # ---------------- MAIN LOOP ----------------

    async def run(self):
        recs = await self.fetch_records()

        sem = asyncio.Semaphore(5)

        async def wrapper(r):
            async with sem:
                await self.process_record(r)

        await asyncio.gather(*(wrapper(r) for r in recs))

        logger.info(f"DONE → {self.stats}")


# ---------------------------------------------------------------------
# ENTRYPOINT
# ---------------------------------------------------------------------


async def main():
    grader = ChatGPTAutograder()
    await grader.run()


if __name__ == "__main__":
    asyncio.run(main())
