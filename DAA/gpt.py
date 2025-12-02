#!/usr/bin/env python3
"""
GPT-5.1 Autograder (Text + Images + PDFs)
Corrected + Fully Hardened Version
OpenAI SDK 2.8.1 — Assistants API
"""

import asyncio
import functools
import json
import logging
import os
import re
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv
from openai import OpenAI
from pyairtable import Api
from tqdm import tqdm

# ---------------------------------------------------------------------
# ENV & CONSTANTS
# ---------------------------------------------------------------------

load_dotenv()

AIRTABLE_API_KEY = os.getenv("AIRTABLE_API_KEY")
AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID")
AIRTABLE_TABLE_ID = os.getenv("AIRTABLE_TABLE")
EVAL_VIEW_ID = os.getenv("AIRTABLE_VIEW_EVALSET")
GPT_API_KEY = os.getenv("GPT_API_KEY")
MODEL = os.getenv("CHATGPT_MODEL", "gpt-4o-mini")

if not AIRTABLE_API_KEY:
    raise RuntimeError("Missing AIRTABLE_API_KEY")
if not AIRTABLE_BASE_ID or not AIRTABLE_TABLE_ID:
    raise RuntimeError("Missing Airtable config")
if not EVAL_VIEW_ID:
    raise RuntimeError("Missing AIRTABLE_VIEW_EVALSET")
if not GPT_API_KEY:
    raise RuntimeError("Missing GPT_API_KEY")

RUBRIC_FIELD = "Rubric JSON"
TEXT_SOLUTION_FIELD = "Consolidated Gemini Response - 10/25"
FILE_SOLUTION_FIELD = "Gemini 2.5 Pro Response (File Output)"

GEM_SCORE_FIELD = "GPT Autorater - Gemini 2.5 Response Score"
GEM_SUMMARY_FIELD = "GPT Autorater - Gemini 2.5 Response Summary"

CACHE_DIR = Path("./attachment_cache")
CACHE_DIR.mkdir(exist_ok=True)

MAX_RETRIES = 4
PARALLEL = 8

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 🔇 Brutally silence httpx noise
httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)
httpx_logger.disabled = True
openai_logger = logging.getLogger("openai")
openai_logger.setLevel(logging.WARNING)

# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------


async def run_in_thread(fn, *args, **kwargs):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, functools.partial(fn, *args, **kwargs))


def safe_json(s: str):
    try:
        return json.loads(s)
    except Exception:
        return None


async def retry(coro):
    for i in range(MAX_RETRIES):
        try:
            return await coro()
        except Exception:
            if i == MAX_RETRIES - 1:
                raise
            await asyncio.sleep(0.5 * (2**i))


def is_image(att: dict) -> bool:
    name = att.get("filename", "").lower()
    return name.endswith((".png", ".jpg", ".jpeg", ".webp", ".gif"))


def cache_path(att: dict):
    url = att.get("url")
    ext = Path(att.get("filename", "file")).suffix
    key = str(abs(hash(url)))[:12]
    return CACHE_DIR / f"{key}{ext}"


def download(att: dict, max_bytes=50_000_000) -> Optional[Path]:
    url = att.get("url")
    if not url:
        return None

    target = cache_path(att)
    if target.exists() and target.stat().st_size > 0:
        return target

    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        if len(r.content) > max_bytes:
            return None
        target.write_bytes(r.content)
        return target
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return None


async def download_async(att):
    return await run_in_thread(download, att)


# ---------------------------------------------------------------------
# MAIN AUTOGRADER
# ---------------------------------------------------------------------


class Autograder:
    def __init__(self):
        self.client = OpenAI(api_key=GPT_API_KEY)
        self.air = Api(AIRTABLE_API_KEY).table(AIRTABLE_BASE_ID, AIRTABLE_TABLE_ID)

        logger.info("Creating assistant…")
        self.assistant = self.client.beta.assistants.create(
            name="GPT Autograder",
            model=MODEL,
            instructions=(
                "You are an expert grader.\n"
                "For each rubric criterion:\n"
                "1. Provide EXACTLY 10 sentences of reasoning.\n"
                "2. Provide a boolean decision.\n"
                "Output ONLY valid JSON.\n"
                "Never output lists unless explicitly required.\n"
                "Always output objects with 'decision' and 'reasoning'."
            ),
            tools=[],  # no tools; we are not using file_search / code_interpreter
        )
        logger.info(f"Assistant created: {self.assistant.id}")

    # --------------------------------------------------------------

    async def fetch_records(self):
        fields = [
            RUBRIC_FIELD,
            TEXT_SOLUTION_FIELD,
            FILE_SOLUTION_FIELD,
            GEM_SCORE_FIELD,
            GEM_SUMMARY_FIELD,
        ]

        recs = await run_in_thread(self.air.all, view=EVAL_VIEW_ID, fields=fields)

        todo = []
        for r in recs:
            f = r.get("fields", {})
            if not f.get(RUBRIC_FIELD):
                continue
            if not (f.get(TEXT_SOLUTION_FIELD) or f.get(FILE_SOLUTION_FIELD)):
                continue
            if (
                f.get(GEM_SCORE_FIELD) is not None
                and f.get(GEM_SUMMARY_FIELD) is not None
            ):
                continue
            todo.append(r)

        logger.info(f"Records needing grading: {len(todo)}")
        return todo

    # --------------------------------------------------------------

    async def upload_file(self, path: Path) -> str:
        logger.info(f"Uploading: {path}")
        resp = await run_in_thread(
            self.client.files.create, file=open(path, "rb"), purpose="assistants"
        )
        return resp.id

    # --------------------------------------------------------------

    async def grade_record(self, rec):
        fields = rec["fields"]
        rec_id = rec["id"]

        # ---- Parse rubric ----
        rubric_raw = fields.get(RUBRIC_FIELD)
        rubric = safe_json(rubric_raw)
        if not isinstance(rubric, list):
            logger.error(f"{rec_id}: invalid rubric JSON")
            return

        # ---- Model output ----
        text_solution = fields.get(TEXT_SOLUTION_FIELD, "")

        # Normalize attachments field: if empty/None/"" → []
        attachments_raw = fields.get(FILE_SOLUTION_FIELD)
        attachments = attachments_raw or []

        # ---- Build criteria dict ----
        crit_map = {
            list(c.keys())[0]: (c[list(c.keys())[0]] or {}).get("description", "")
            for c in rubric
        }

        # ---- Build user prompt ----
        user_prompt = (
            "Grade the following model output.\n\n"
            f"MODEL OUTPUT (text):\n{text_solution}\n\n"
            "CRITERIA:\n"
            f"{json.dumps(crit_map)}\n"
            "Return ONLY valid JSON.\n"
        )

        # ---- Build content blocks ----
        content = [{"type": "text", "text": user_prompt}]

        # ------------------------------------------------------
        # Process attachments ONLY for images.
        # Non-image files (PDFs, etc.) are ignored to avoid
        # Assistants attachments/tools requirements.
        # ------------------------------------------------------
        if attachments:
            image_atts = [att for att in attachments if is_image(att)]

            if image_atts:
                local_files = await asyncio.gather(
                    *[asyncio.create_task(download_async(att)) for att in image_atts]
                )

                for path, att in zip(local_files, image_atts):
                    if not path:
                        continue
                    file_id = await self.upload_file(path)

                    content.append(
                        {
                            "type": "image_file",
                            "image_file": {
                                "file_id": file_id,
                            },
                        }
                    )

        # ---- Create thread ----
        thread = await run_in_thread(self.client.beta.threads.create)

        # ---- Send message (no attachments param at all) ----
        await run_in_thread(
            self.client.beta.threads.messages.create,
            thread_id=thread.id,
            role="user",
            content=content,
        )

        # ---- Run assistant ----
        run = await run_in_thread(
            self.client.beta.threads.runs.create,
            thread_id=thread.id,
            assistant_id=self.assistant.id,
        )

        # ---- Poll until done (chill: 5s) ----
        while True:
            run = await run_in_thread(
                self.client.beta.threads.runs.retrieve,
                thread_id=thread.id,
                run_id=run.id,
            )
            if run.status in ("completed", "failed", "cancelled"):
                break
            await asyncio.sleep(5)

        if run.status != "completed":
            logger.error(f"{rec_id}: run failed ({run.status})")
            return

        # ---- Retrieve assistant output ----
        msgs = await run_in_thread(
            self.client.beta.threads.messages.list, thread_id=thread.id
        )

        assistant_msg = next((m for m in msgs.data if m.role == "assistant"), None)
        if not assistant_msg:
            logger.error(f"{rec_id}: no assistant output")
            return

        raw = assistant_msg.content[0].text.value
        parsed = safe_json(raw)

        if not isinstance(parsed, dict):
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            parsed = safe_json(m.group(0)) if m else {}

        if not isinstance(parsed, dict):
            logger.error(f"{rec_id}: assistant produced invalid JSON")
            return

        # -----------------------------------------------------------------
        # FINAL HARDENED SCHEMA NORMALIZATION
        # -----------------------------------------------------------------

        graded = []
        score = 0

        for c in rubric:
            key = list(c.keys())[0]
            meta = c[key]

            item = parsed.get(key, {})

            # Normalize item into dict form
            if isinstance(item, dict):
                raw_decision = item.get("decision", False)
                raw_reasoning = item.get("reasoning", "")
            elif isinstance(item, bool):
                raw_decision = item
                raw_reasoning = ""
            else:
                raw_decision = False
                raw_reasoning = ""

            # Normalize reasoning into string
            if isinstance(raw_reasoning, list):
                flat = []
                for x in raw_reasoning:
                    if isinstance(x, list):
                        flat.extend(x)
                    else:
                        flat.append(x)
                reasoning = " ".join(str(x) for x in flat)
            else:
                reasoning = str(raw_reasoning)

            reasoning = reasoning.replace("\n", " ").strip()
            decision = bool(raw_decision)

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
                score += 1

        pct = score * 100 / len(graded) if graded else 0
        summary = json.dumps(graded, separators=(",", ":"))

        # ---- Save results ----
        try:
            await run_in_thread(
                self.air.update,
                rec_id,
                {GEM_SCORE_FIELD: pct, GEM_SUMMARY_FIELD: summary},
            )
            logger.info(f"{rec_id}: ✓ updated")
        except Exception as e:
            logger.error(f"{rec_id}: Airtable update failed: {e}")

    # --------------------------------------------------------------

    async def run(self):
        records = await self.fetch_records()

        if not records:
            logger.info("No records to grade.")
            return

        sem = asyncio.Semaphore(PARALLEL)

        pbar = tqdm(total=len(records), desc="Grading records", unit="record")

        async def worker(rec):
            nonlocal pbar
            async with sem:
                await self.grade_record(rec)
                pbar.update(1)

        try:
            await asyncio.gather(*(worker(r) for r in records))
        finally:
            pbar.close()


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------


async def main():
    grader = Autograder()
    await grader.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting cleanly.")
