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

Behavior:
    - For each record in the Eval view:
        * If Rubric JSON is present AND
          (text solution OR file/image attachments are present) AND
          at least one of the two Gemini autorater fields is missing:
            → Call Gemini with:
                - Text solution (if present)
                - Uploaded files (if present)
            → For each rubric criterion, Gemini returns:
                - decision: true/false
                - reasoning: exactly 10 sentences
            → We compute a percentage score and store a detailed JSON summary.

Environment variables needed:
    - AIRTABLE_API_KEY
    - AIRTABLE_BASE_ID
    - AIRTABLE_TABLE
    - AIRTABLE_VIEW_EVALSET
    - GOOGLE_API_KEY
    - GEMINI_MODEL (optional, defaults to "gemini-3-pro-preview")
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
AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID", "appgeueGlH9mCUTvu")
AIRTABLE_TABLE_ID = os.getenv("AIRTABLE_TABLE", "tblfy3EPxl1PHvKV7")
EVAL_VIEW_ID = os.getenv("AIRTABLE_VIEW_EVALSET")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not AIRTABLE_API_KEY:
    raise RuntimeError("Missing AIRTABLE_API_KEY in .env")
if not AIRTABLE_BASE_ID or not AIRTABLE_TABLE_ID:
    raise RuntimeError("Missing Airtable base/table config in .env")
if not EVAL_VIEW_ID:
    raise RuntimeError("Missing AIRTABLE_VIEW_EVALSET (Eval view) in .env")
if not GOOGLE_API_KEY:
    raise RuntimeError("Missing GOOGLE_API_KEY in .env")

# Input fields
RUBRIC_FIELD = "Rubric JSON"
PROMPT_FIELD = "Consolidated Prompt - 10/25"
TEXT_SOLUTION_FIELD = "Consolidated Gemini Response - 10/25"
FILE_SOLUTION_FIELD = "Gemini 2.5 Pro Response (File Output)"

# Output fields (Gemini-only autorater)
GEM_SCORE_FIELD = "Gemini Autorater - Gemini 2.5 Response Score"
GEM_SUMMARY_FIELD = "Gemini Autorater - Gemini 2.5 Response Summary"

# Models & tunables
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3-pro-preview")
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "4"))
PER_KEY_PARALLEL = int(os.getenv("PER_KEY_PARALLEL", "4"))

# Gemini cache dir for attachments
GEMINI_CACHE_DIR = Path(os.getenv("GEMINI_CACHE_DIR", "./gemini_cache_2_5"))
GEMINI_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Helpers (async, cache, retries)
# ---------------------------------------------------------------------

async def run_in_thread(fn, *args, **kwargs):
    """Run a blocking function in a thread pool."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, functools.partial(fn, *args, **kwargs))


def _safe_json_loads(s: str):
    try:
        return json.loads(s)
    except Exception:
        return None


async def _with_retries(
    fn,
    retries: int = MAX_RETRIES,
    base: float = 0.5,
    jitter: float = 0.25,
):
    """
    Generic retry helper for async functions.

    - Retries on transient errors.
    - Treats clear client/parameter issues (400/INVALID_ARGUMENT) as fatal.
    """
    for attempt in range(retries):
        try:
            return await fn()
        except Exception as e:
            msg = str(e)
            if (
                "INVALID_ARGUMENT" in msg
                or "HTTP/1.1 400" in msg
                or "invalid_request_error" in msg
            ):
                # Permanent failure → do not retry
                raise

            if attempt == retries - 1:
                raise

            sleep = base * (2**attempt) + random.uniform(0, jitter)
            logger.warning(f"Transient error: {e} — retrying in {sleep:.2f}s")
            await asyncio.sleep(sleep)


# ---- Attachment caching helpers ----

def _attachment_cache_key(att: Dict[str, Any]) -> str:
    """
    Build a stable cache key per Airtable attachment.

    Prefer attachment 'id'; fall back to a SHA1 of (url + filename).
    """
    att_id = att.get("id")
    if att_id:
        return att_id

    h = hashlib.sha1()
    url = (att.get("url") or "").encode("utf-8", errors="ignore")
    name = (att.get("filename") or "").encode("utf-8", errors="ignore")
    h.update(url + b"|" + name)
    return h.hexdigest()


def _attachment_cache_path(att: Dict[str, Any]) -> Path:
    """
    Return the path under GEMINI_CACHE_DIR where this attachment should live.
    We preserve the extension when possible.
    """
    key = _attachment_cache_key(att)
    filename = att.get("filename") or "file"
    ext = Path(filename).suffix or ""
    return GEMINI_CACHE_DIR / f"{key}{ext}"


def _download_attachment_to_cache(
    att: Dict[str, Any],
    max_bytes: int = 20 * 1024 * 1024,
) -> Optional[Path]:
    """
    Download an Airtable attachment to the cache (if not already present).

    Returns:
        Path to cached file, or None on failure/oversize.
    """
    url = att.get("url")
    if not url:
        return None

    target = _attachment_cache_path(att)

    # If it’s already cached and non-empty, reuse it
    if target.exists() and target.stat().st_size > 0:
        logger.info(f"[CACHE] Reusing cached file: {target}")
        return target

    logger.info(f"[CACHE] Downloading attachment from {url} -> {target}")

    try:
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        content = resp.content
    except Exception as e:
        logger.error(f"[CACHE] Error downloading {url}: {e}")
        return None

    if len(content) > max_bytes:
        logger.warning(
            f"[CACHE] Skipping download for {url}: {len(content)} bytes > {max_bytes}"
        )
        return None

    try:
        target.write_bytes(content)
    except Exception as e:
        logger.error(f"[CACHE] Failed to write cache file {target}: {e}")
        return None

    logger.info(f"[CACHE] Saved {len(content)} bytes to {target}")
    return target


async def download_attachment_to_cache_async(
    att: Dict[str, Any],
    max_bytes: int = 20 * 1024 * 1024,
) -> Optional[Path]:
    """Async wrapper around _download_attachment_to_cache (runs in thread)."""
    return await run_in_thread(_download_attachment_to_cache, att, max_bytes)


def _describe_attachments_for_text(attachments: List[Dict[str, Any]]) -> str:
    """
    Build a simple textual description of attachments for inclusion
    in Gemini prompts.
    """
    lines = []
    for att in attachments:
        if not isinstance(att, dict):
            continue
        fname = att.get("filename", "file")
        mime = att.get("type", "application/octet-stream")
        size = att.get("size", "unknown")
        lines.append(f"- {fname} (mime={mime}, size={size})")
    return "\n".join(lines)


# ---------------------------------------------------------------------
# Gemini-only Dual Autograder
# ---------------------------------------------------------------------

class Gemini25DualAutograder:
    """
    Single autograder class that:
      - Reads records from the Eval view
      - Uses both text + files as the "solution"
      - Grades only with Gemini
      - Writes to Gemini 2.5 autorater fields
    """

    def __init__(self):
        # Airtable
        self.air = Api(AIRTABLE_API_KEY).table(AIRTABLE_BASE_ID, AIRTABLE_TABLE_ID)

        # Gemini client
        self.gem_client = genai.Client(
            api_key=Google_API_KEY := GOOGLE_API_KEY,
            http_options=types.HttpOptions(api_version="v1beta"),
        )

        # Concurrency
        self.global_sem = asyncio.Semaphore(max(4, PER_KEY_PARALLEL * 2))
        self.gem_sem = asyncio.Semaphore(PER_KEY_PARALLEL)

        self.stats = {"processed": 0, "graded": 0, "failed": 0, "skipped": 0}

        # System prompt specifically for Gemini
        self.gem_system_prompt = (
            "You are an expert grader evaluating a model's response "
            "that may include TEXT and/or FILE/IMAGE outputs.\n"
            "You CAN see the attached files/images.\n\n"
            "For EACH rubric criterion, do two things:\n"
            "1) Produce EXACTLY 10 sentences of reasoning explaining your evaluation.\n"
            "2) Output a boolean decision whether the solution meets the criterion.\n\n"
            "IMPORTANT OUTPUT FORMAT (single JSON object):\n"
            "{\n"
            '  \"<criterion_key>\": {\n'
            '    \"decision\": true|false,\n'
            '    \"reasoning\": \"Exactly 10 sentences.\"\n'
            "  }, ...\n"
            "}\n"
            "No extra keys, no markdown, no code fences."
        )

    # ----------------- Fetch records from Eval view -----------------

    async def fetch_records(self) -> List[Dict[str, Any]]:
        fields = [
            RUBRIC_FIELD,
            PROMPT_FIELD,
            TEXT_SOLUTION_FIELD,
            FILE_SOLUTION_FIELD,
            GEM_SCORE_FIELD,
            GEM_SUMMARY_FIELD,
        ]

        logger.info(f"Fetching records from Eval view: {EVAL_VIEW_ID}")
        recs = await run_in_thread(
            self.air.all,
            view=EVAL_VIEW_ID,
            fields=fields,
        )

        filtered = []
        for r in recs:
            f = r.get("fields", {})
            rubric_raw = f.get(RUBRIC_FIELD)
            text_sol = (f.get(TEXT_SOLUTION_FIELD) or "").strip()
            files = f.get(FILE_SOLUTION_FIELD)

            # Need a rubric + some solution (text or files)
            if not rubric_raw:
                continue
            if not text_sol and (not isinstance(files, list) or not files):
                continue

            # If outputs already present, skip
            if f.get(GEM_SCORE_FIELD) is not None and f.get(GEM_SUMMARY_FIELD) is not None:
                continue

            filtered.append(r)

        logger.info(f"Found {len(filtered)} records needing Gemini 2.5 grading.")
        return filtered

    # ----------------- Gemini grading logic -----------------

    async def grade_with_gemini(
        self,
        prompt: str,
        rubric: List[Dict[str, Any]],
        text_solution: str,
        attachments: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Main Gemini grading call.
        Uses:
          - text_solution (if non-empty)
          - file attachments (if any) uploaded to Gemini
        """

        # Build {criterion_key: description} map
        crit_map: Dict[str, str] = {}
        for c in rubric:
            key = list(c.keys())[0]
            desc = (c[key] or {}).get("description", "")
            crit_map[key] = desc

        # Attachments textual metadata (helps model link context)
        file_desc = _describe_attachments_for_text(attachments)

        # Build textual prompt for Gemini
        text_solution_section = (
            f"SOLUTION (TEXT):\n{text_solution}\n\n" if text_solution else ""
        )

        text_prompt = (
            "You are grading a model's response that may contain TEXT and/or "
            "FILE/IMAGE outputs.\n"
            "The attached files/images (if any) are part of the solution. "
            "You can see them.\n\n"
            f"ORIGINAL PROMPT:\n{prompt}\n\n"
            f"{text_solution_section}"
            "CRITERIA (JSON):\n"
            f"{json.dumps(crit_map, ensure_ascii=False)}\n\n"
            "ATTACHMENTS (metadata only, the actual files are uploaded separately):\n"
            f"{file_desc}\n\n"
            "Return ONLY the required JSON object with decisions and reasoning."
        )

        # ---- Upload attachments to Gemini ----

        async def handle_attachment(att: Dict[str, Any]):
            if not isinstance(att, dict):
                return None

            local_path = await download_attachment_to_cache_async(att)
            if not local_path:
                return None

            mime = att.get("type", "application/octet-stream")

            async with self.gem_sem:
                logger.info(f"[GEMINI] Uploading {local_path} (mime={mime})")
                try:
                    f_obj = await run_in_thread(
                        self.gem_client.files.upload,
                        file=str(local_path),
                        config={"mime_type": mime},
                    )
                except Exception as e:
                    logger.error(f"[GEMINI] Upload failed for {local_path}: {e}")
                    return None

            uri = getattr(f_obj, "uri", None)
            if not uri:
                logger.warning(
                    f"[GEMINI] Uploaded file missing uri; skipping {local_path}"
                )
                return None

            return {
                "fileData": {
                    "fileUri": uri,
                    "mimeType": mime,
                }
            }

        uploaded_parts: List[Dict[str, Any]] = []
        upload_tasks = [
            asyncio.create_task(handle_attachment(att)) for att in attachments or []
        ]

        for task in asyncio.as_completed(upload_tasks):
            part = await task
            if part:
                uploaded_parts.append(part)

        # Build parts for Gemini call
        parts = [{"text": text_prompt}] + uploaded_parts

        async def _call():
            async with self.global_sem:
                return await run_in_thread(
                    self.gem_client.models.generate_content,
                    model=GEMINI_MODEL,
                    contents=[{"role": "user", "parts": parts}],
                    config={
                        "system_instruction": {
                            "parts": [{"text": self.gem_system_prompt}]
                        }
                    },
                )

        resp = await _with_retries(_call)

        # Extract plain text from Gemini response
        text_out = getattr(resp, "text", None)
        if not text_out:
            cands = getattr(resp, "candidates", [])
            if cands and getattr(cands[0], "content", None):
                out_parts = cands[0].content.parts or []
                text_out = "\n".join(
                    getattr(p, "text", "") for p in out_parts if getattr(p, "text", "")
                )
        if not text_out:
            text_out = "{}"

        # Parse JSON (allow for extra junk, so we regex the first {...})
        data = _safe_json_loads(text_out)
        if not isinstance(data, dict):
            m = re.search(r"\{.*\}", text_out, re.DOTALL)
            data = _safe_json_loads(m.group(0)) if m else {}

        if not isinstance(data, dict):
            logger.warning("Gemini response could not be parsed as JSON; using empty dict.")
            data = {}

        # Convert into rubric summary payload
        graded = []
        true_count = 0

        for c in rubric:
            key = list(c.keys())[0]
            meta = c[key]
            obj = data.get(key, {})
            decision = bool(obj.get("decision"))
            reasoning = (obj.get("reasoning") or "").strip()

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

        pct = (true_count / len(graded) * 100) if graded else 0.0
        summary_json = json.dumps(graded, separators=(",", ":"))

        return {"percentage": pct, "summary": summary_json}

    # ----------------- Per-record processing -----------------

    async def process_record(self, rec: Dict[str, Any]):
        self.stats["processed"] += 1
        rec_id = rec["id"]
        f = rec.get("fields", {})

        rubric_raw = f.get(RUBRIC_FIELD)
        text_solution = (f.get(TEXT_SOLUTION_FIELD) or "").strip()
        attachments = f.get(FILE_SOLUTION_FIELD) or []
        prompt = f.get(PROMPT_FIELD) or ""

        if not rubric_raw:
            logger.info(f"{rec_id}: Missing rubric → skipping.")
            self.stats["skipped"] += 1
            return

        rubric = _safe_json_loads(rubric_raw)
        if not isinstance(rubric, list):
            logger.info(f"{rec_id}: Invalid rubric JSON → skipping.")
            self.stats["skipped"] += 1
            return

        if not text_solution and (not isinstance(attachments, list) or not attachments):
            logger.info(f"{rec_id}: No text or file solution → skipping.")
            self.stats["skipped"] += 1
            return

        logger.info(f"{rec_id}: Grading with Gemini 2.5 (text + files if any)...")

        try:
            result = await self.grade_with_gemini(
                prompt=prompt,
                rubric=rubric,
                text_solution=text_solution,
                attachments=attachments if isinstance(attachments, list) else [],
            )
        except Exception as e:
            logger.error(f"{rec_id}: Gemini grading failed: {e}")
            self.stats["failed"] += 1
            return

        updates = {
            GEM_SCORE_FIELD: result["percentage"],
            GEM_SUMMARY_FIELD: result["summary"],
        }

        try:
            await run_in_thread(self.air.update, rec_id, updates)
            logger.info(f"{rec_id}: ✅ Updated Gemini 2.5 autorater fields.")
            self.stats["graded"] += 1
        except Exception as e:
            logger.error(f"{rec_id}: Airtable update failed: {e}")
            self.stats["failed"] += 1

    # ----------------- Runner -----------------

    async def run(self):
        records = await self.fetch_records()
        if not records:
            logger.info("No records need Gemini 2.5 grading for Eval view.")
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
            f"🎉 DONE (Eval view). Processed={self.stats['processed']} "
            f"Graded={self.stats['graded']} Failed={self.stats['failed']} "
            f"Skipped={self.stats['skipped']}"
        )


# ---------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------

async def main():
    grader = Gemini25DualAutograder()
    await grader.run()


if __name__ == "__main__":
    asyncio.run(main())
