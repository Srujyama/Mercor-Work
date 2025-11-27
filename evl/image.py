#!/usr/bin/env python3
"""
Dual Autograder (GPT-5 + Gemini) for FILE/IMAGE outputs ONLY
-------------------------------------------------------------

Improved version with better parallelization:
- Uses asyncio + semaphores for concurrency control.
- Runs blocking IO (Airtable, requests, Gemini Python client) in thread pool.
- GPT-5 and Gemini grading for each record run in parallel.
- Attachment downloads + Gemini file uploads are parallelized per record.

Rules:
- Only grade tasks where Requested Outputs contains "file" or "image".
- Only use:  "Gemini 3.0 Pro Response (File Output) - Gemini App" as the source of files.
- If that field has no attachments:
    → Set "No Image/file generated" = True
    → Skip grading

Behavior:
- GPT-5 grades into GPT autorater columns ONLY:
    * GPT5 Autorater - Gemini 3.0 Response Score
    * GPT5 Autorater - Gemini 3.0 Response Summary
- Gemini grades into Gemini autorater columns ONLY:
    * Gemini Autorater - Gemini 3.0 Response Score
    * Gemini Autorater - Gemini 3.0 Response Summary

- GPT-5:
    * Uses image_url with Airtable file URLs.
    * If the primary GPT model returns a 400 invalid_image_* error,
      it optionally switches to a fallback GPT model (OPENAI_FALLBACK_MODEL)
      and retries once.

- Gemini:
    * Downloads Airtable attachments into a local cache folder (GEMINI_CACHE_DIR).
    * Uploads cached files to Gemini via client.files.upload().
    * Uses the returned fileUri in generate_content().
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
from openai import AsyncOpenAI
from pyairtable import Api

# ----------------------- Load ENV -----------------------
load_dotenv()

AIRTABLE_API_KEY = os.getenv("AIRTABLE_API_KEY")
AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID")
AIRTABLE_TABLE_ID = os.getenv("AIRTABLE_TABLE")
GENERAL_VIEW_ID = os.getenv("AIRTABLE_VIEW_General")

GPT_API_KEY = os.getenv("GPT_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not AIRTABLE_API_KEY or not AIRTABLE_BASE_ID or not AIRTABLE_TABLE_ID:
    raise RuntimeError("Missing Airtable config in .env")
if not GPT_API_KEY:
    raise RuntimeError("Missing GPT_API_KEY in .env")
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

# Autorater output fields
GPT_SCORE_FIELD = "GPT5 Autorater - Gemini 3.0 Response Score"
GPT_SUMMARY_FIELD = "GPT5 Autorater - Gemini 3.0 Response Summary"
GEM_SCORE_FIELD = "Gemini Autorater - Gemini 3.0 Response Score"
GEM_SUMMARY_FIELD = "Gemini Autorater - Gemini 3.0 Response Summary"

# Models
GPT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5")
GPT_FALLBACK_MODEL = os.getenv("OPENAI_FALLBACK_MODEL", "")  # optional
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3-pro-preview")

MAX_RETRIES = int(os.getenv("MAX_RETRIES", "4"))
PER_KEY_PARALLEL = int(os.getenv("PER_KEY_PARALLEL", "4"))

# Gemini cache dir
GEMINI_CACHE_DIR = Path(os.getenv("GEMINI_CACHE_DIR", "./gemini_cache"))
GEMINI_CACHE_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ----------------------- Generic async helpers -----------------------
async def run_in_thread(fn, *args, **kwargs):
    """Run a blocking function in a thread pool."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, functools.partial(fn, *args, **kwargs))


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
    """
    Retry helper.

    - Retries on transient errors (timeouts, 5xx, connection issues).
    - Treats clear client/parameter issues (400 INVALID_ARGUMENT, invalid_image_*)
      as fatal and does NOT retry.
    """
    for attempt in range(retries):
        try:
            return await fn()
        except Exception as e:
            msg = str(e)

            # Non-retriable: bad request / invalid input
            if (
                "INVALID_ARGUMENT" in msg
                or "invalid_request_error" in msg
                or "invalid_image_format" in msg
                or "invalid_image_url" in msg
                or "HTTP/1.1 400" in msg
            ):
                raise

            if attempt == retries - 1:
                raise

            sleep = base * (2**attempt) + random.uniform(0, jitter)
            logger.warning(f"Transient error: {e} — retrying in {sleep:.2f}s")
            await asyncio.sleep(sleep)


# ---- Gemini cache helpers ----
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
    att: Dict[str, Any], max_bytes: int = 20 * 1024 * 1024
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
    att: Dict[str, Any], max_bytes: int = 20 * 1024 * 1024
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


# ----------------------- Grader Class -----------------------
class DualAutograder:
    def __init__(self):
        self.air = Api(AIRTABLE_API_KEY).table(AIRTABLE_BASE_ID, AIRTABLE_TABLE_ID)

        # GPT client
        self.gpt_client = AsyncOpenAI(api_key=GPT_API_KEY)
        self.global_sem = asyncio.Semaphore(max(4, PER_KEY_PARALLEL * 2))
        self.gpt_sem = asyncio.Semaphore(PER_KEY_PARALLEL)

        # Gemini client (sync SDK but called via run_in_thread)
        self.gem_client = genai.Client(
            api_key=GOOGLE_API_KEY,
            http_options=types.HttpOptions(api_version="v1beta"),
        )
        self.gem_sem = asyncio.Semaphore(int(os.getenv("GEMINI_PARALLEL", "4")))

        self.stats = {"processed": 0, "graded": 0, "failed": 0, "skipped": 0}

        # System prompt for GPT-5
        self.gpt_system_prompt = (
            "You are an expert grader evaluating solutions against specific criteria.\n"
            "You can see the images/files provided to you.\n"
            "For EACH criterion, do two things:\n"
            "1) Produce EXACTLY 10 sentences of reasoning explaining your evaluation.\n"
            "2) Output a boolean decision whether the solution meets the criterion.\n\n"
            "IMPORTANT OUTPUT FORMAT:\n"
            "{\n"
            '  "<criterion_key>": {\n'
            '    "decision": true|false,\n'
            '    "reasoning": "Exactly 10 sentences."\n'
            "  }, ...\n"
            "}\n"
            "No extra keys, no markdown, no code fences."
        )

        # System prompt for Gemini
        self.gem_system_prompt = (
            "You are an expert grader evaluating images/files.\n"
            "You CAN see the attached images/files.\n"
            "For each criterion:\n"
            "1) Give EXACTLY 10 sentences of reasoning.\n"
            "2) Decide true/false.\n"
            "Output ONLY a single JSON object of the shape:\n"
            "{\n"
            '  "<criterion_key>": {\n'
            '    "decision": true|false,\n'
            '    "reasoning": "Exactly 10 sentences."\n'
            "  }, ...\n"
            "}"
        )

    # ---------------- fetch records ----------------
    async def fetch_records(self) -> List[Dict[str, Any]]:
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

        recs = await run_in_thread(
            self.air.all,
            view=GENERAL_VIEW_ID,
            fields=fields,
        )

        filtered = []
        for r in recs:
            f = r.get("fields", {})
            if _has_file_or_image(f.get(REQUESTED_OUTPUTS_FIELD)):
                filtered.append(r)

        logger.info(f"Found {len(filtered)} records requesting file/image output.")
        return filtered

    # ---------------- GPT-5 grading (multimodal) ----------------
    async def grade_with_gpt(
        self,
        prompt: str,
        rubric: list,
        attachments: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Use GPT-5 (and optionally a fallback GPT model) to grade based on the files/images.

        We pass the file URLs as image_url parts so the model can see them.
        """
        # Build {criterion_key: description}
        crit_map: Dict[str, str] = {}
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
        def _user_content() -> List[Dict[str, Any]]:
            uc: List[Dict[str, Any]] = [{"type": "text", "text": text_part}]
            for att in attachments:
                if not isinstance(att, dict):
                    continue
                url = att.get("url")
                if not url:
                    continue
                uc.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": url},
                    }
                )
            return uc

        async def _call_model(model_name: str):
            messages = [
                {
                    "role": "system",
                    "content": self.gpt_system_prompt,
                },
                {
                    "role": "user",
                    "content": _user_content(),
                },
            ]
            async with self.global_sem, self.gpt_sem:
                return await self.gpt_client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    response_format={"type": "json_object"},
                )

        # Try primary model with retry logic (but not on 400 invalid_image_*)
        try:
            resp = await _with_retries(lambda: _call_model(GPT_MODEL))
        except Exception as e:
            msg = str(e)
            if GPT_FALLBACK_MODEL and (
                "invalid_image_format" in msg
                or "invalid_image_url" in msg
                or "INVALID_ARGUMENT" in msg
                or "HTTP/1.1 400" in msg
            ):
                logger.warning(
                    f"Primary GPT model {GPT_MODEL} failed with image error; "
                    f"retrying once with fallback model {GPT_FALLBACK_MODEL}..."
                )
                resp = await _call_model(GPT_FALLBACK_MODEL)
            else:
                raise

        content = resp.choices[0].message.content or "{}"
        data = _safe_json(content)
        if data is None:
            m = re.search(r"\{.*\}", content, re.DOTALL)
            data = _safe_json(m.group(0)) if m else None

        # If still not JSON, fail with a structured default
        if not isinstance(data, dict):
            data = {}

        # Convert decisions into rubric summary
        decisions: Dict[str, Dict[str, Any]] = {}
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

    # ---------------- Gemini grading (via local cache & upload) ----------------
    async def grade_with_gemini(
        self,
        prompt: str,
        rubric: list,
        attachments: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Use Gemini to grade based on real file uploads.

        - Downloads attachments to local cache.
        - Uploads cached files to Gemini.
        - Uses fileUri in generate_content().
        """

        # Build criteria map
        crit_map: Dict[str, str] = {}
        for c in rubric:
            key = list(c.keys())[0]
            desc = (c[key] or {}).get("description", "")
            crit_map[key] = desc

        # Build text prompt
        file_desc = _describe_attachments_for_text(attachments)
        text_prompt = (
            "You are grading images/files.\n"
            "Carefully analyze the attached files which have been uploaded to Gemini.\n\n"
            "ORIGINAL PROMPT:\n"
            f"{prompt}\n\n"
            "CRITERIA:\n"
            f"{json.dumps(crit_map, ensure_ascii=False)}\n\n"
            "ATTACHMENTS (metadata):\n"
            f"{file_desc}\n\n"
            "Return ONLY the required JSON format."
        )

        # Download attachments to cache + upload to Gemini concurrently
        uploaded_parts: List[Dict[str, Any]] = []

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

        # Kick off concurrent upload tasks
        upload_tasks = [
            asyncio.create_task(handle_attachment(att)) for att in attachments
        ]

        for task in asyncio.as_completed(upload_tasks):
            part = await task
            if part:
                uploaded_parts.append(part)

        # Build multimodal contents for Gemini
        parts = [{"text": text_prompt}] + uploaded_parts

        async def _call():
            async with self.gem_sem:
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

        text = getattr(resp, "text", None)
        if not text:
            # try candidates
            cands = getattr(resp, "candidates", [])
            if cands and getattr(cands[0], "content", None):
                parts_out = cands[0].content.parts or []
                text = "\n".join(
                    getattr(p, "text", "") for p in parts_out if getattr(p, "text", "")
                )
        if not text:
            text = "{}"

        data = _safe_json(text)
        if not isinstance(data, dict):
            m = re.search(r"\{.*\}", text, re.DOTALL)
            data = _safe_json(m.group(0)) if m else {}

        if not isinstance(data, dict):
            data = {}

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
                await run_in_thread(self.air.update, rec_id, {NO_FILE_FIELD: True})
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

        logger.info(f"{rec_id}: Grading file/image output with GPT-5 + Gemini...")

        # Run GPT and Gemini grading in parallel for this record
        gpt_task = asyncio.create_task(self.grade_with_gpt(prompt, rubric, attachments))
        gem_task = asyncio.create_task(
            self.grade_with_gemini(prompt, rubric, attachments)
        )

        done = await asyncio.gather(gpt_task, gem_task, return_exceptions=True)
        gpt_out, gem_out = done

        gpt_res = None
        gem_res = None

        if isinstance(gpt_out, Exception):
            logger.error(f"{rec_id}: GPT-5 grading failed: {gpt_out}")
            self.stats["failed"] += 1
        else:
            gpt_res = gpt_out

        if isinstance(gem_out, Exception):
            logger.error(f"{rec_id}: Gemini grading failed: {gem_out}")
            self.stats["failed"] += 1
        else:
            gem_res = gem_out

        if not gpt_res and not gem_res:
            # Nothing to write
            return

        updates: Dict[str, Any] = {}

        if gpt_res:
            updates[GPT_SCORE_FIELD] = gpt_res["percentage"]
            updates[GPT_SUMMARY_FIELD] = gpt_res["summary"]

        if gem_res:
            updates[GEM_SCORE_FIELD] = gem_res["percentage"]
            updates[GEM_SUMMARY_FIELD] = gem_res["summary"]

        if not updates:
            return

        try:
            await run_in_thread(self.air.update, rec_id, updates)
            logger.info(f"{rec_id}: ✅ Updated autorater fields.")
            self.stats["graded"] += 1
        except Exception as e:
            logger.error(f"{rec_id}: Airtable update failed: {e}")
            self.stats["failed"] += 1

    # ---------------- runner ----------------
    async def run(self):
        records = await self.fetch_records()
        if not records:
            logger.info("No file/image tasks found.")
            return

        # Control number of records graded in parallel
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
