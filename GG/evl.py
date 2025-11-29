#!/usr/bin/env python3
# pip install --upgrade google-genai python-dotenv requests tqdm

import argparse
import json
import os
import tempfile
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types
from tqdm import tqdm

# ----------------------------
# Environment & Constants
# ----------------------------
load_dotenv()

AIRTABLE_API_KEY = os.getenv("AIRTABLE_API_KEY")
AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID")
AIRTABLE_TABLE = os.getenv("AIRTABLE_TABLE")

# Single view (hard-coded)
AIRTABLE_VIEW = "viwvAyBaCl2YgsFD6"

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

# Single output field
GEMINI_OUTPUT_FIELD = "Gemini 3.0 model responses"

# Relevant Airtable fields
PROMPT_FIELD = "Consolidated Prompt - 10/25"
REQUESTED_OUTPUTS_FIELD = "Requested Outputs"
SHASHAANK_RUN_FIELD = "Shashaank Run"

AIRTABLE_API = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_TABLE}"
AIRTABLE_HEADERS = {
    "Authorization": f"Bearer {AIRTABLE_API_KEY}",
    "Content-Type": "application/json",
}

# How long we're willing to wait for one Gemini call before treating it as failed (in seconds)
GEMINI_MAX_SECONDS = int(os.getenv("GEMINI_MAX_SECONDS", "300"))


# ----------------------------
# Airtable helpers
# ----------------------------
def airtable_list_records(view_id: str, page_size: int = 100) -> List[Dict[str, Any]]:
    records = []
    params = {"pageSize": page_size, "view": view_id}
    while True:
        resp = requests.get(
            AIRTABLE_API, headers=AIRTABLE_HEADERS, params=params, timeout=60
        )
        if resp.status_code == 429:
            print("[AIRTABLE] 429 rate limit, sleeping...")
            time.sleep(1.5)
            continue
        resp.raise_for_status()
        data = resp.json()
        records.extend(data.get("records", []))
        offset = data.get("offset")
        if not offset:
            break
        params["offset"] = offset
        time.sleep(0.2)
    return records


def airtable_update_record(
    record_id: str, fields: Dict[str, Any], max_retries: int = 5
) -> None:
    """PATCH update a single record with retries on 429/5xx."""
    url = f"{AIRTABLE_API}/{record_id}"
    payload = {"fields": fields}
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.patch(
                url, headers=AIRTABLE_HEADERS, data=json.dumps(payload), timeout=60
            )
        except Exception as e:
            print(f"[AIRTABLE] Request exception on PATCH {record_id}: {e}")
            if attempt == max_retries:
                raise
            time.sleep(min(2 * attempt, 8))
            continue

        if resp.status_code in (429, 500, 502, 503, 504):
            print(
                f"[AIRTABLE] {resp.status_code} on PATCH {record_id}, attempt {attempt}, retrying..."
            )
            if attempt == max_retries:
                resp.raise_for_status()
            time.sleep(min(2 * attempt, 8))
            continue

        resp.raise_for_status()
        return


# ----------------------------
# Prompt + attachment helpers
# ----------------------------
def get_prompt(fields: Dict[str, Any]) -> Optional[str]:
    """Return ONLY the 'Consolidated Prompt - 10/25' value (or None if missing/blank)."""
    v = fields.get(PROMPT_FIELD)
    if isinstance(v, str) and v.strip():
        return v.strip()
    return None


def extract_shashaank_attachments(fields: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Returns a list of attachment dicts from the 'Shashaank Run' field.
    Each item typically has: id, url, filename, size, type
    """
    v = fields.get(SHASHAANK_RUN_FIELD)
    if isinstance(v, list):
        atts = [att for att in v if isinstance(att, dict) and att.get("url")]
        if atts:
            print(f"[ATTACH] Found {len(atts)} attachments in Shashaank Run")
        return atts
    return []


def build_full_prompt(fields: Dict[str, Any]) -> Optional[str]:
    """
    Build the base text prompt (without files) from the consolidated prompt.
    We keep this text-only; files are passed separately.
    """
    base_prompt = get_prompt(fields)
    if not base_prompt:
        return None

    attachments = extract_shashaank_attachments(fields)
    if attachments:
        file_list = "\n".join(
            f"- {att.get('filename', 'file')} ({att.get('type', 'unknown')})"
            for att in attachments
        )
        return (
            f"{base_prompt}\n\n"
            "Attached files (images, PDFs, etc.) from the 'Shashaank Run' column:\n"
            f"{file_list}\n\n"
            "Use the attached files as additional context when answering."
        )

    return base_prompt


def should_skip_by_output(fields: Dict[str, Any]) -> bool:
    """
    Skip if 'Requested Outputs' contains 'image' or 'file' (case-insensitive).
    Supports string, list, or Airtable multi-select dicts.
    """
    v = fields.get(REQUESTED_OUTPUTS_FIELD)
    if v is None:
        return False

    def contains_disallowed(s: str) -> bool:
        s = s.lower()
        return ("image" in s) or ("file" in s)

    if isinstance(v, str):
        return contains_disallowed(v)

    if isinstance(v, list):
        for item in v:
            if isinstance(item, str) and contains_disallowed(item):
                return True
            if isinstance(item, dict):
                name = item.get("name")
                if isinstance(name, str) and contains_disallowed(name):
                    return True
        return False

    if isinstance(v, dict):
        name = v.get("name")
        if isinstance(name, str) and contains_disallowed(name):
            return True
        for vv in v.values():
            if isinstance(vv, str) and contains_disallowed(vv):
                return True

    return False


# ----------------------------
# Gemini helpers (multimodal)
# ----------------------------
def build_genai_client(api_key: str, api_version: str = "v1beta") -> genai.Client:
    print(f"[GENAI] Building client with api_version={api_version}")
    return genai.Client(
        api_key=api_key, http_options=types.HttpOptions(api_version=api_version)
    )


def download_attachments(
    attachments: List[Dict[str, Any]], max_bytes: int = 20 * 1024 * 1024
) -> List[Tuple[str, str]]:
    """
    Download attachments locally.
    Returns list of (local_path, mime_type).
    Skips files over max_bytes.
    """
    local_files: List[Tuple[str, str]] = []
    for att in attachments:
        url = att.get("url")
        mime = att.get("type", "application/octet-stream")
        filename = att.get("filename") or "file"
        if not url:
            continue

        size = att.get("size")
        if isinstance(size, int) and size > max_bytes:
            print(
                f"[DOWNLOAD] Skipping {filename} (size {size} > max_bytes {max_bytes})"
            )
            continue

        try:
            print(f"[DOWNLOAD] Fetching {filename} from {url}")
            resp = requests.get(url, timeout=120)
            resp.raise_for_status()
            content = resp.content
            if len(content) > max_bytes:
                print(
                    f"[DOWNLOAD] Skipping {filename}, downloaded size {len(content)} > max_bytes {max_bytes}"
                )
                continue

            suffix = os.path.splitext(filename)[1] or ""
            fd, path = tempfile.mkstemp(prefix="shashaank_", suffix=suffix)
            with os.fdopen(fd, "wb") as f:
                f.write(content)
            print(f"[DOWNLOAD] Saved {filename} to {path} ({len(content)} bytes)")
            local_files.append((path, mime))
        except Exception as e:
            print(f"[DOWNLOAD ERROR] {filename}: {e}")
            traceback.print_exc()
            continue

    return local_files


def upload_files_to_gemini(client: genai.Client, local_files: List[Tuple[str, str]]):
    """
    Upload local files to Gemini and return a list of uploaded file objects.
    Uses client.files.upload(file=..., config=...) which matches newer google-genai.
    """
    uploaded = []
    for path, mime in local_files:
        try:
            print(f"[UPLOAD] Uploading {path} (mime={mime}) to Gemini")
            f = client.files.upload(file=path, config={"mime_type": mime})
            uri = getattr(f, "uri", None)
            name = getattr(f, "name", None)
            print(f"[UPLOAD] Uploaded file uri={uri}, name={name}")
            uploaded.append((f, mime))
        except Exception as e:
            print(f"[UPLOAD ERROR] {path}: {e}")
            traceback.print_exc()
            continue
    return uploaded


def call_gemini(
    client: genai.Client,
    model_name: str,
    prompt: str,
    uploaded_files: Optional[List[Tuple[Any, str]]] = None,
    system_instructions: Optional[str] = None,
    max_retries: int = 3,
) -> Tuple[str, Optional[str]]:
    """
    Call Gemini with text + optional files.
    uploaded_files: list of (file_obj, mime_type)

    NOTE: This version does NOT use request_options (your client version
    doesn't accept it). Timeout is enforced outside this function.
    """
    guard_rails = (
        "Important: Produce a plain text answer only. "
        "Do not include or output new images in your response, but you may use attached files as context. "
    )
    sys_part = f"{system_instructions}\n\n" if system_instructions else ""
    composed = f"{guard_rails}\n\n{sys_part}{prompt}"

    def build_parts() -> List[Dict[str, Any]]:
        parts: List[Dict[str, Any]] = [{"text": composed}]
        if uploaded_files:
            for f_obj, mime in uploaded_files:
                file_uri = getattr(f_obj, "uri", None)
                if not file_uri:
                    print("[PARTS] Uploaded file missing uri, skipping")
                    continue
                print(f"[PARTS] Adding file part uri={file_uri}, mime={mime}")
                parts.append(
                    {
                        "fileData": {
                            "fileUri": file_uri,
                            "mimeType": mime,
                        }
                    }
                )
        return parts

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            print(f"[GENAI] Calling generate_content, attempt {attempt}")
            parts = build_parts()
            resp = client.models.generate_content(
                model=model_name,
                contents=[
                    {
                        "role": "user",
                        "parts": parts,
                    },
                ],
            )
            model_used = getattr(resp, "model", None)
            print(f"[GENAI] Response model: {model_used}")

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
            text = (text or "").strip()
            return text, model_used
        except Exception as e:
            last_err = e
            print(f"[GENAI ERROR] attempt {attempt}: {e}")
            traceback.print_exc()
            time.sleep(1.5 * attempt)
    raise RuntimeError(f"Gemini call failed after {max_retries} attempts: {last_err}")


# ----------------------------
# Simple token bucket limiter for Airtable writes
# ----------------------------
class RateLimiter:
    def __init__(self, rate_per_sec: float, capacity: Optional[int] = None):
        self.rate = float(rate_per_sec)
        self.capacity = int(capacity or max(1, int(rate_per_sec)))
        self.tokens = self.capacity
        self.lock = threading.Lock()
        self.last = time.monotonic()

    def acquire(self):
        while True:
            with self.lock:
                now = time.monotonic()
                refill = (now - self.last) * self.rate
                if refill > 0:
                    self.tokens = min(self.capacity, self.tokens + refill)
                    self.last = now
                if self.tokens >= 1:
                    self.tokens -= 1
                    return
            time.sleep(0.01)


# ----------------------------
# Core processing
# ----------------------------
def process_view(
    client: genai.Client,
    model_name: str,
    view_id: str,
    overwrite: bool = False,
    dry_run: bool = False,
    limit: Optional[int] = None,
    workers: int = 4,
    airtable_rps: float = 4.0,
) -> Dict[str, int]:
    print(f"\n[VIEW] Listing records for view={view_id}")
    records = airtable_list_records(view_id)
    if limit is not None:
        records = records[:limit]
    total = len(records)
    print(f"[VIEW] Total records in this view: {total}")

    jobs = []  # (rec_id, prompt, attachments)
    already_filled = skipped_no_prompt = skipped_output_tasks = 0

    for rec in records:
        rec_id = rec.get("id")
        fields = rec.get("fields", {})

        if should_skip_by_output(fields):
            skipped_output_tasks += 1
            continue

        # Always check our single output field
        existing = fields.get(GEMINI_OUTPUT_FIELD)
        if existing and not overwrite:
            already_filled += 1
            continue

        prompt = build_full_prompt(fields)
        if not prompt:
            skipped_no_prompt += 1
            continue

        attachments = extract_shashaank_attachments(fields)
        jobs.append((rec_id, prompt, attachments))

    print(
        f"[VIEW] Jobs to run: {len(jobs)}, already_filled={already_filled}, "
        f"skipped_no_prompt={skipped_no_prompt}, skipped_output_tasks={skipped_output_tasks}"
    )

    if dry_run:
        print(f"[DRY RUN] Would generate for {len(jobs)} tasks.")
        return {
            "processed": len(jobs),
            "skipped_no_prompt": skipped_no_prompt,
            "skipped_output_tasks": skipped_output_tasks,
            "already_filled": already_filled,
            "errors": 0,
            "total_seen": total,
        }

    errors = 0

    def _gen_task(rec_id: str, prompt: str, attachments: List[Dict[str, Any]]):
        print(f"[TASK] Starting task for record {rec_id}")
        local_files = download_attachments(attachments)
        uploaded = upload_files_to_gemini(client, local_files)

        try:
            # Run call_gemini in a tiny one-off executor so we can enforce a timeout
            with ThreadPoolExecutor(max_workers=1) as inner_pool:
                fut = inner_pool.submit(
                    call_gemini,
                    client,
                    model_name,
                    prompt,
                    uploaded_files=uploaded,
                )
                text, served = fut.result(timeout=GEMINI_MAX_SECONDS)
        except TimeoutError:
            print(
                f"[TIMEOUT] Gemini call for record {rec_id} exceeded {GEMINI_MAX_SECONDS}s, skipping."
            )
            text = f"[ERROR] Gemini call timed out after {GEMINI_MAX_SECONDS} seconds."
            served = None
        except Exception as e:
            print(f"[GEN_TASK ERROR] record {rec_id}: {e}")
            traceback.print_exc()
            text = f"[ERROR] Gemini call failed: {e}"
            served = None
        finally:
            # Always clean up temp files
            for path, _ in local_files:
                try:
                    os.remove(path)
                    print(f"[CLEANUP] Removed temp file {path}")
                except OSError as e:
                    print(f"[CLEANUP ERROR] {path}: {e}")

        return rec_id, text, served

    limiter = RateLimiter(rate_per_sec=float(airtable_rps))
    processed = 0
    write_errors = 0

    # Generate + write as each task finishes
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        future_to_rec = {
            pool.submit(_gen_task, rid, pr, atts): rid for rid, pr, atts in jobs
        }
        with tqdm(
            total=len(future_to_rec),
            desc=f"Generating+Writing ({view_id})",
            unit="task",
        ) as pbar:
            for fut in as_completed(future_to_rec):
                rec_id = future_to_rec[fut]
                try:
                    rec_id, text, served = fut.result()
                    # Write to Airtable immediately, always to GEMINI_OUTPUT_FIELD
                    limiter.acquire()
                    airtable_update_record(rec_id, {GEMINI_OUTPUT_FIELD: text})
                    processed += 1
                    tag = f"{served}" if served else "ok"
                    pbar.set_postfix_str(f"⬆️ {rec_id[:6]} {tag}")
                except Exception as e:
                    errors += 1
                    write_errors += 1
                    print("\n[GENERATION OR WRITE ERROR]")
                    print(f"View: {view_id}")
                    print(f"Record: {rec_id}")
                    print(f"Error type: {type(e).__name__}")
                    print(f"Error: {e}")
                    traceback.print_exc()
                    pbar.set_postfix_str(f"❌ {rec_id[:6]} err")
                finally:
                    pbar.update(1)

    errors += write_errors
    return {
        "processed": processed,
        "skipped_no_prompt": skipped_no_prompt,
        "skipped_output_tasks": skipped_output_tasks,
        "already_filled": already_filled,
        "errors": errors,
        "total_seen": total,
    }


# ----------------------------
# CLI
# ----------------------------
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Parallel Gemini 3.0 Airtable generator for a single view "
            "(multimodal with Shashaank Run attachments, with debug logging)."
        )
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing responses."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Skip API calls and Airtable writes."
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit rows (for testing)."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("GEMINI_MODEL", "gemini-3-pro-preview"),
        help="Gemini model to use (default gemini-3-pro-preview).",
    )
    parser.add_argument(
        "--api-version",
        type=str,
        default=os.getenv("GENAI_API_VERSION", "v1beta"),
        help="GenAI API version (default v1beta).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.getenv("EVL_WORKERS", "4")),
        help="Parallel Gemini calls (default 4).",
    )
    parser.add_argument(
        "--airtable-rps",
        type=float,
        default=float(os.getenv("AIRTABLE_RPS", "4")),
        help="Max Airtable writes/sec (default 4).",
    )
    args = parser.parse_args()

    if not GOOGLE_API_KEY:
        raise SystemExit("Missing GOOGLE_API_KEY or GEMINI_API_KEY.")
    if not AIRTABLE_API_KEY or not AIRTABLE_BASE_ID or not AIRTABLE_TABLE:
        raise SystemExit(
            "Missing one of AIRTABLE_API_KEY, AIRTABLE_BASE_ID, AIRTABLE_TABLE."
        )

    if not AIRTABLE_VIEW:
        raise SystemExit("AIRTABLE_VIEW is not configured.")

    client = build_genai_client(GOOGLE_API_KEY, api_version=args.api_version)

    print("\n=== Processing single view ===")
    summary = process_view(
        client,
        args.model,
        AIRTABLE_VIEW,
        args.overwrite,
        args.dry_run,
        args.limit,
        args.workers,
        args.airtable_rps,
    )

    print("\n=== Done ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
