import hashlib
import io
import json
import mimetypes
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import fitz  # PyMuPDF
import requests
from dotenv import load_dotenv

# === Google Gen AI (new SDK) ===
from google import genai
from google.genai import types
from tqdm import tqdm

# -----------------------
# Environment & constants
# -----------------------
load_dotenv()

# Airtable
AIRTABLE_API_KEY = os.getenv("AIRTABLE_API_KEY")
AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID")
AIRTABLE_TABLE_NAME = os.getenv("AIRTABLE_TABLE_NAME")
AIRTABLE_VIEW_NAME = os.getenv("AIRTABLE_VIEW_NAME")

# Fields
FIELD_ATTACHMENTS = os.getenv("FIELD_ATTACHMENTS", "Shashaank Run (make PDF to images)")
FIELD_CONTEXT = os.getenv("FIELD_CONTEXT", "Conversation Context (Multi-Turn)")
FIELD_PROMPT = os.getenv("FIELD_PROMPT", "MT Prompt")

# Output fields (created if missing, best-effort)
OUTPUT_FIELDS = [
    "Eval (Web OFF) #1",
    "Eval (Web OFF) #2",
    "Eval (Web OFF) #3",
    "Eval (Web ON) #1",
    "Eval (Web ON) #2",
    "Eval (Web ON) #3",
]

# Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")

RUNS_PER_MODE = int(os.getenv("RUNS_PER_MODE", "3"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./run_outputs"))
REQUEST_TIMEOUT = 60


# -----------------------
# Basic guards
# -----------------------
def require_env(name: str):
    val = os.getenv(name)
    if not val:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return val


require_env("AIRTABLE_API_KEY")
require_env("AIRTABLE_BASE_ID")
require_env("AIRTABLE_TABLE_NAME")
require_env("AIRTABLE_VIEW_NAME")
require_env("GEMINI_API_KEY")


# -----------------------
# Airtable helpers (records)
# -----------------------
def fetch_airtable_records(base_id: str, table: str, view: str) -> List[Dict[str, Any]]:
    url = f"https://api.airtable.com/v0/{base_id}/{requests.utils.quote(table)}"
    headers = {"Authorization": f"Bearer {AIRTABLE_API_KEY}"}
    params = {"view": view, "pageSize": 100}
    out = []
    while True:
        r = requests.get(url, headers=headers, params=params, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        out.extend(data.get("records", []))
        offset = data.get("offset")
        if not offset:
            break
        params["offset"] = offset
    return out


def patch_airtable_record_fields(
    base_id: str, table: str, record_id: str, fields: Dict[str, Any]
):
    url = f"https://api.airtable.com/v0/{base_id}/{requests.utils.quote(table)}/{record_id}"
    headers = {
        "Authorization": f"Bearer {AIRTABLE_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {"fields": fields}
    r = requests.patch(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
    try:
        r.raise_for_status()
    except Exception:
        print(f"[WARN] Failed to update record {record_id}: {r.text[:500]}")


# -----------------------
# Airtable Web API (schema) helpers – best-effort field creation
# -----------------------
def get_tables_via_web_api(base_id: str) -> Optional[List[Dict[str, Any]]]:
    url = f"https://api.airtable.com/v0/meta/bases/{base_id}/tables"
    headers = {"Authorization": f"Bearer {AIRTABLE_API_KEY}"}
    r = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
    if r.status_code != 200:
        print(
            f"[INFO] Could not fetch tables via Web API (status {r.status_code}). Skipping field creation."
        )
        return None
    return r.json().get("tables", [])


def get_table_id_by_name(base_id: str, table_name: str) -> Optional[str]:
    tables = get_tables_via_web_api(base_id)
    if not tables:
        return None
    for t in tables:
        if t.get("name") == table_name:
            return t.get("id")
    print("[INFO] Table not found in Web API listing (or missing access).")
    return None


def get_existing_field_names(base_id: str, table_id: str) -> Optional[set]:
    url = f"https://api.airtable.com/v0/meta/bases/{base_id}/tables/{table_id}"
    headers = {"Authorization": f"Bearer {AIRTABLE_API_KEY}"}
    r = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
    if r.status_code != 200:
        print(
            f"[INFO] Could not fetch table detail via Web API (status {r.status_code})."
        )
        return None
    fields = r.json().get("fields", [])
    return set([f.get("name") for f in fields if "name" in f])


def create_field_text(base_id: str, table_id: str, field_name: str) -> bool:
    url = f"https://api.airtable.com/v0/meta/bases/{base_id}/tables/{table_id}/fields"
    headers = {
        "Authorization": f"Bearer {AIRTABLE_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {"type": "singleLineText", "name": field_name}
    r = requests.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
    if r.status_code in (200, 201):
        return True
    print(
        f"[INFO] Could not create field '{field_name}' (status {r.status_code}): {r.text[:300]}"
    )
    return False


def ensure_output_fields_exist(base_id: str, table_name: str, field_names: List[str]):
    table_id = get_table_id_by_name(base_id, table_name)
    if not table_id:
        return
    existing = get_existing_field_names(base_id, table_id)
    if existing is None:
        return
    for name in field_names:
        if name not in existing:
            if create_field_text(base_id, table_id, name):
                print(f"[SCHEMA] Created field: {name}")


# -----------------------
# File / PDF -> images
# -----------------------
def sha256_of_bytes(b: bytes) -> str:
    import hashlib as _h

    return _h.sha256(b).hexdigest()[:16]


def download_url(url: str, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, stream=True, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    filename = None
    cd = r.headers.get("content-disposition")
    if cd and "filename=" in cd:
        filename = cd.split("filename=")[1].strip("\"' ")
    if not filename:
        filename = (
            url.split("?")[0].split("/")[-1] or f"file-{sha256_of_bytes(r.content)}"
        )
    path = dest_dir / filename
    with open(path, "wb") as f:
        for chunk in r.iter_content(1024 * 128):
            f.write(chunk)
    return path


def pdf_to_images(pdf_path: Path, out_dir: Path) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    images = []
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc):
            pix = page.get_pixmap(dpi=200, alpha=False)
            img_path = out_dir / f"{pdf_path.stem}_p{i + 1}.png"
            pix.save(img_path.as_posix())
            images.append(img_path)
    return images


def collect_images_from_attachments(
    attachments: List[Dict[str, Any]], work_dir: Path
) -> List[Path]:
    all_images: List[Path] = []
    downloads_dir = work_dir / "downloads"
    for att in attachments or []:
        url = att.get("url") or att.get("thumbnails", {}).get("full", {}).get("url")
        if not url:
            continue
        local = download_url(url, downloads_dir)
        mime, _ = mimetypes.guess_type(local.name)
        if (mime or "").lower() == "application/pdf" or local.suffix.lower() == ".pdf":
            page_dir = work_dir / f"{local.stem}_pages"
            imgs = pdf_to_images(local, page_dir)
            all_images.extend(imgs)
        else:
            all_images.append(local)
    return all_images


# -----------------------
# Text coercion helpers (fix list/lookup fields)
# -----------------------
def coerce_text(x) -> str:
    """Coerce Airtable field values (string/number/list/lookup/dict) to a readable string."""
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, (int, float, bool)):
        return str(x)
    if isinstance(x, list):
        parts = []
        for it in x:
            if isinstance(it, str):
                parts.append(it)
            elif isinstance(it, (int, float, bool)):
                parts.append(str(it))
            elif isinstance(it, dict):
                # Try common keys from lookups/linked records
                for k in ("text", "value", "name", "email", "title", "id"):
                    if k in it and isinstance(it[k], (str, int, float, bool)):
                        parts.append(str(it[k]))
                        break
                else:
                    parts.append(json.dumps(it, ensure_ascii=False))
            else:
                parts.append(str(it))
        return "\n".join(parts)
    if isinstance(x, dict):
        for k in ("text", "value", "name", "title"):
            if k in x and isinstance(x[k], (str, int, float, bool)):
                return str(x[k])
        return json.dumps(x, ensure_ascii=False)
    return str(x)


# -----------------------
# Gemini setup & calls (new SDK + grounding)
# -----------------------
def make_client() -> genai.Client:
    return genai.Client(api_key=GEMINI_API_KEY)


def upload_files(client: genai.Client, image_paths: List[Path]) -> List[Any]:
    """Upload images via Files API and return uploaded file handles."""
    file_handles = []
    for p in tqdm(image_paths, desc="Uploading to Gemini Files API"):
        # New SDK supports upload by path param name 'path' or 'file'
        file_handle = client.files.upload(path=p.as_posix())
        file_handles.append(file_handle)
    return file_handles


def build_user_text(mt_prompt: Any, convo_context: Optional[Any]) -> str:
    """If context empty, omit it — use attachments + MT Prompt only."""
    mt_prompt_s = coerce_text(mt_prompt).strip()
    convo_context_s = coerce_text(convo_context).strip()
    parts = ["### Main Task (MT Prompt)\n", mt_prompt_s if mt_prompt_s else "(empty)"]
    if convo_context_s:
        parts.append("\n\n### Conversation Context (Multi-Turn)\n")
        parts.append(convo_context_s)
    return "".join(parts)


def generate_many(
    client: genai.Client,
    model_name: str,
    text_prompt: str,
    files: List[Any],
    n: int,
    web_on: bool,
) -> List[Dict[str, Any]]:
    """
    Calls the model n times.
    - When web_on=True, attach Grounding with Google Search tool.
    - Include uploaded files in 'contents'.
    """
    results = []
    # Prepare tool config for grounding (web_on)
    tools = None
    if web_on:
        tools = [types.Tool(google_search=types.GoogleSearch())]  # grounding tool

    for i in range(1, n + 1):
        try:
            contents = [text_prompt]
            # Pass file handles directly in contents; the SDK resolves URIs.
            if files:
                contents.extend(files)

            config = types.GenerateContentConfig(tools=tools) if tools else None

            resp = client.models.generate_content(
                model=model_name, contents=contents, config=config
            )
            out_text = (getattr(resp, "text", None) or "").strip()
            results.append(
                {
                    "run_label": "web_on" if web_on else "web_off",
                    "iteration": i,
                    "output_text": out_text,
                }
            )
        except Exception as e:
            results.append(
                {
                    "run_label": "web_on" if web_on else "web_off",
                    "iteration": i,
                    "error": repr(e),
                }
            )
        time.sleep(0.4)
    return results


# -----------------------
# Orchestration per record
# -----------------------
def process_record(
    client: genai.Client, record: Dict[str, Any], base_out_dir: Path
) -> Dict[str, Any]:
    rec_id = record.get("id")
    fields = record.get("fields", {})
    work_dir = base_out_dir / f"record_{rec_id}"
    work_dir.mkdir(parents=True, exist_ok=True)

    # Attachments → images
    attachments = fields.get(FIELD_ATTACHMENTS, [])
    if isinstance(attachments, dict):
        attachments = [attachments]
    # Some Airtable setups wrap attachments in lookups: list[list[dict]]; flatten once if needed
    if (
        attachments
        and isinstance(attachments, list)
        and len(attachments) == 1
        and isinstance(attachments[0], list)
    ):
        attachments = attachments[0]
    images = (
        collect_images_from_attachments(attachments, work_dir) if attachments else []
    )

    # Upload images to Gemini Files API
    file_handles = upload_files(client, images) if images else []

    # Prompt text (context optional) — now robust to list/lookup fields
    mt_prompt = fields.get(FIELD_PROMPT, "")
    convo_context = fields.get(FIELD_CONTEXT, "")
    text_prompt = build_user_text(mt_prompt, convo_context)

    # 3x web_off + 3x web_on
    off_results = generate_many(
        client, GEMINI_MODEL, text_prompt, file_handles, RUNS_PER_MODE, web_on=False
    )
    on_results = generate_many(
        client, GEMINI_MODEL, text_prompt, file_handles, RUNS_PER_MODE, web_on=True
    )

    # Persist locally
    meta = {
        "record_id": rec_id,
        "images": [p.as_posix() for p in images],
        "runs": off_results + on_results,
    }
    with open(work_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    with open(base_out_dir / "all_results.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")

    # Write back 6 columns
    payload = {}
    # OFF
    for i in range(RUNS_PER_MODE):
        label = f"Eval (Web OFF) #{i + 1}"
        if i < len(off_results):
            r = off_results[i]
            payload[label] = (
                r.get("output_text") or f"ERROR: {r.get('error', 'unknown')}"
            )[:50000]
    # ON
    for i in range(RUNS_PER_MODE):
        label = f"Eval (Web ON) #{i + 1}"
        if i < len(on_results):
            r = on_results[i]
            payload[label] = (
                r.get("output_text") or f"ERROR: {r.get('error', 'unknown')}"
            )[:50000]

    if payload:
        patch_airtable_record_fields(
            AIRTABLE_BASE_ID, AIRTABLE_TABLE_NAME, rec_id, payload
        )

    return meta


# -----------------------
# Main
# -----------------------
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Best-effort: create output fields if missing
    ensure_output_fields_exist(AIRTABLE_BASE_ID, AIRTABLE_TABLE_NAME, OUTPUT_FIELDS)

    print("Creating Gemini client…")
    client = make_client()

    print("Fetching Airtable records…")
    records = fetch_airtable_records(
        AIRTABLE_BASE_ID, AIRTABLE_TABLE_NAME, AIRTABLE_VIEW_NAME
    )
    print(f"Found {len(records)} record(s)")

    summaries = []
    for rec in records:
        summary = process_record(client, rec, OUTPUT_DIR)
        summaries.append(
            {
                "record_id": summary["record_id"],
                "num_images": len(summary["images"]),
                "num_runs": len(summary["runs"]),
                "errors": [r for r in summary["runs"] if "error" in r],
            }
        )
        err_cnt = len(summaries[-1]["errors"])
        print(
            f"Record {summary['record_id']}: images={len(summary['images'])}, runs={len(summary['runs'])}, errors={err_cnt}"
        )

    with open(OUTPUT_DIR / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)
    print("\nDone. Artifacts saved to:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()


