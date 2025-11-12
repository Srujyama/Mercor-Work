import hashlib
import io
import json
import mimetypes
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF
import requests
from dotenv import load_dotenv

# === Google Gen AI (new SDK) ===
from google import genai
from google.genai import types
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

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
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "60"))

# Performance knobs
MAX_WORKERS = int(os.getenv("MAX_WORKERS", str(max(4, (os.cpu_count() or 4) * 4))))
DOWNLOAD_CHUNK = int(os.getenv("DOWNLOAD_CHUNK", str(1024 * 256)))
PDF_DPI = int(os.getenv("PDF_DPI", "150"))  # 150 is usually good enough & faster
BATCH_UPDATE_SIZE = int(os.getenv("BATCH_UPDATE_SIZE", "10"))

# Caches (persisted across runs to avoid rework)
CACHE_DIR = OUTPUT_DIR / "cache"
FILES_CACHE_PATH = CACHE_DIR / "files_cache.json"  # url -> {path, etag}
UPLOADS_CACHE_PATH = CACHE_DIR / "uploads_cache.json"  # sha16 -> {file_uri}

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
# Session with retries (connection pooling)
# -----------------------


def make_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "POST", "PATCH"),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=100, pool_maxsize=100)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s


SESSION = make_session()

# -----------------------
# Lightweight JSON cache helpers
# -----------------------


def _load_cache(path: Path) -> Dict[str, Any]:
    if path.exists():
        try:
            return json.loads(path.read_text("utf-8"))
        except Exception:
            return {}
    return {}


def _save_cache(path: Path, data: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


FILES_CACHE = _load_cache(FILES_CACHE_PATH)
UPLOADS_CACHE = _load_cache(UPLOADS_CACHE_PATH)

# -----------------------
# Airtable helpers (records)
# -----------------------


def fetch_airtable_records(base_id: str, table: str, view: str) -> List[Dict[str, Any]]:
    """Fetch records from Airtable, optimistically requesting only needed fields.
    If Airtable returns 422 (unknown fields), automatically retry without the
    fields[] filter so we don't fail the whole run.
    """
    url = f"https://api.airtable.com/v0/{base_id}/{requests.utils.quote(table)}"
    headers = {"Authorization": f"Bearer {AIRTABLE_API_KEY}"}

    base_params = {"view": view, "pageSize": 100}

    # Prefer fetching only the fields we use (faster), but fall back if they don't exist.
    params = base_params | {
        "fields[]": [FIELD_ATTACHMENTS, FIELD_PROMPT, FIELD_CONTEXT]
    }

    out: List[Dict[str, Any]] = []

    def _drain(params_: Dict[str, Any]):
        nonlocal out
        local_params = dict(params_)
        while True:
            r = SESSION.get(
                url, headers=headers, params=local_params, timeout=REQUEST_TIMEOUT
            )
            if r.status_code == 422:
                # Unknown/invalid fields or view – fall back to full payload
                raise ValueError("AIRTABLE_422")
            r.raise_for_status()
            data = r.json()
            out.extend(data.get("records", []))
            offset = data.get("offset")
            if not offset:
                break
            local_params["offset"] = offset

    try:
        _drain(params)
    except ValueError as e:
        if str(e) == "AIRTABLE_422":
            print("[INFO] Airtable 422 with fields[]; retrying without field filter…")
            out = []
            _drain(base_params)
        else:
            raise

    return out


def batch_patch_airtable_records(
    base_id: str, table: str, updates: List[Tuple[str, Dict[str, Any]]]
):
    """Batch update records in chunks for fewer roundtrips."""
    if not updates:
        return
    url = f"https://api.airtable.com/v0/{base_id}/{requests.utils.quote(table)}"
    headers = {
        "Authorization": f"Bearer {AIRTABLE_API_KEY}",
        "Content-Type": "application/json",
    }
    for i in range(0, len(updates), BATCH_UPDATE_SIZE):
        chunk = updates[i : i + BATCH_UPDATE_SIZE]
        payload = {"records": [{"id": rid, "fields": fields} for rid, fields in chunk]}
        r = SESSION.patch(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
        if r.status_code >= 400:
            print(f"[WARN] Batch update failed: {r.text[:300]}")


# -----------------------
# Airtable Web API (schema) helpers – best-effort field creation
# -----------------------


def get_tables_via_web_api(base_id: str) -> Optional[List[Dict[str, Any]]]:
    url = f"https://api.airtable.com/v0/meta/bases/{base_id}/tables"
    headers = {"Authorization": f"Bearer {AIRTABLE_API_KEY}"}
    r = SESSION.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
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
    r = SESSION.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
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
    r = SESSION.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
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
    to_create = [name for name in field_names if name not in existing]
    for name in to_create:
        if create_field_text(base_id, table_id, name):
            print(f"[SCHEMA] Created field: {name}")


# -----------------------
# File / PDF -> images (parallelized + cached)
# -----------------------


def sha256_of_bytes(b: bytes) -> str:
    import hashlib as _h

    return _h.sha256(b).hexdigest()[:16]


def _download_one(url: str, dest_dir: Path) -> Optional[Path]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    headers = {}
    cached = FILES_CACHE.get(url)
    if cached and cached.get("etag"):
        headers["If-None-Match"] = cached["etag"]
    r = SESSION.get(url, headers=headers, stream=True, timeout=REQUEST_TIMEOUT)
    if r.status_code == 304 and cached:
        return Path(cached["path"]) if cached.get("path") else None
    if r.status_code >= 400:
        print(f"[WARN] Download failed {url}: {r.status_code}")
        return None

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
        for chunk in r.iter_content(DOWNLOAD_CHUNK):
            if chunk:
                f.write(chunk)
    FILES_CACHE[url] = {"path": path.as_posix(), "etag": r.headers.get("ETag")}
    return path


def download_urls(urls: List[str], dest_dir: Path) -> List[Path]:
    paths: List[Path] = []
    with ThreadPoolExecutor(max_workers=min(16, MAX_WORKERS)) as ex:
        futs = {ex.submit(_download_one, u, dest_dir): u for u in urls}
        for fut in as_completed(futs):
            p = fut.result()
            if p:
                paths.append(p)
    _save_cache(FILES_CACHE_PATH, FILES_CACHE)
    return paths


def pdf_to_images(pdf_path: Path, out_dir: Path) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    images = []
    with fitz.open(pdf_path) as doc:
        # matrix for target DPI
        zoom = PDF_DPI / 72.0
        mat = fitz.Matrix(zoom, zoom)
        for i, page in enumerate(doc):
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img_path = out_dir / f"{pdf_path.stem}_p{i + 1}.png"
            pix.save(img_path.as_posix())
            images.append(img_path)
    return images


def collect_images_from_attachments(
    attachments: List[Dict[str, Any]], work_dir: Path
) -> List[Path]:
    # Flatten possible nested lookup format
    if isinstance(attachments, dict):
        attachments = [attachments]
    if (
        attachments
        and isinstance(attachments, list)
        and len(attachments) == 1
        and isinstance(attachments[0], list)
    ):
        attachments = attachments[0]

    urls = []
    for att in attachments or []:
        url = att.get("url") or att.get("thumbnails", {}).get("full", {}).get("url")
        if url:
            urls.append(url)

    downloads_dir = work_dir / "downloads"
    locals_ = download_urls(urls, downloads_dir)

    all_images: List[Path] = []
    for local in locals_:
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
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, (int, float, bool)):
        return str(x)
    if isinstance(x, list):
        parts = []
        for it in x:
            if isinstance(it, (str, int, float, bool)):
                parts.append(str(it))
            elif isinstance(it, dict):
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


def _sha16_path(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def upload_files(client: genai.Client, image_paths: List[Path]) -> List[Any]:
    """Upload images via Files API and return uploaded file handles (with cache)."""
    file_handles = []

    def _upload_one(p: Path):
        sha = _sha16_path(p)
        cached = UPLOADS_CACHE.get(sha)
        if cached and cached.get("file_uri"):
            # Reconstruct a lightweight handle using the file URI
            return types.File(
                name=cached["file_uri"]
            )  # SDK accepts file ref by name/uri
        fh = client.files.upload(path=p.as_posix())
        UPLOADS_CACHE[sha] = {
            "file_uri": getattr(fh, "name", None) or getattr(fh, "uri", None)
        }
        return fh

    with ThreadPoolExecutor(max_workers=min(8, MAX_WORKERS)) as ex:
        futs = {ex.submit(_upload_one, p): p for p in image_paths}
        for fut in tqdm(
            as_completed(futs), total=len(futs), desc="Uploading to Gemini"
        ):
            file_handles.append(fut.result())

    _save_cache(UPLOADS_CACHE_PATH, UPLOADS_CACHE)
    return file_handles


def build_user_text(mt_prompt: Any, convo_context: Optional[Any]) -> str:
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
    results = []
    tools = [types.Tool(google_search=types.GoogleSearch())] if web_on else None
    config = types.GenerateContentConfig(tools=tools) if tools else None

    def _one_call(i: int):
        try:
            contents = [text_prompt]
            if files:
                contents.extend(files)
            resp = client.models.generate_content(
                model=model_name, contents=contents, config=config
            )
            out_text = (getattr(resp, "text", None) or "").strip()
            return {
                "run_label": "web_on" if web_on else "web_off",
                "iteration": i,
                "output_text": out_text,
            }
        except Exception as e:
            return {
                "run_label": "web_on" if web_on else "web_off",
                "iteration": i,
                "error": repr(e),
            }

    # Parallelize the N calls while keeping order on write
    with ThreadPoolExecutor(max_workers=min(n, 6)) as ex:
        futs = {ex.submit(_one_call, i): i for i in range(1, n + 1)}
        for fut in as_completed(futs):
            results.append(fut.result())

    # Sort by iteration for determinism
    results.sort(key=lambda r: r.get("iteration", 0))
    return results


# -----------------------
# Orchestration per record (parallelizable across records)
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
    images = (
        collect_images_from_attachments(attachments, work_dir) if attachments else []
    )

    # Upload images to Gemini Files API (cached)
    file_handles = upload_files(make_client(), images) if images else []

    # Prompt text (context optional)
    mt_prompt = fields.get(FIELD_PROMPT, "")
    convo_context = fields.get(FIELD_CONTEXT, "")
    text_prompt = build_user_text(mt_prompt, convo_context)

    # 3x web_off + 3x web_on (in parallel per mode)
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
    (work_dir / "result.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with open(base_out_dir / "all_results.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")

    # Prepare writeback payload (truncate defensively)
    payload = {}
    for i in range(RUNS_PER_MODE):
        label = f"Eval (Web OFF) #{i + 1}"
        if i < len(off_results):
            r = off_results[i]
            payload[label] = (
                r.get("output_text") or f"ERROR: {r.get('error', 'unknown')}"
            )[:50000]
    for i in range(RUNS_PER_MODE):
        label = f"Eval (Web ON) #{i + 1}"
        if i < len(on_results):
            r = on_results[i]
            payload[label] = (
                r.get("output_text") or f"ERROR: {r.get('error', 'unknown')}"
            )[:50000]

    return {
        "id": rec_id,
        "fields": payload,
        "errors": [r for r in meta["runs"] if "error" in r],
    }


# -----------------------
# Main (record-level parallelism + batch updates)
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

    results: List[Dict[str, Any]] = []

    # Process records concurrently
    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, 12)) as ex:
        futs = {
            ex.submit(process_record, client, rec, OUTPUT_DIR): rec.get("id")
            for rec in records
        }
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Processing records"):
            try:
                results.append(fut.result())
            except Exception as e:
                print(f"[ERROR] Record failed: {e}")

    # Batch write results back to Airtable
    updates: List[Tuple[str, Dict[str, Any]]] = []
    for r in results:
        if r and r.get("fields"):
            updates.append((r["id"], r["fields"]))
    batch_patch_airtable_records(AIRTABLE_BASE_ID, AIRTABLE_TABLE_NAME, updates)

    # Write run summary
    summaries = []
    for r in results:
        summaries.append(
            {
                "record_id": r.get("id"),
                "num_runs": sum(1 for _ in (r.get("fields") or {}).values()),
            }
        )
    (OUTPUT_DIR / "run_summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nDone. Artifacts saved to:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
