#!/usr/bin/env python3
"""
Combined View Dual Autograder (FILE/IMAGE ONLY) with tqdm
---------------------------------------------------------
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
from tqdm.auto import tqdm  # <-- progress bar

# ----------------------- Load ENV -----------------------
load_dotenv()

AIRTABLE_API_KEY = os.getenv("AIRTABLE_API_KEY")
AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID", "appgeueGlH9mCUTvu")
AIRTABLE_TABLE_ID = os.getenv("AIRTABLE_TABLE", "tblfy3EPxl1PHvKV7")
AIRTABLE_VIEW_COMBINED = os.getenv("AIRTABLE_VIEW_COMBINED", "viwliqVDSuuEXNxgN")

GPT_API_KEY = os.getenv("GPT_API_KEY")
if not GPT_API_KEY:
    raise RuntimeError("Missing GPT_API_KEY in .env")

GOOGLE_API_KEY = (
    os.getenv("GOOGLE_API_KEY")
    or os.getenv("GOOGLE_API_KEY_1")
    or os.getenv("GOOGLE_API_KEY_2")
)
if not GOOGLE_API_KEY:
    raise RuntimeError("Missing GOOGLE_API_KEY / GOOGLE_API_KEY_1 in .env")

GPT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3-pro-preview")

GPT_KEY_CONCURRENCY = int(os.getenv("GPT_KEY_CONCURRENCY", "5"))
GEMINI_KEY_CONCURRENCY = int(os.getenv("GEMINI_KEY_CONCURRENCY", "3"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "4"))

GEMINI_CACHE_DIR = Path(os.getenv("GEMINI_CACHE_DIR", "./gemini_cache_combined"))
GEMINI_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- Fields ----------------
PROMPT_FIELD = "Consolidated Prompt - 10/25"
RUBRIC_FIELD = "Rubric JSON"

FILE_GEM25_FIELD = "Gemini Final Output Files"
FILE_GEM30_FIELD = "Gemini 3.0 Pro Response (File Output) - Gemini App"  # <-- fixed
FILE_GPT5_FIELD = "GPT5 Response (File Output)"

RESP_TYPES = [
    {
        "name": "Gemini 2.5 (file)",
        "file_field": FILE_GEM25_FIELD,
        "gpt_score_field": "GPT5 Autorater - Gemini Response Score",
        "gpt_summary_field": "[GPT5 graded] Gemini Response Scoring Summary",
        "gem_score_field": "Gemini Autorater - Gemini Response Score",
        "gem_summary_field": "[Gemini graded] Gemini Response Scoring Summary",
    },
    {
        "name": "Gemini 3.0 (file)",
        "file_field": FILE_GEM30_FIELD,
        "gpt_score_field": "GPT5 Autorater - Gemini 3.0 Response Score",
        "gpt_summary_field": "GPT5 Autorater - Gemini 3.0 Response Summary",
        "gem_score_field": "Gemini Autorater - Gemini 3.0 Response Score",
        "gem_summary_field": "Gemini Autorater - Gemini 3.0 Response Summary",
    },
    {
        "name": "GPT-5 (file)",
        "file_field": FILE_GPT5_FIELD,
        "gpt_score_field": "GPT5 Autorater - GPT5 Response Score",
        "gpt_summary_field": "[GPT5 graded] GPT5 Response Scoring Summary",
        "gem_score_field": "Gemini Autorater - GPT5 Response Score",
        "gem_summary_field": "[Gemini graded] GPT5 Response Scoring Summary",
    },
]

# ----------------------- Logging -----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ----------------------- Helpers -----------------------
async def run_in_thread(fn, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, functools.partial(fn, *args, **kwargs))


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
            msg = str(e)
            if (
                "INVALID_ARGUMENT" in msg
                or "invalid_request_error" in msg
                or "invalid_image_format" in msg
                or "invalid_image_url" in msg
                or "HTTP/1.1 400" in msg
            ) and attempt == 0:
                raise
            if attempt == retries - 1:
                raise
            sleep = base * (2**attempt) + random.uniform(0, jitter)
            logger.warning(f"Transient error: {e} — retrying in {sleep:.2f}s")
            await asyncio.sleep(sleep)


def _attachment_cache_key(att: Dict[str, Any]) -> str:
    att_id = att.get("id")
    if att_id:
        return att_id
    h = hashlib.sha1()
    url = (att.get("url") or "").encode("utf-8", errors="ignore")
    name = (att.get("filename") or "").encode("utf-8", errors="ignore")
    h.update(url + b"|" + name)
    return h.hexdigest()


def _attachment_cache_path(att: Dict[str, Any]) -> Path:
    key = _attachment_cache_key(att)
    filename = att.get("filename") or "file"
    ext = Path(filename).suffix or ""
    return GEMINI_CACHE_DIR / f"{key}{ext}"


def _download_attachment_to_cache(
    att: Dict[str, Any], max_bytes: int = 20 * 1024 * 1024
) -> Optional[Path]:
    url = att.get("url")
    if not url:
        return None

    target = _attachment_cache_path(att)
    if target.exists() and target.stat().st_size > 0:
        return target

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

    return target


async def download_attachment_to_cache_async(
    att: Dict[str, Any], max_bytes: int = 20 * 1024 * 1024
) -> Optional[Path]:
    return await run_in_thread(_download_attachment_to_cache, att, max_bytes)


def _describe_attachments_for_text(attachments: List[Dict[str, Any]]) -> str:
    lines = []
    for att in attachments:
        if not isinstance(att, dict):
            continue
        fname = att.get("filename", "file")
        mime = att.get("type", "application/octet-stream")
        size = att.get("size", "unknown")
        lines.append(f"- {fname} (mime={mime}, size={size})")
    return "\n".join(lines)


class CombinedFileDualAutograder:
    def __init__(self):
        self.air = Api(AIRTABLE_API_KEY).table(AIRTABLE_BASE_ID, AIRTABLE_TABLE_ID)

        self.gpt_client = AsyncOpenAI(api_key=GPT_API_KEY)
        self.global_sem = asyncio.Semaphore(
            max(4, GPT_KEY_CONCURRENCY + GEMINI_KEY_CONCURRENCY)
        )
        self.gpt_sem = asyncio.Semaphore(GPT_KEY_CONCURRENCY)

        self.gem_client = genai.Client(
            api_key=GOOGLE_API_KEY,
            http_options=types.HttpOptions(api_version="v1beta"),
        )
        self.gem_sem = asyncio.Semaphore(GEMINI_KEY_CONCURRENCY)

        self.stats = {"processed": 0, "graded": 0, "failed": 0, "skipped": 0}

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

    async def fetch_records(self) -> List[Dict[str, Any]]:
        fields = [PROMPT_FIELD, RUBRIC_FIELD]
        for cfg in RESP_TYPES:
            fields.extend(
                [
                    cfg["file_field"],
                    cfg["gpt_score_field"],
                    cfg["gpt_summary_field"],
                    cfg["gem_score_field"],
                    cfg["gem_summary_field"],
                ]
            )

        recs = await run_in_thread(
            self.air.all,
            view=AIRTABLE_VIEW_COMBINED,
            fields=list(set(fields)),
        )

        filtered = []
        for r in recs:
            f = r.get("fields", {})
            rub = (f.get(RUBRIC_FIELD) or "").strip()
            if not rub:
                continue

            include = False
            for cfg in RESP_TYPES:
                attachments = f.get(cfg["file_field"])
                if not isinstance(attachments, list) or not attachments:
                    continue

                gpt_score = f.get(cfg["gpt_score_field"])
                gpt_summary = f.get(cfg["gpt_summary_field"])
                gem_score = f.get(cfg["gem_score_field"])
                gem_summary = f.get(cfg["gem_summary_field"])

                if (
                    gpt_score is None
                    or gpt_summary is None
                    or gem_score is None
                    or gem_summary is None
                ):
                    include = True
                    break

            if include:
                filtered.append(r)

        logger.info(
            f"[Combined FILE] Found {len(filtered)} records with file/image outputs needing grading."
        )
        return filtered

    async def grade_with_gpt(
        self,
        prompt: str,
        rubric: list,
        attachments: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        crit_map: Dict[str, str] = {}
        for c in rubric:
            if not isinstance(c, dict) or not c:
                continue
            key = list(c.keys())[0]
            desc = (c.get(key) or {}).get("description", "")
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

        async def _call():
            messages = [
                {"role": "system", "content": self.gpt_system_prompt},
                {"role": "user", "content": _user_content()},
            ]
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
        if not isinstance(data, dict):
            data = {}

        decisions: Dict[str, Dict[str, Any]] = {}
        for key, obj in data.items():
            decision = bool((obj or {}).get("decision"))
            reasoning = (obj or {}).get("reasoning", "") or ""
            decisions[key] = {"decision": decision, "reasoning": reasoning.strip()}

        graded = []
        true_count = 0
        for c in rubric:
            if not isinstance(c, dict) or not c:
                continue
            key = list(c.keys())[0]
            meta = c.get(key, {}) or {}
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
            if ok:
                true_count += 1

        pct = (true_count / len(graded) * 100) if graded else 0.0
        return {"percentage": pct, "summary": json.dumps(graded, separators=(",", ":"))}

    async def grade_with_gemini(
        self,
        prompt: str,
        rubric: list,
        attachments: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        crit_map: Dict[str, str] = {}
        for c in rubric:
            if not isinstance(c, dict) or not c:
                continue
            key = list(c.keys())[0]
            desc = (c.get(key) or {}).get("description", "")
            crit_map[key] = desc

        file_desc = _describe_attachments_for_text(attachments)
        text_prompt = (
            "You are grading images/files.\n"
            "Carefully analyze the attached files which have been uploaded to Gemini.\n\n"
            f"ORIGINAL PROMPT:\n{prompt}\n\n"
            "CRITERIA:\n"
            f"{json.dumps(crit_map, ensure_ascii=False)}\n\n"
            "ATTACHMENTS (metadata):\n"
            f"{file_desc}\n\n"
            "Return ONLY the required JSON format."
        )

        uploaded_parts: List[Dict[str, Any]] = []

        async def handle_attachment(att: Dict[str, Any]):
            if not isinstance(att, dict):
                return None
            local_path = await download_attachment_to_cache_async(att)
            if not local_path:
                return None
            mime = att.get("type", "application/octet-stream")

            async with self.gem_sem:
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
                return None

            return {"fileData": {"fileUri": uri, "mimeType": mime}}

        upload_tasks = [
            asyncio.create_task(handle_attachment(att)) for att in attachments
        ]
        for task in asyncio.as_completed(upload_tasks):
            part = await task
            if part:
                uploaded_parts.append(part)

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

        graded = []
        true_count = 0
        for c in rubric:
            if not isinstance(c, dict) or not c:
                continue
            key = list(c.keys())[0]
            meta = c.get(key, {}) or {}
            obj = data.get(key, {})
            decision = bool(obj.get("decision"))
            reasoning = obj.get("reasoning", "") or ""
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
        return {"percentage": pct, "summary": json.dumps(graded, separators=(",", ":"))}

    async def process_record(self, rec: Dict[str, Any]):
        self.stats["processed"] += 1
        f = rec.get("fields", {})
        rec_id = rec["id"]

        rubric_raw = (f.get(RUBRIC_FIELD) or "").strip()
        if not rubric_raw:
            self.stats["skipped"] += 1
            return

        rubric = _safe_json(rubric_raw)
        if not isinstance(rubric, list):
            self.stats["skipped"] += 1
            return

        prompt = f.get(PROMPT_FIELD) or ""
        updates: Dict[str, Any] = {}
        any_graded = False

        for cfg in RESP_TYPES:
            name = cfg["name"]
            attachments = f.get(cfg["file_field"])
            if not isinstance(attachments, list) or not attachments:
                continue

            gpt_score = f.get(cfg["gpt_score_field"])
            gpt_summary = f.get(cfg["gpt_summary_field"])
            gem_score = f.get(cfg["gem_score_field"])
            gem_summary = f.get(cfg["gem_summary_field"])

            if (
                gpt_score is not None
                and gpt_summary is not None
                and gem_score is not None
                and gem_summary is not None
            ):
                continue

            logger.info(
                f"[Combined FILE] {rec_id}: Grading {name} output with GPT-5 + Gemini..."
            )

            tasks = []
            need_gpt = gpt_score is None or gpt_summary is None
            need_gem = gem_score is None or gem_summary is None

            if need_gpt:
                tasks.append(("gpt", self.grade_with_gpt(prompt, rubric, attachments)))
            if need_gem:
                tasks.append(
                    ("gem", self.grade_with_gemini(prompt, rubric, attachments))
                )

            if not tasks:
                continue

            results = await asyncio.gather(
                *[t[1] for t in tasks], return_exceptions=True
            )

            for (backend, _), res in zip(tasks, results):
                if isinstance(res, Exception):
                    logger.error(
                        f"[Combined FILE] {rec_id}: {backend} grading failed for {name}: {res}"
                    )
                    self.stats["failed"] += 1
                    continue
                if backend == "gpt":
                    updates[cfg["gpt_score_field"]] = res["percentage"]
                    updates[cfg["gpt_summary_field"]] = res["summary"]
                    any_graded = True
                elif backend == "gem":
                    updates[cfg["gem_score_field"]] = res["percentage"]
                    updates[cfg["gem_summary_field"]] = res["summary"]
                    any_graded = True

        if not any_graded or not updates:
            self.stats["skipped"] += 1
            return

        # ✅ As soon as this call succeeds, the autorater fields are in Airtable
        try:
            await run_in_thread(self.air.update, rec_id, updates)
            logger.info(f"[Combined FILE] {rec_id}: ✅ Updated autorater fields.")
            self.stats["graded"] += 1
        except Exception as e:
            logger.error(f"[Combined FILE] {rec_id}: Airtable update failed: {e}")
            self.stats["failed"] += 1

    async def run(self):
        records = await self.fetch_records()
        if not records:
            logger.info("[Combined FILE] No file/image tasks found.")
            return

        # Sequential over records, tqdm for progress; concurrency is inside each record
        for r in tqdm(records, desc="Grading FILE records"):
            await self.process_record(r)

        logger.info(
            f"🎉 DONE [Combined FILE]. Processed={self.stats['processed']} "
            f"Graded={self.stats['graded']} Failed={self.stats['failed']} "
            f"Skipped={self.stats['skipped']}"
        )


async def main():
    await CombinedFileDualAutograder().run()


if __name__ == "__main__":
    asyncio.run(main())
