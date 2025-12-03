#!/usr/bin/env python3
"""
Combined View Dual Autograder (TEXT + FILE/IMAGE) with tqdm
-----------------------------------------------------------
PRODUCTION MODE (no test slicing)

• Scans AIRTABLE_VIEW_COMBINED
• Grades:
    - Gemini 2.5 response (text + files)
    - Gemini 3.0 response (text + files)
    - GPT-5 response (text + files)
• Autoraters:
    - GPT on all three
    - Gemini on all three
• Handles missing file outputs by grading literal "no file generated"
• Runs multiple Airtable records concurrently
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
from tqdm.auto import tqdm

# ----------------------- Load ENV -----------------------
load_dotenv()

AIRTABLE_API_KEY = os.getenv("AIRTABLE_API_KEY")
AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID", "appgeueGlH9mCUTvu")
AIRTABLE_TABLE_ID = os.getenv("AIRTABLE_TABLE", "tblfy3EPxl1PHvKV7")
AIRTABLE_VIEW_COMBINED = os.getenv("AIRTABLE_VIEW_COMBINED", "viwliqVDSuuEXNxgN")

if not AIRTABLE_API_KEY:
    raise RuntimeError("Missing AIRTABLE_API_KEY in .env")

GPT_API_KEY = os.getenv("GPT_API_KEY")
if not GPT_API_KEY:
    raise RuntimeError("Missing GPT_API_KEY in .env")

GOOGLE_API_KEY = (
    os.getenv("GOOGLE_API_KEY")
    or os.getenv("GOOGLE_API_KEY_1")
    or os.getenv("GOOGLE_API_KEY_2")
)
if not GOOGLE_API_KEY:
    raise RuntimeError("Missing GOOGLE_API_KEY in .env")

GPT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3-pro-preview")

GPT_KEY_CONCURRENCY = int(os.getenv("GPT_KEY_CONCURRENCY", "5"))
GEMINI_KEY_CONCURRENCY = int(os.getenv("GEMINI_KEY_CONCURRENCY", "3"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "4"))

# Run up to N Airtable rows at once
MAX_PARALLEL_RECORDS = int(os.getenv("MAX_PARALLEL_RECORDS", "3"))

GEMINI_CACHE_DIR = Path(os.getenv("GEMINI_CACHE_DIR", "./gemini_cache_combined"))
GEMINI_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------- Logging -----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ----------------------- Column Names -----------------------
PROMPT_FIELD = "Consolidated Prompt - 10/25"
RUBRIC_FIELD = "Rubric JSON"

# TEXT responses
RESP_GEM25_FIELD = "Consolidated Gemini Response - 10/25"
RESP_GEM30_FIELD = "Gemini 3.0 model responses"
RESP_GPT5_FIELD = "GPT5 Response"

# FILE outputs
FILE_GEM25_FIELD = "Gemini Final Output Files"
FILE_GEM30_FIELD = "Gemini 3.0 Pro Response (File Output) - Gemini App"
FILE_GPT5_FIELD = "GPT5 Response (File Output)"

# Autorater field definitions
RESP_CONFIGS = [
    {
        "name": "Gemini 2.5",
        "text_field": RESP_GEM25_FIELD,
        "file_field": FILE_GEM25_FIELD,
        "gpt_score_field": "GPT5 Autorater - Gemini Response Score",
        "gpt_summary_field": "[GPT5 graded] Gemini Response Scoring Summary",
        "gem_score_field": "Gemini Autorater - Gemini Response Score",
        "gem_summary_field": "[Gemini graded] Gemini Response Scoring Summary",
    },
    {
        "name": "Gemini 3.0",
        "text_field": RESP_GEM30_FIELD,
        "file_field": FILE_GEM30_FIELD,
        "gpt_score_field": "GPT5 Autorater - Gemini 3.0 Response Score",
        "gpt_summary_field": "GPT5 Autorater - Gemini 3.0 Response Summary",
        "gem_score_field": "Gemini Autorater - Gemini 3.0 Response Score",
        "gem_summary_field": "Gemini Autorater - Gemini 3.0 Response Summary",
    },
    {
        "name": "GPT-5",
        "text_field": RESP_GPT5_FIELD,
        "file_field": FILE_GPT5_FIELD,
        "gpt_score_field": "GPT5 Autorater - GPT5 Response Score",
        "gpt_summary_field": "[GPT5 graded] GPT5 Response Scoring Summary",
        "gem_score_field": "Gemini Autorater - GPT5 Response Score",
        "gem_summary_field": "[Gemini graded] GPT5 Response Scoring Summary",
    },
]


# ============================================================================ #
#                        HELPER FUNCTIONS (unchanged)
# ============================================================================ #
def _safe_json(s: str):
    try:
        return json.loads(s)
    except Exception:
        return None

def _parse_boolean(x):
    if isinstance(x, bool): return x
    if isinstance(x, str): return x.strip().lower() == "true"
    return False

async def run_in_thread(fn, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, functools.partial(fn, *args, **kwargs))

async def _with_retries(fn, retries=MAX_RETRIES, base=0.5, jitter=0.25):
    for attempt in range(retries):
        try:
            return await fn()
        except Exception as e:
            msg = str(e)
            param_fail = any(k in msg for k in [
                "INVALID_ARGUMENT", "invalid_request_error",
                "invalid_image_format", "invalid_image_url", "HTTP/1.1 400"
            ])
            if param_fail or attempt == retries - 1:
                raise
            sleep = base * (2**attempt) + random.uniform(0, jitter)
            logger.warning(f"Transient error: {e} — retrying in {sleep:.2f}s")
            await asyncio.sleep(sleep)

def _attachment_cache_key(att):
    if att.get("id"): return att["id"]
    h = hashlib.sha1()
    h.update((att.get("url","") + "|" + att.get("filename","")).encode("utf-8"))
    return h.hexdigest()

def _attachment_cache_path(att):
    key = _attachment_cache_key(att)
    ext = Path(att.get("filename","file")).suffix
    return GEMINI_CACHE_DIR / f"{key}{ext}"

def _download_attachment_to_cache(att, max_bytes=20_000_000):
    url = att.get("url")
    if not url: return None
    path = _attachment_cache_path(att)
    if path.exists() and path.stat().st_size > 0:
        return path
    try:
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        if len(r.content) > max_bytes:
            return None
        path.write_bytes(r.content)
        return path
    except:
        return None

async def download_attachment_to_cache_async(att):
    return await run_in_thread(_download_attachment_to_cache, att)

def _describe_attachments_for_text(attachments):
    return "\n".join(f"- {a.get('filename','file')} (mime={a.get('type')})"
                     for a in attachments if isinstance(a, dict))


# ============================================================================ #
#                    MAIN CLASS — FULL LOGIC (unchanged but no slice)
# ============================================================================ #
class CombinedDualAutograder:
    def __init__(self):
        self.air = Api(AIRTABLE_API_KEY).table(AIRTABLE_BASE_ID, AIRTABLE_TABLE_ID)

        self.gpt_client = AsyncOpenAI(api_key=GPT_API_KEY)
        self.gem_client = genai.Client(api_key=GOOGLE_API_KEY)

        self.global_sem = asyncio.Semaphore(max(4, GPT_KEY_CONCURRENCY + GEMINI_KEY_CONCURRENCY))
        self.gpt_sem = asyncio.Semaphore(GPT_KEY_CONCURRENCY)
        self.gem_sem = asyncio.Semaphore(GEMINI_KEY_CONCURRENCY)

        self.stats = {"processed":0,"graded":0,"failed":0,"skipped":0}

        # ---- prompts unchanged ----
        self.text_system_prompt = (
            "You are an expert grader evaluating solutions against criteria.\n"
            "For EACH criterion:\n"
            "1) Produce EXACTLY 10 sentences.\n"
            "2) Output a boolean decision.\n\n"
            "Return ONLY JSON..."
        )
        self.file_system_prompt_gpt = self.text_system_prompt
        self.file_system_prompt_gemini = (
            "You are an expert grader evaluating images/files.\n"
            "Give EXACTLY 10 sentences per criterion.\n"
            "Return ONLY JSON..."
        )

    # ---------------------------------------------------------------------- #
    # GET RECORDS
    # ---------------------------------------------------------------------- #
    def get_records(self):
        fields = [PROMPT_FIELD, RUBRIC_FIELD]
        for cfg in RESP_CONFIGS:
            fields.extend([
                cfg["text_field"], cfg["file_field"],
                cfg["gpt_score_field"], cfg["gpt_summary_field"],
                cfg["gem_score_field"], cfg["gem_summary_field"],
            ])

        all_records = self.air.all(view=AIRTABLE_VIEW_COMBINED, fields=list(set(fields)))
        logger.info(f"[Combined] Fetched {len(all_records)} total records")

        needing = []
        for rec in all_records:
            f = rec.get("fields",{})
            rubric = (f.get(RUBRIC_FIELD) or "").strip()
            if not rubric: continue

            for cfg in RESP_CONFIGS:
                text = (f.get(cfg["text_field"]) or "").strip()
                files = f.get(cfg["file_field"])
                files_ok = isinstance(files, list) and len(files)>0

                gpt_score = f.get(cfg["gpt_score_field"])
                gpt_sum   = f.get(cfg["gpt_summary_field"])
                gem_score = f.get(cfg["gem_score_field"])
                gem_sum   = f.get(cfg["gem_summary_field"])

                if (text or files_ok) and (
                    gpt_score is None or gpt_sum is None or
                    gem_score is None or gem_sum is None
                ):
                    needing.append(rec)
                    break

        logger.info(f"[Combined] Found {len(needing)} needing grading")
        return needing

    # ---------------------------------------------------------------------- #
    # GPT text JSON call
    # ---------------------------------------------------------------------- #
    async def _gpt_chat_json(self, messages):
        async def do():
            async with self.global_sem, self.gpt_sem:
                return await self.gpt_client.chat.completions.create(
                    model=GPT_MODEL, messages=messages,
                    response_format={"type":"json_object"}
                )
        try:
            resp = await _with_retries(do)
            data = _safe_json(resp.choices[0].message.content or "{}")
            return data if isinstance(data, dict) else None
        except:
            # Try non-JSON mode
            async def fallback():
                async with self.global_sem, self.gpt_sem:
                    return await self.gpt_client.chat.completions.create(
                        model=GPT_MODEL, messages=messages)
            resp = await _with_retries(fallback)
            return _safe_json(resp.choices[0].message.content or "{}")

    # ---------------------------------------------------------------------- #
    # Gemini text call
    # ---------------------------------------------------------------------- #
    async def _gemini_chat_json(self, messages):
        prompt = "\n".join(
            f"{m['role'].upper()}:\n{m['content']}\n"
            for m in messages
        )

        async def do():
            async with self.global_sem, self.gem_sem:
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(
                    None,
                    lambda: self.gem_client.models.generate_content(
                        model=GEMINI_MODEL, contents=prompt
                    )
                )
        resp = await _with_retries(do)

        text = getattr(resp, "text", None)
        if not text:
            cands = getattr(resp, "candidates", [])
            if cands:
                parts = cands[0].content.parts
                text = "\n".join(p.text for p in parts if getattr(p,"text",None))
        data = _safe_json(text or "{}")
        return data if isinstance(data, dict) else {}

    # ---------------------------------------------------------------------- #
    # TEXT GRADING (GPT/Gemini)
    # ---------------------------------------------------------------------- #
    async def grade_text(self, backend, solution, rubric, prompt):
        crit_map = {list(c.keys())[0]: (c[list(c.keys())[0]] or {}).get("description","")
                    for c in rubric if isinstance(c,dict) and c}

        user_prompt = (
            "Evaluate the SOLUTION against each CRITERION.\n"
            "For each criterion: 10 sentences + boolean decision.\n\n"
            f"PROMPT:\n{prompt}\n\n"
            f"SOLUTION:\n{solution}\n\n"
            f"CRITERIA:\n{json.dumps(crit_map, ensure_ascii=False)}"
        )

        messages = [
            {"role":"system","content":self.text_system_prompt},
            {"role":"user","content":user_prompt},
        ]

        if backend=="gpt":
            data = await self._gpt_chat_json(messages)
        else:
            data = await self._gemini_chat_json(messages)

        if not isinstance(data, dict): data = {}

        graded = []
        true_count = 0
        for c in rubric:
            if not isinstance(c, dict) or not c: continue
            key = list(c.keys())[0]
            meta = c.get(key,{})
            entry = data.get(key,{})
            decision = _parse_boolean(entry.get("decision"))
            reasoning = entry.get("reasoning","")
            graded.append({
                "autorating": decision,
                "description": meta.get("description",""),
                "weight": meta.get("weight",""),
                "criterion_type": meta.get("criterion_type",[]),
                "dependent_criteria": meta.get("dependent_criteria",[]),
                "justification": meta.get("justification",""),
                "sources": meta.get("sources",""),
                "human_rating": meta.get("human_rating",""),
                "reasoning": reasoning,
            })
            if decision: true_count+=1

        pct = (true_count / len(graded) * 100) if graded else 0
        return {"percentage":pct, "summary":json.dumps(graded, separators=(",",":"))}

    # ---------------------------------------------------------------------- #
    # FILE GRADING (GPT/Gemini)
    # ---------------------------------------------------------------------- #
    async def grade_files_gpt(self, prompt, rubric, attachments):
        crit_map = {list(c.keys())[0]: (c[list(c.keys())[0]] or {}).get("description","")
                    for c in rubric if isinstance(c,dict) and c}

        text_intro = (
            "You are grading a FILE-based solution.\n"
            "For each criterion: 10 sentences + boolean.\n\n"
            f"PROMPT:\n{prompt}\n\n"
            f"CRITERIA:\n{json.dumps(crit_map)}\n"
        )

        content = [{"type":"text","text":text_intro}]
        for att in attachments:
            if isinstance(att, dict) and att.get("url"):
                content.append({"type":"image_url","image_url":{"url":att["url"]}})

        messages=[
            {"role":"system","content":self.file_system_prompt_gpt},
            {"role":"user","content":content},
        ]

        async def do():
            async with self.global_sem, self.gpt_sem:
                return await self.gpt_client.chat.completions.create(
                    model=GPT_MODEL,
                    messages=messages,
                    response_format={"type":"json_object"},
                )
        resp = await _with_retries(do)
        data = _safe_json(resp.choices[0].message.content or "{}")
        if not isinstance(data, dict): data={}

        # Convert to standard summary structure
        graded=[]
        true_count=0
        for c in rubric:
            key=list(c.keys())[0]
            meta=c.get(key,{})
            entry=data.get(key,{})
            decision=_parse_boolean(entry.get("decision"))
            graded.append({
                "autorating": decision,
                "description": meta.get("description",""),
                "weight": meta.get("weight",""),
                "criterion_type": meta.get("criterion_type",[]),
                "dependent_criteria": meta.get("dependent_criteria",[]),
                "justification": meta.get("justification",""),
                "sources": meta.get("sources",""),
                "human_rating": meta.get("human_rating",""),
                "reasoning": entry.get("reasoning",""),
            })
            if decision: true_count+=1
        pct=(true_count/len(graded)*100) if graded else 0
        return {"percentage":pct,"summary":json.dumps(graded,separators=(",",":"))}

    async def grade_files_gemini(self, prompt, rubric, attachments):
        crit_map = {list(c.keys())[0]: (c[list(c.keys())[0]] or {}).get("description","")
                    for c in rubric if isinstance(c,dict) and c}

        intro = (
            "You are grading FILES.\n"
            "10 sentences + boolean per criterion.\n\n"
            f"PROMPT:\n{prompt}\n\n"
            f"CRITERIA:\n{json.dumps(crit_map)}\n"
            f"ATTACHMENTS:\n{_describe_attachments_for_text(attachments)}"
        )

        parts=[{"text":intro}]
        # upload attachments to Gemini
        uploaded=[]
        for att in attachments:
            p=await download_attachment_to_cache_async(att)
            if not p: continue
            mime=att.get("type","application/octet-stream")
            async with self.gem_sem:
                up=await run_in_thread(
                    self.gem_client.files.upload,
                    file=str(p),
                    config={"mime_type":mime}
                )
            uploaded.append({"fileData":{"fileUri":up.uri,"mimeType":mime}})
        parts += uploaded

        async def do():
            async with self.gem_sem:
                return await run_in_thread(
                    self.gem_client.models.generate_content,
                    model=GEMINI_MODEL,
                    contents=[{"role":"user","parts":parts}],
                    config={
                        "system_instruction":{"parts":[{"text":self.file_system_prompt_gemini}]}
                    }
                )
        resp = await _with_retries(do)
        text = getattr(resp,"text",None)
        if not text:
            cands=getattr(resp,"candidates",[])
            if cands:
                text="\n".join(
                    p.text for p in cands[0].content.parts if getattr(p,"text",None)
                )
        data=_safe_json(text or "{}")
        if not isinstance(data,dict): data={}

        graded=[]
        true_count=0
        for c in rubric:
            key=list(c.keys())[0]
            meta=c.get(key,{})
            entry=data.get(key,{})
            decision=_parse_boolean(entry.get("decision"))
            graded.append({
                "autorating":decision,
                "description":meta.get("description",""),
                "weight":meta.get("weight",""),
                "criterion_type":meta.get("criterion_type",[]),
                "dependent_criteria":meta.get("dependent_criteria",[]),
                "justification":meta.get("justification",""),
                "sources":meta.get("sources",""),
                "human_rating":meta.get("human_rating",""),
                "reasoning":entry.get("reasoning",""),
            })
            if decision: true_count+=1
        pct=(true_count/len(graded)*100) if graded else 0
        return {"percentage":pct,"summary":json.dumps(graded,separators=(",",":"))}

    # ---------------------------------------------------------------------- #
    # PROCESS RECORD
    # ---------------------------------------------------------------------- #
    async def process_record(self, rec):
        self.stats["processed"]+=1
        f=rec.get("fields",{})
        rec_id=rec["id"]

        rubric_raw=(f.get(RUBRIC_FIELD) or "").strip()
        if not rubric_raw:
            self.stats["skipped"]+=1
            return
        rubric=_safe_json(rubric_raw)
        if not isinstance(rubric,list):
            self.stats["skipped"]+=1
            return

        prompt=f.get(PROMPT_FIELD) or ""
        updates={}
        any_graded=False

        for cfg in RESP_CONFIGS:
            name=cfg["name"]
            text_solution=(f.get(cfg["text_field"]) or "").strip()
            attachments=f.get(cfg["file_field"])
            if not isinstance(attachments,list):
                attachments=[]

            need_gpt = f.get(cfg["gpt_score_field"]) is None or f.get(cfg["gpt_summary_field"]) is None
            need_gem = f.get(cfg["gem_score_field"]) is None or f.get(cfg["gem_summary_field"]) is None

            if not need_gpt and not need_gem:
                continue

            # If both empty → force text: "no file generated"
            if not text_solution and not attachments:
                text_solution="no file generated"

            if not text_solution and not attachments:
                continue

            logger.info(f"[Combined] Grading record {rec_id} for {name}")

            tasks=[]
            # TEXT grading
            if text_solution:
                if need_gpt:
                    tasks.append(("gpt_text",
                                  self.grade_text("gpt",text_solution,rubric,prompt)))
                if need_gem:
                    tasks.append(("gem_text",
                                  self.grade_text("gemini",text_solution,rubric,prompt)))

            # FILE grading
            if attachments:
                if need_gpt:
                    tasks.append(("gpt_file",
                                  self.grade_files_gpt(prompt,rubric,attachments)))
                if need_gem:
                    tasks.append(("gem_file",
                                  self.grade_files_gemini(prompt,rubric,attachments)))

            results=await asyncio.gather(*[t[1] for t in tasks],return_exceptions=True)

            gpt_res=None
            gem_res=None
            for (tag,_),res in zip(tasks,results):
                if isinstance(res,Exception):
                    logger.error(f"[Combined] {rec_id}: {tag} failed: {res}")
                    self.stats["failed"]+=1
                    continue
                if tag.startswith("gpt_"):
                    gpt_res=res
                else:
                    gem_res=res

            if gpt_res and need_gpt:
                updates[cfg["gpt_score_field"]] = gpt_res["percentage"]
                updates[cfg["gpt_summary_field"]] = gpt_res["summary"]
                any_graded=True

            if gem_res and need_gem:
                updates[cfg["gem_score_field"]] = gem_res["percentage"]
                updates[cfg["gem_summary_field"]] = gem_res["summary"]
                any_graded=True

        if not any_graded:
            self.stats["skipped"]+=1
            return

        try:
            self.air.update(rec_id, updates)
            logger.info(f"[Combined] ✅ Updated {rec_id}")
            self.stats["graded"]+=1
        except Exception as e:
            logger.error(f"[Combined] Airtable update fail {rec_id}: {e}")
            self.stats["failed"]+=1

    # ---------------------------------------------------------------------- #
    # RUNNER — NOW PRODUCTION MODE
    # ---------------------------------------------------------------------- #
    async def run(self):
        logger.info("🤖 Starting Combined Autograder (FULL RUN)")
        records=self.get_records()
        if not records:
            logger.info("No records need grading.")
            return

        total=len(records)
        logger.info(f"[Combined] Processing {total} records...")

        sem=asyncio.Semaphore(MAX_PARALLEL_RECORDS)

        async def worker(r):
            async with sem:
                await self.process_record(r)

        tasks=[asyncio.create_task(worker(r)) for r in records]

        with tqdm(total=total, desc="Grading records") as bar:
            for coro in asyncio.as_completed(tasks):
                await coro
                bar.update(1)

        logger.info(
            f"🎉 DONE. Processed={self.stats['processed']} "
            f"Graded={self.stats['graded']} Failed={self.stats['failed']} "
            f"Skipped={self.stats['skipped']}"
        )


# -------------------------------------------------------------------------- #
# ENTRY POINT
# -------------------------------------------------------------------------- #
async def main():
    await CombinedDualAutograder().run()

if __name__=="__main__":
    asyncio.run(main())
