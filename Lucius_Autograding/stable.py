#!/usr/bin/env python3
"""
Multi-turn GPT-5 Autograder - Lucius EDU (batched + verbose, fixed)
- One model call per solution (fast) while keeping 10-sentence reasoning per criterion
- Structured JSON output for easy parsing
- Removes unsupported params (temperature/top_p) to avoid 400s
- Retries only transient errors; invalid-parameter 400s are treated as fatal
- Safe concurrency + multi-key throughput; robust single-criterion fallback retained
"""

import asyncio
import json
import logging
import os
import random
import re

from dotenv import load_dotenv
from openai import AsyncOpenAI
from pyairtable import Api

# ----------------------- Load .env secrets -----------------------
load_dotenv()
AIRTABLE_API_KEY = os.getenv("AIRTABLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not AIRTABLE_API_KEY:
    raise RuntimeError("Missing AIRTABLE_API_KEY in .env")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY in .env")

# Multiple keys for throughput (comma-separated)
OPENAI_API_KEYS = [
    k.strip()
    for k in os.getenv("OPENAI_API_KEYS", OPENAI_API_KEY).split(",")
    if k.strip()
]

# Tunables
OPENAI_MODEL = os.getenv(
    "OPENAI_MODEL", "gpt-5"
)  # keep your full model; switch to gpt-5-mini if desired
PER_KEY_PARALLEL = int(os.getenv("PER_KEY_PARALLEL", "4"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "4"))

# ----------------------- Logging -----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ----------------------- Airtable Config ------------------------
SOURCE_BASE_ID = os.environ.get("EVAL_BASE_ID", "appgeueGlH9mCUTvu")
SOURCE_TABLE_ID = os.environ.get("EVAL_TABLE_ID", "tblfy3EPxl1PHvKV7")
VIEW_ID = os.environ.get("EVAL_VIEW_ID", "viwxC835JCTcb42S0")

EVAL_INPUT_FIELDS = [
    "Eval (Web OFF) #1",
    "Eval (Web OFF) #2",
    "Eval (Web OFF) #3",
    "Eval (Web ON) #1",
    "Eval (Web ON) #2",
    "Eval (Web ON) #3",
]


def score_field_name(input_field: str) -> str:
    return f"GPT5 Autorater - {input_field} Score"


def summary_field_name(input_field: str) -> str:
    return f"[GPT5 graded] {input_field} Scoring Summary"


# ----------------------- Helper functions -----------------------
async def _with_retries(coro_factory, *, retries=MAX_RETRIES, base=0.5, jitter=0.25):
    """
    Exponential backoff for async ops.
    Treat invalid-parameter 4xx as fatal (no retries).
    """
    for attempt in range(retries):
        try:
            return await coro_factory()
        except Exception as e:
            msg = str(e)
            # Heuristics for invalid-request (400) / unsupported parameter: fatal
            is_param_error = (
                "invalid_request_error" in msg
                or "unsupported_value" in msg
                or "'param':" in msg
                or "HTTP/1.1 400" in msg
            )
            if is_param_error or attempt == retries - 1:
                raise
            sleep_time = base * (2**attempt) + random.uniform(0, jitter)
            logger.warning(f"Transient error ({e}), retrying in {sleep_time:.2f}s")
            await asyncio.sleep(sleep_time)


def _safe_json_loads(s: str):
    try:
        return json.loads(s)
    except Exception:
        return None


def _parse_boolean(x):
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        return x.strip().lower() == "true"
    return False


# ----------------------- Main Class -----------------------
class MultiTurnGPT5Autograder:
    def __init__(self):
        self.api = Api(AIRTABLE_API_KEY)
        self.source_table = self.api.table(SOURCE_BASE_ID, SOURCE_TABLE_ID)
        self.stats = {"processed": 0, "graded": 0, "failed": 0}

        self.clients = [AsyncOpenAI(api_key=k) for k in OPENAI_API_KEYS]
        self.key_index = 0
        self.key_lock = asyncio.Lock()
        # concurrency scaled by number of keys
        self.api_semaphore = asyncio.Semaphore(
            max(1, PER_KEY_PARALLEL * len(self.clients))
        )

        # System prompts
        # Batched: keep verbose reasoning (10 sentences) + decision, but insist on compact JSON
        self.system_prompt_batched = (
            "You are an expert grader evaluating solutions against specific criteria.\n"
            "For EACH criterion, do two things:\n"
            "1) Produce EXACTLY 10 sentences of reasoning explaining your evaluation.\n"
            "2) Output a boolean decision whether the solution meets the criterion.\n\n"
            "IMPORTANT OUTPUT FORMAT:\n"
            "Return ONLY a single JSON object of the shape:\n"
            "{\n"
            '  "<criterion_key>": {\n'
            '    "decision": true|false,\n'
            '    "reasoning": "Exactly 10 sentences."\n'
            "  }, ...\n"
            "}\n"
            "No extra keys, no markdown, no code fences."
        )

        # Single criterion fallback (kept terse but valid)
        self.system_prompt_single = (
            "You are a strict grader. Respond ONLY with 'FINAL_DECISION: true' or "
            "'FINAL_DECISION: false', followed by EXACTLY 10 sentences of reasoning."
        )

    # ------------------- Discovery -------------------
    def get_records_needing_grading(self) -> list:
        try:
            # Fetch only fields we need to reduce payload/latency
            records = self.source_table.all(
                view=VIEW_ID,
                fields=[
                    "Rubric JSON",
                    "MT Prompt",
                    *EVAL_INPUT_FIELDS,
                    *[score_field_name(c) for c in EVAL_INPUT_FIELDS],
                ],
            )
            logger.info(f"📊 Fetched {len(records)} total records from Airtable")
            needing = []
            for record in records:
                fields = record.get("fields", {})
                for col in EVAL_INPUT_FIELDS:
                    val = str(fields.get(col, "") or "").strip()
                    if val and not fields.get(score_field_name(col)):
                        needing.append(record)
                        break
            logger.info(f"🎯 Found {len(needing)} records needing grading")
            return needing
        except Exception as e:
            logger.error(f"Error fetching records: {e}")
            return []

    # ------------------- OpenAI call helpers -------------------
    async def _pick_client(self) -> AsyncOpenAI:
        async with self.key_lock:
            client = self.clients[self.key_index]
            self.key_index = (self.key_index + 1) % len(self.clients)
            return client

    async def _chat_json(self, messages):
        """
        Prefer JSON response_format for reliable parsing.
        Fallback to plain text if provider rejects the param.
        (No temperature/top_p to avoid model 400s.)
        """
        client = await self._pick_client()

        async def _call_json():
            return await client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                response_format={"type": "json_object"},
            )

        try:
            resp = await _with_retries(_call_json)
            content = resp.choices[0].message.content or "{}"
            data = _safe_json_loads(content)
            if data is None:
                # Some providers wrap JSON in extra text; try extracting the first {...}
                m = re.search(r"\{.*\}", content, re.DOTALL)
                data = _safe_json_loads(m.group(0)) if m else None
            return data if isinstance(data, dict) else None
        except Exception as e:
            logger.warning(f"JSON mode failed ({e}); retrying without response_format.")

            async def _call_text():
                return await client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=messages,
                )

            resp = await _with_retries(_call_text)
            return resp.choices[0].message.content or ""

    # ------------------- Fast batched grading -------------------
    async def grade_solution_batched(self, solution: str, rubric: list, prompt: str):
        """
        Make ONE model call to grade ALL criteria for the solution, preserving
        exactly 10-sentence reasoning per criterion.
        Returns: dict[criterion_key] -> {"decision": bool, "reasoning": str}
        """
        # Build {criterion_key: description}
        crit_map = {}
        for c in rubric:
            key = list(c.keys())[0]
            desc = (c[key] or {}).get("description", "")
            crit_map[key] = desc

        user_prompt = (
            "Evaluate the SOLUTION against each CRITERION.\n"
            "For each criterion, return `decision` (true/false) and `reasoning` (exactly 10 sentences).\n\n"
            f"ORIGINAL PROMPT:\n{prompt}\n\n"
            f"SOLUTION:\n{solution}\n\n"
            f"CRITERIA (JSON):\n{json.dumps(crit_map, ensure_ascii=False)}\n"
        )

        messages = [
            {"role": "system", "content": self.system_prompt_batched},
            {"role": "user", "content": user_prompt},
        ]

        async with self.api_semaphore:
            result = await self._chat_json(messages)

        # Parse decisions + reasoning
        decisions = {}
        if isinstance(result, dict):
            for key, obj in result.items():
                decision = _parse_boolean((obj or {}).get("decision"))
                reasoning = (obj or {}).get("reasoning", "") or ""
                decisions[key] = {"decision": decision, "reasoning": reasoning.strip()}
        else:
            # Fallback: try to recover JSON from text
            data = _safe_json_loads(result) or {}
            for key in crit_map.keys():
                obj = data.get(key, {}) or {}
                decision = _parse_boolean(obj.get("decision"))
                reasoning = (obj.get("reasoning") or "").strip()
                decisions[key] = {"decision": decision, "reasoning": reasoning}
        return decisions

    # ------------------- Slow single-criterion fallback -------------------
    async def grade_single_criterion(
        self, solution: str, criterion: dict, prompt: str
    ) -> dict:
        """
        Fallback path: one request per criterion (kept for robustness).
        Returns (decision: bool, reasoning: str)
        """
        async with self.api_semaphore:
            criterion_key = list(criterion.keys())[0]
            data = criterion[criterion_key]
            desc = data.get("description", "")

            grading_prompt = (
                "Does the SOLUTION meet the CRITERION? "
                "Respond ONLY with 'FINAL_DECISION: true' or 'FINAL_DECISION: false', "
                "THEN exactly 10 sentences of reasoning.\n\n"
                f"CRITERION:\n{desc}\n\n"
                f"ORIGINAL PROMPT:\n{prompt}\n\n"
                f"SOLUTION:\n{solution}\n"
            )

            client = await self._pick_client()

            async def _call():
                return await client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": self.system_prompt_single},
                        {"role": "user", "content": grading_prompt},
                    ],
                )

            try:
                resp = await _with_retries(_call)
                content = resp.choices[0].message.content or ""
                # Extract decision
                m = re.search(r"FINAL_DECISION:\s*(true|false)", content, re.IGNORECASE)
                decision = bool(m and m.group(1).lower() == "true")
                # Extract reasoning (everything after the decision line)
                lines = content.splitlines()
                if m:
                    idx = next(
                        (
                            i
                            for i, ln in enumerate(lines)
                            if "FINAL_DECISION" in ln.upper()
                        ),
                        -1,
                    )
                    reasoning_text = (
                        "\n".join(lines[idx + 1 :]).strip() if idx >= 0 else ""
                    )
                else:
                    reasoning_text = "\n".join(lines).strip()
                return decision, reasoning_text
            except Exception as e:
                logger.error(f"Error grading criterion: {e}")
                return False, ""

    # ------------------- Per-solution grading API -------------------
    async def grade_solution(self, solution: str, rubric: list, prompt: str) -> dict:
        """
        Primary path: batched grading (single model call, verbose reasoning kept).
        Fallback to per-criterion calls if batching fails.
        """
        # 1) Batched route
        try:
            decisions = await self.grade_solution_batched(solution, rubric, prompt)
            graded, total = [], 0
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
                total += int(ok)
            pct = (total / len(rubric) * 100) if rubric else 0
            return {
                "percentage": pct,
                "summary": json.dumps(graded, separators=(",", ":")),
            }
        except Exception as e:
            logger.warning(
                f"Batched verbose grading failed ({e}); falling back to per-criterion calls."
            )

        # 2) Fallback: per-criterion (still produce reasoning)
        tasks = [self.grade_single_criterion(solution, c, prompt) for c in rubric]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        graded, total = [], 0
        for c, res in zip(rubric, results):
            key = list(c.keys())[0]
            meta = c[key]
            if isinstance(res, Exception):
                ok, reasoning = False, ""
            else:
                ok, reasoning = res
            graded.append(
                {
                    "autorating": bool(ok),
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
            total += int(bool(ok))
        pct = (total / len(rubric) * 100) if rubric else 0
        return {"percentage": pct, "summary": json.dumps(graded, separators=(",", ":"))}

    # ------------------- Per-record flow -------------------
    async def process_record(self, record: dict) -> bool:
        fields = record.get("fields", {})
        record_id = record["id"]
        rubric_json = fields.get("Rubric JSON", "")
        prompt = str(fields.get("MT Prompt", "") or "")
        if not rubric_json:
            logger.warning(f"No rubric found for {record_id}")
            return False
        try:
            rubric = json.loads(rubric_json)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid rubric JSON: {e}")
            return False

        updates, graded_any = {}, False

        for col in EVAL_INPUT_FIELDS:
            text = str(fields.get(col, "") or "").strip()
            if not text:
                continue
            out_score = score_field_name(col)
            out_summary = summary_field_name(col)
            if fields.get(out_score):
                continue
            logger.info(f"📝 Grading '{col}' for {record_id}")
            result = await self.grade_solution(text, rubric, prompt)
            updates[out_score] = result["percentage"]
            updates[out_summary] = result["summary"]
            graded_any = True

        if updates:

            async def _update():
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(
                    None, self.source_table.update, record_id, updates
                )

            try:
                await _with_retries(_update)
                logger.info(f"✅ Updated record {record_id}")
                return graded_any
            except Exception as e:
                logger.error(f"Airtable update failed: {e}")
        return False

    # ------------------- Runner -------------------
    async def run(self):
        logger.info("🤖 Starting Multi-turn GPT-5 Autograder (batched+verbose, fixed)")
        records = self.get_records_needing_grading()
        if not records:
            logger.info("✅ No records need GPT-5 grading")
            return

        # Limit overall concurrency to be friendly to Airtable + OpenAI
        overall_limit = min(8, max(3, len(self.clients) * 2))
        semaphore = asyncio.Semaphore(overall_limit)

        async def process_with_semaphore(rec):
            async with semaphore:
                ok = await self.process_record(rec)
                self.stats["processed"] += 1
                if ok:
                    self.stats["graded"] += 1
                else:
                    self.stats["failed"] += 1

        await asyncio.gather(*(process_with_semaphore(r) for r in records))
        logger.info(
            f"🎉 Done. Processed={self.stats['processed']} "
            f"Graded={self.stats['graded']} Failed={self.stats['failed']}"
        )


# ------------------- Entrypoint -------------------
async def main():
    grader = MultiTurnGPT5Autograder()
    await grader.run()


if __name__ == "__main__":
    asyncio.run(main())
