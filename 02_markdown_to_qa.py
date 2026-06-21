"""
02_markdown_to_qa.py
======================
Markdown files -> Q&A + Summary pairs (Gemini API) -> JSONL -> Hugging Face Hub

Uses the NEW `google-genai` SDK (the old `google.generativeai` is deprecated).

Features:
  - HARD DAILY BUDGET: every API call (success OR failure/retry, because
    Google counts those too) is counted, and the script stops the instant it
    reaches `daily_request_limit`. Your real usage can never exceed it.
  - RESUME SUPPORT: progress is saved to progress.json after every chunk, so
    you can stop and continue later without redoing or losing work.

All tunable parameters live in config.yaml (same folder).

USAGE
-----
    python 02_markdown_to_qa.py generate
    python 02_markdown_to_qa.py generate --push
    python 02_markdown_to_qa.py push
    python 02_markdown_to_qa.py status        <- today's quota usage / remaining work

REQUIRES
--------
    pip install google-genai python-dotenv datasets huggingface_hub pyyaml
"""

import os
import re
import json
import time
import glob
import hashlib
import argparse
import datetime
from pathlib import Path

import yaml

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ---------------------------------------------------------------------------
# CONFIG LOADING
# ---------------------------------------------------------------------------
def load_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        raise SystemExit(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# PROGRESS / QUOTA TRACKING (enables resume + daily-limit awareness)
# ---------------------------------------------------------------------------
def today_str() -> str:
    return datetime.date.today().isoformat()


def load_progress(progress_path: str) -> dict:
    if not os.path.exists(progress_path):
        return {
            "date": today_str(),
            "requests_used_today": 0,
            "completed_tasks": [],
            "seen_instruction_hashes": [],
        }

    with open(progress_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Reset the daily counter if it's a new day; keep history (resume + dedup)
    if data.get("date") != today_str():
        data["date"] = today_str()
        data["requests_used_today"] = 0

    data.setdefault("completed_tasks", [])
    data.setdefault("seen_instruction_hashes", [])
    return data


def save_progress(progress_path: str, data: dict):
    with open(progress_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def task_id(source_name: str, kind: str, chunk_index: int) -> str:
    """Unique id for a (book, qa/summary, chunk#) unit of work -- used for resume."""
    return f"{source_name}::{kind}::{chunk_index}"


# ---------------------------------------------------------------------------
# CHUNKING
# ---------------------------------------------------------------------------
def chunk_text(text: str, chunk_size: int, overlap: int, min_words: int):
    words = text.split()
    if not words:
        return []
    # Guard: if overlap >= chunk_size the step would be <= 0 and loop forever.
    step = max(1, chunk_size - overlap)
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if len(chunk.split()) >= min_words:
            chunks.append(chunk)
        start += step
    return chunks


# ---------------------------------------------------------------------------
# GEMINI GENERATION
# ---------------------------------------------------------------------------
QA_PROMPT_TEMPLATE = """You are building an instruction-tuning dataset from a textbook. Given the TEXT below, \
generate UP TO {max_n} instruction/output Q&A pairs that are answerable using ONLY this text. \
Use your judgment: if the text is dense with distinct facts/concepts, generate more (up to {max_n}); \
if it's thin or repetitive, generate fewer (1-2) rather than inventing filler questions. \
Vary instruction style: direct questions, "Explain...", "List...", "What is...".

Rules:
- Every output must be a complete, accurate answer grounded ONLY in the TEXT below.
- Do not invent facts not present in the TEXT.
- Return ONLY valid JSON (a list of objects), nothing else. No markdown fences, no preamble.

Format:
[
  {{"instruction": "...", "input": "", "output": "...", "type": "qa"}},
  ...
]

TEXT:
\"\"\"
{chunk}
\"\"\"
"""

SUMMARY_PROMPT_TEMPLATE = """You are building an instruction-tuning dataset from a textbook. Given the TEXT below, \
generate ONE summary instruction/output pair: the instruction should ask to summarize or explain \
this section, and the output should be a clear, complete summary (4-8 sentences) grounded ONLY \
in the TEXT below.

Return ONLY valid JSON (a list with exactly one object), nothing else. No markdown fences, no preamble.

Format:
[
  {{"instruction": "...", "input": "", "output": "...", "type": "summary"}}
]

TEXT:
\"\"\"
{chunk}
\"\"\"
"""


class QuotaExhausted(Exception):
    """Raised when the daily request budget has been used up."""
    pass


def _extract_text(response):
    """Safely pull text out of a Gemini response; return None if empty/blocked."""
    try:
        if getattr(response, "text", None):
            return response.text
    except Exception:
        pass
    try:
        parts = response.candidates[0].content.parts
        text = "".join(getattr(p, "text", "") or "" for p in parts)
        return text or None
    except Exception:
        return None


def _strip_fences(raw_text: str) -> str:
    t = raw_text.strip()
    t = re.sub(r"^```(json)?", "", t).strip()
    t = re.sub(r"```$", "", t).strip()
    return t


def call_gemini(client, model_name, gen_config, prompt, rate_cfg, progress, progress_path) -> list:
    """
    Calls Gemini (new google-genai SDK), retrying only on per-minute rate limits.

    Every attempt is counted against today's budget BEFORE the call, because
    Google counts failed/retried calls too. This keeps our counter honest so
    real usage never silently exceeds the free cap.

    Returns parsed records ([] on parse failure / non-rate-limit error),
    or raises QuotaExhausted when the budget is reached.
    """
    budget = rate_cfg["daily_request_limit"]
    backoff = rate_cfg["initial_backoff_seconds"]

    for attempt in range(1, rate_cfg["max_retries"] + 1):
        if progress["requests_used_today"] >= budget:
            raise QuotaExhausted()

        # Count the attempt up-front (conservative: protects your daily cap).
        progress["requests_used_today"] += 1
        save_progress(progress_path, progress)

        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=gen_config,
            )

            raw_text = _extract_text(response)
            if raw_text is None:
                print("  [warn] Empty/blocked response, skipping this chunk")
                return []

            cleaned = _strip_fences(raw_text)
            records = json.loads(cleaned)
            return records if isinstance(records, list) else []

        except json.JSONDecodeError:
            print("  [warn] Model output wasn't valid JSON, skipping this chunk")
            return []

        except Exception as e:
            err = str(e).lower()
            is_daily_quota = ("generaterequestsperdayperprojectpermodel" in err
                              or "free_tier" in err
                              or "perday" in err
                              or "per day" in err)
            is_rate_limit = ("429" in err
                             or "quota" in err
                             or "rate" in err
                             or "resource_exhausted" in err)

            if is_daily_quota:
                print("  [quota] Daily free-tier limit reached (confirmed by API).")
                progress["requests_used_today"] = budget
                save_progress(progress_path, progress)
                raise QuotaExhausted()

            if is_rate_limit and attempt < rate_cfg["max_retries"]:
                print(f"  [rate limit] Attempt {attempt}/{rate_cfg['max_retries']}. Waiting {backoff}s...")
                time.sleep(backoff)
                backoff *= rate_cfg["backoff_multiplier"]
                continue

            print(f"  [error] Gemini call failed: {e}")
            return []

    return []


# ---------------------------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------------------------
def validate_and_clean(records: list, source_name: str, val_cfg: dict, seen_hashes: set) -> list:
    clean = []
    for r in records:
        if not isinstance(r, dict):
            continue
        instruction = str(r.get("instruction", "")).strip()
        output = str(r.get("output", "")).strip()
        input_field = str(r.get("input", "")).strip()
        record_type = str(r.get("type", "qa")).strip()

        if not instruction or not output:
            continue
        if len(output.split()) < val_cfg["min_output_words"]:
            continue

        if val_cfg.get("deduplicate", True):
            dedup_key = hashlib.md5(instruction.lower().encode()).hexdigest()
            if dedup_key in seen_hashes:
                continue
            seen_hashes.add(dedup_key)

        clean.append({
            "instruction": instruction,
            "input": input_field,
            "output": output,
            "type": record_type,
            "source": source_name,
        })
    return clean


# ---------------------------------------------------------------------------
# SAVE
# ---------------------------------------------------------------------------
def append_jsonl(records: list, path: str):
    if not records:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# GENERATE COMMAND
# ---------------------------------------------------------------------------
def run_generate(cfg: dict, do_push: bool):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY not set. Add it to your .env file.")

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    gen_cfg = cfg["generation"]
    model_name = gen_cfg["model"]
    gen_config = types.GenerateContentConfig(
        temperature=gen_cfg.get("temperature", 0.7),
        max_output_tokens=gen_cfg.get("max_output_tokens", 65536),
        response_mime_type="application/json",  # force valid JSON, no markdown fences
    )

    paths = cfg["paths"]
    input_dir = paths["markdown_dir"]
    output_dir = paths["output_dir"]
    combined_path = os.path.join(output_dir, paths["combined_filename"])
    os.makedirs(output_dir, exist_ok=True)

    progress_path = os.path.join(output_dir, cfg["rate_limit"]["progress_file"])
    progress = load_progress(progress_path)
    completed = set(progress["completed_tasks"])
    seen_hashes = set(progress["seen_instruction_hashes"])  # persists across runs

    md_files = sorted(glob.glob(os.path.join(input_dir, "*.md")))
    if not md_files:
        raise SystemExit(f"No .md files found in {input_dir}")

    chunk_cfg = cfg["chunking"]
    rate_cfg = cfg["rate_limit"]
    val_cfg = cfg["validation"]
    daily_limit = rate_cfg["daily_request_limit"]

    print(f"Today's quota usage so far: {progress['requests_used_today']}/{daily_limit}")
    if progress["requests_used_today"] >= daily_limit:
        print("Daily quota already used up. Run this again after the quota resets (next day).")
        return

    grand_total = 0
    quota_hit = False

    def persist_after_chunk(tid):
        progress["completed_tasks"].append(tid)
        completed.add(tid)
        progress["seen_instruction_hashes"] = list(seen_hashes)
        save_progress(progress_path, progress)

    try:
        for md_path in md_files:
            source_name = Path(md_path).stem
            per_book_path = os.path.join(output_dir, f"{source_name}.jsonl")

            with open(md_path, "r", encoding="utf-8") as f:
                text = f.read()

            qa_chunks = chunk_text(
                text,
                chunk_size=chunk_cfg["qa"]["chunk_size_words"],
                overlap=chunk_cfg["qa"]["overlap_words"],
                min_words=chunk_cfg["qa"]["min_chunk_words"],
            )
            summary_chunks = chunk_text(
                text,
                chunk_size=chunk_cfg["summary"]["chunk_size_words"],
                overlap=chunk_cfg["summary"]["overlap_words"],
                min_words=chunk_cfg["summary"]["min_chunk_words"],
            )

            print(f"\n=== {source_name} === "
                  f"{len(qa_chunks)} QA-chunks, {len(summary_chunks)} summary-chunks")

            book_total = 0

            # --- Q&A pass ---
            if gen_cfg["generate_qa"]:
                for i, chunk in enumerate(qa_chunks):
                    tid = task_id(source_name, "qa", i)
                    if tid in completed:
                        continue

                    print(f"[QA {i + 1}/{len(qa_chunks)}] {source_name}...")
                    prompt = QA_PROMPT_TEMPLATE.format(max_n=gen_cfg["max_qa_per_chunk"], chunk=chunk)
                    raw = call_gemini(client, model_name, gen_config, prompt, rate_cfg, progress, progress_path)
                    clean = validate_and_clean(raw, source_name, val_cfg, seen_hashes)

                    append_jsonl(clean, per_book_path)
                    append_jsonl(clean, combined_path)
                    book_total += len(clean)
                    grand_total += len(clean)
                    print(f"  -> {len(clean)} records (book total: {book_total}, "
                          f"quota used: {progress['requests_used_today']}/{daily_limit})")

                    persist_after_chunk(tid)
                    time.sleep(rate_cfg["seconds_between_requests"])

            # --- Summary pass ---
            if gen_cfg["generate_summary"]:
                for i, chunk in enumerate(summary_chunks):
                    tid = task_id(source_name, "summary", i)
                    if tid in completed:
                        continue

                    print(f"[Summary {i + 1}/{len(summary_chunks)}] {source_name}...")
                    prompt = SUMMARY_PROMPT_TEMPLATE.format(chunk=chunk)
                    raw = call_gemini(client, model_name, gen_config, prompt, rate_cfg, progress, progress_path)
                    clean = validate_and_clean(raw, source_name, val_cfg, seen_hashes)

                    append_jsonl(clean, per_book_path)
                    append_jsonl(clean, combined_path)
                    book_total += len(clean)
                    grand_total += len(clean)
                    print(f"  -> {len(clean)} records (book total: {book_total}, "
                          f"quota used: {progress['requests_used_today']}/{daily_limit})")

                    persist_after_chunk(tid)
                    time.sleep(rate_cfg["seconds_between_requests"])

            print(f"  Finished {source_name} for this run: {book_total} new records this session")

    except QuotaExhausted:
        quota_hit = True
        print(f"\n[STOPPED] Daily budget ({daily_limit} requests) reached for today.")
        print(f"Progress saved to {progress_path}.")
        print("Run the same command again after the quota resets to continue -- "
              "already-completed chunks are skipped automatically.")

    print(f"\nThis session: {grand_total} new records generated.")
    print(f"Combined file so far: {combined_path}")

    if quota_hit:
        return

    if do_push:
        run_push(cfg, combined_path)


# ---------------------------------------------------------------------------
# STATUS COMMAND — see today's usage and remaining work without making calls
# ---------------------------------------------------------------------------
def run_status(cfg: dict):
    paths = cfg["paths"]
    output_dir = paths["output_dir"]
    progress_path = os.path.join(output_dir, cfg["rate_limit"]["progress_file"])
    progress = load_progress(progress_path)
    daily_limit = cfg["rate_limit"]["daily_request_limit"]

    print(f"Date: {progress['date']}")
    print(f"Requests used today: {progress['requests_used_today']}/{daily_limit}")
    print(f"Total chunks completed (all-time, across runs): {len(progress['completed_tasks'])}")

    input_dir = paths["markdown_dir"]
    md_files = sorted(glob.glob(os.path.join(input_dir, "*.md")))
    if not md_files:
        print(f"\n(No .md files found in {input_dir})")
        return

    chunk_cfg = cfg["chunking"]
    completed = set(progress["completed_tasks"])

    total_remaining = 0
    for md_path in md_files:
        source_name = Path(md_path).stem
        with open(md_path, "r", encoding="utf-8") as f:
            text = f.read()
        qa_chunks = chunk_text(text, chunk_cfg["qa"]["chunk_size_words"],
                               chunk_cfg["qa"]["overlap_words"], chunk_cfg["qa"]["min_chunk_words"])
        summary_chunks = chunk_text(text, chunk_cfg["summary"]["chunk_size_words"],
                                    chunk_cfg["summary"]["overlap_words"], chunk_cfg["summary"]["min_chunk_words"])

        qa_done = sum(1 for i in range(len(qa_chunks)) if task_id(source_name, "qa", i) in completed)
        sum_done = sum(1 for i in range(len(summary_chunks)) if task_id(source_name, "summary", i) in completed)
        remaining = (len(qa_chunks) - qa_done) + (len(summary_chunks) - sum_done)
        total_remaining += remaining

        print(f"  {source_name}: QA {qa_done}/{len(qa_chunks)} done, "
              f"Summary {sum_done}/{len(summary_chunks)} done, {remaining} remaining "
              f"(total this file = {len(qa_chunks) + len(summary_chunks)} requests)")

    print(f"\nTotal remaining chunks (= remaining API calls needed): {total_remaining}")
    if daily_limit > 0:
        days_needed = (total_remaining + daily_limit - 1) // daily_limit if total_remaining else 0
        print(f"At {daily_limit} requests/day, this needs ~{days_needed} more day(s) to finish.")


# ---------------------------------------------------------------------------
# PUSH COMMAND
# ---------------------------------------------------------------------------
def run_push(cfg: dict, jsonl_path: str = None):
    from datasets import load_dataset

    hf_cfg = cfg["huggingface"]
    jsonl_path = jsonl_path or os.path.join(
        cfg["paths"]["output_dir"], cfg["paths"]["combined_filename"]
    )

    if not os.path.exists(jsonl_path):
        raise SystemExit(f"File not found: {jsonl_path}")

    repo_id = hf_cfg["repo_id"]
    if "your-username" in repo_id:
        raise SystemExit("Set huggingface.repo_id in config.yaml before pushing.")

    dataset = load_dataset("json", data_files=jsonl_path, split="train")
    print(f"Loaded {len(dataset)} records from {jsonl_path}")

    dataset.push_to_hub(repo_id, private=hf_cfg.get("private", False))
    print(f"Pushed to https://huggingface.co/datasets/{repo_id}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Markdown -> Q&A/Summary JSONL -> Hugging Face Hub (config-driven)")
    parser.add_argument("command", choices=["generate", "push", "status"], help="Which step to run")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--push", action="store_true", help="With 'generate': also push to Hugging Face after")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.command == "generate":
        run_generate(cfg, do_push=args.push)
    elif args.command == "push":
        run_push(cfg)
    elif args.command == "status":
        run_status(cfg)


if __name__ == "__main__":
    main()