# Markdown → Q&A / Summary Dataset Pipeline

Turns a Markdown textbook into an instruction-tuning dataset (Q&A + summary
pairs) using the **Gemini API**, saves it as JSONL, and pushes it to the
**Hugging Face Hub**.

The pipeline is **free-tier friendly**: it counts every API request, stops
before going over your daily quota, and can resume the next day exactly where
it stopped.

## What it does

1. Splits the Markdown into chunks (small chunks for Q&A, large chunks for summaries).
2. Sends each chunk to Gemini and asks for grounded Q&A / summary pairs.
3. Validates, de-duplicates, and writes records to `train.jsonl`.
4. Optionally uploads the dataset to Hugging Face.

## Setup

```bash
pip install google-genai python-dotenv datasets huggingface_hub pyyaml
```

Create a `.env` file in the project folder (this file is **git-ignored** and
must never be shared):

```
GEMINI_API_KEY=your_gemini_api_key_here
```

Edit `config.yaml` to set your paths and your Hugging Face `repo_id`.

## Usage

```bash
# See today's quota usage and how much work is left (uses NO API calls)
python 02_markdown_to_qa.py status

# Generate Q&A + summaries (stops automatically at the daily budget)
python 02_markdown_to_qa.py generate

# Upload the finished dataset to Hugging Face
python 02_markdown_to_qa.py push

# Generate and push in one go
python 02_markdown_to_qa.py generate --push
```

## Notes

- All settings live in `config.yaml` — you should not need to edit the script.
- `daily_request_limit` is a hard ceiling; the script never exceeds it.
- If it stops on the daily limit, just run the same command again after the
  quota resets — completed chunks are skipped automatically.