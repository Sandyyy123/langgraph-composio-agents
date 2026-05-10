# Project 5 - Recruiter Pipeline Assistant

## Summary
Parses resume, scores vs JD from Notion, books screening call if score>=70, updates Pipeline Notion DB.

## Composio apps to connect (one OAuth click each)
- [ ] Notion
- [ ] Google Calendar

## Setup steps

1. Sign up at https://composio.dev and copy your API key
2. Add to `.env`:
   ```
   COMPOSIO_API_KEY=cs_...
   OPENAI_API_KEY=sk-...
   ```
3. In Composio dashboard, click each app above and complete OAuth.
4. `pip install langgraph composio composio_langgraph langchain-openai python-dotenv langgraph-checkpoint-sqlite`

## Run

```bash
cd 05_recruiter_pipeline
python recruiter_pipeline.py
```

## Demo input baked into the script
Resume of Jane Q Developer (Senior Backend, 6 years, Python/FastAPI/AWS)

## Expected output
Parsed JSON, score 0-100, calendar invite next Tuesday 14:00 if qualified, Notion Pipeline row

## Files in this folder
- `recruiter_pipeline.py` - the working LangGraph + Composio agent
- `README.md` - this file
- `graph.png` (auto-generated on first run) - the supervisor + workers diagram
- `recruiter_pipeline.db` (auto-generated) - SQLite checkpoint store

## Architecture pattern

```
START -> supervisor -> [worker_1 | worker_2 | ...] -> supervisor -> ... -> END
```

Workers always return to the supervisor. The supervisor decides what's next based on state.
