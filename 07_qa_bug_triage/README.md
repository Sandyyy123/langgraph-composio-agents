# Project 7 - QA Bug Triage Agent

## Summary
Classifies severity P0-P3, checks Linear for duplicates, fetches Drive screenshot, creates Linear issue, DMs reporter.

## Composio apps to connect (one OAuth click each)
- [ ] Linear
- [ ] Google Drive
- [ ] Slack

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
cd 07_qa_bug_triage
python qa_bug_triage.py
```

## Demo input baked into the script
Bug: 'Export to PDF freezes app for 30s then blank screen, blocks weekly report delivery'

## Expected output
Severity P1, dedup check, screenshot link, Linear issue ENG-XXXX, Slack DM to reporter

## Files in this folder
- `qa_bug_triage.py` - the working LangGraph + Composio agent
- `README.md` - this file
- `graph.png` (auto-generated on first run) - the supervisor + workers diagram
- `qa_bug_triage.db` (auto-generated) - SQLite checkpoint store

## Architecture pattern

```
START -> supervisor -> [worker_1 | worker_2 | ...] -> supervisor -> ... -> END
```

Workers always return to the supervisor. The supervisor decides what's next based on state.
