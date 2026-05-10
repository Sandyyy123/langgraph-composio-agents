# Project 9 - Compliance Document Reviewer

## Summary
Drive file -> Notion checklist match -> drafts email -> HITL interrupt -> Gmail send -> 7-day Calendar reminder.

## Composio apps to connect (one OAuth click each)
- [ ] Google Drive
- [ ] Notion
- [ ] Gmail
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

## Extra setup
```
Set NOTION_COMPLIANCE_DB_ID in .env. Note: graph PAUSES at email_sender via interrupt_before. Demo shows resume with Command(resume='approved').
```

## Run

```bash
cd 09_compliance_reviewer
python compliance_reviewer.py
```

## Demo input baked into the script
Drive file id of an MSA contract

## Expected output
Flagged clauses (e.g. 'Indemnification missing'), draft email, PAUSE for approval, send + reminder

## Files in this folder
- `compliance_reviewer.py` - the working LangGraph + Composio agent
- `README.md` - this file
- `graph.png` (auto-generated on first run) - the supervisor + workers diagram
- `compliance_reviewer.db` (auto-generated) - SQLite checkpoint store

## Architecture pattern

```
START -> supervisor -> [worker_1 | worker_2 | ...] -> supervisor -> ... -> END
```

Workers always return to the supervisor. The supervisor decides what's next based on state.
