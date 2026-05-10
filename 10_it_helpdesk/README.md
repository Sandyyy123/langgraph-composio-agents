# Project 10 - Internal IT Helpdesk Agent

## Summary
Classifies request (access/reset/install/other), finds Notion runbook, auto-resolves via Slack DM OR escalates to Linear, status DM.

## Composio apps to connect (one OAuth click each)
- [ ] Notion
- [ ] Slack
- [ ] Linear

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
Set LINEAR_IT_TEAM_ID in .env (defaults to 'IT')
```

## Run

```bash
cd 10_it_helpdesk
python it_helpdesk.py
```

## Demo input baked into the script
Two cases: (a) password lockout (auto-resolved); (b) VPN drops on subnet (escalated)

## Expected output
Classified category, runbook found, Slack DM with steps if self-service, else Linear ticket + status DM

## Files in this folder
- `it_helpdesk.py` - the working LangGraph + Composio agent
- `README.md` - this file
- `graph.png` (auto-generated on first run) - the supervisor + workers diagram
- `it_helpdesk.db` (auto-generated) - SQLite checkpoint store

## Architecture pattern

```
START -> supervisor -> [worker_1 | worker_2 | ...] -> supervisor -> ... -> END
```

Workers always return to the supervisor. The supervisor decides what's next based on state.
