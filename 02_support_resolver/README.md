# Project 2 - Customer Support Ticket Resolver

## Summary
Watches tickets, classifies intent, searches Notion KB, drafts reply, auto-sends low-risk or escalates via HITL.

## Composio apps to connect (one OAuth click each)
- [ ] Notion
- [ ] Gmail
- [ ] Slack

## Setup steps

1. Sign up at https://composio.dev and copy your API key
2. Add to `.env`:
   ```
   COMPOSIO_API_KEY=cs_...
   OPENAI_API_KEY=sk-...
   ```
3. In the Composio dashboard, click each app above and complete OAuth.
4. `pip install langgraph composio composio_langgraph langchain-openai python-dotenv langgraph-checkpoint-sqlite`

## Run

```bash
cd 02_support_resolver
jupyter execute support_resolver.ipynb
```

## Demo input baked into the script
Ticket: 'How do I reset my password?'

## Expected output
Classification, KB lookup, drafted reply, Gmail send (or Slack escalation if high-risk)

## Files in this folder
- `support_resolver.py` (or `.ipynb` for project 2) - the working LangGraph + Composio agent
- `README.md` - this file
- `graph.png` (auto-generated on first run) - the supervisor + workers diagram
- `support_resolver.db` (auto-generated) - SQLite checkpoint store

## Architecture pattern

```
START -> supervisor -> [worker_1 | worker_2 | ...] -> supervisor -> ... -> END
```

Workers always return to the supervisor. The supervisor decides what's next based on state.
