# Project 1 - Sales Lead Triage Agent

## Summary
Pulls leads, enriches via Tavily, scores against ICP, books discovery call for hot leads, notifies Slack.

## Composio apps to connect (one OAuth click each)
- [ ] Tavily
- [ ] Google Calendar
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
cd 01_sales_lead_triage
python sales_lead_triage.py
```

## Demo input baked into the script
Lead vp.eng@acme-saas.com from Acme SaaS Inc.

## Expected output
ICP score 0-100, tier (hot/warm/cold), calendar link if hot, Slack post in #sales-leads

## Files in this folder
- `sales_lead_triage.py` (or `.ipynb` for project 2) - the working LangGraph + Composio agent
- `README.md` - this file
- `graph.png` (auto-generated on first run) - the supervisor + workers diagram
- `sales_lead_triage.db` (auto-generated) - SQLite checkpoint store

## Architecture pattern

```
START -> supervisor -> [worker_1 | worker_2 | ...] -> supervisor -> ... -> END
```

Workers always return to the supervisor. The supervisor decides what's next based on state.
