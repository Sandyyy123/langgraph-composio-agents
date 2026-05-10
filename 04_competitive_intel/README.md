# Project 4 - Competitive Intelligence Monitor

## Summary
Weekly: reads competitor list from Notion, scans web per competitor, writes weekly Notion page, broadcasts to Slack.

## Composio apps to connect (one OAuth click each)
- [ ] Notion
- [ ] Tavily
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

## Extra setup
```
Set NOTION_COMPETITORS_DB_ID and NOTION_PARENT_PAGE_ID in .env (from your Notion DB URLs)
```

## Run

```bash
cd 04_competitive_intel
python competitive_intel.py
```

## Demo input baked into the script
Run weekly competitive intelligence sweep

## Expected output
Notion page 'Weekly CI - {date}' with 5 bullets per competitor, Slack post in #competitive-intel

## Files in this folder
- `competitive_intel.py` - the working LangGraph + Composio agent
- `README.md` - this file
- `graph.png` (auto-generated on first run) - the supervisor + workers diagram
- `competitive_intel.db` (auto-generated) - SQLite checkpoint store

## Architecture pattern

```
START -> supervisor -> [worker_1 | worker_2 | ...] -> supervisor -> ... -> END
```

Workers always return to the supervisor. The supervisor decides what's next based on state.
