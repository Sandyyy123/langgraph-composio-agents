# Project 3 - Meeting Prep Briefer

## Summary
Cron 7am: reads Calendar, researches attendees via Tavily+Notion, pulls Gmail history, DMs brief via Slack.

## Composio apps to connect (one OAuth click each)
- [ ] Google Calendar
- [ ] Tavily
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
3. In Composio dashboard, click each app above and complete OAuth.
4. `pip install langgraph composio composio_langgraph langchain-openai python-dotenv langgraph-checkpoint-sqlite`

## Extra setup
```
Optional: SLACK_DM_TARGET=@yourname (default '@me')
```

## Run

```bash
cd 03_meeting_prep_briefer
python meeting_prep_briefer.py
```

## Demo input baked into the script
Run morning brief for today's meetings

## Expected output
Per-meeting brief (attendees, background, history, talking points), sent to Slack DM

## Files in this folder
- `meeting_prep_briefer.py` - the working LangGraph + Composio agent
- `README.md` - this file
- `graph.png` (auto-generated on first run) - the supervisor + workers diagram
- `meeting_prep_briefer.db` (auto-generated) - SQLite checkpoint store

## Architecture pattern

```
START -> supervisor -> [worker_1 | worker_2 | ...] -> supervisor -> ... -> END
```

Workers always return to the supervisor. The supervisor decides what's next based on state.
