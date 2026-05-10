# Project 8 - PR Review Assistant

## Summary
Fetches PR diff, summarises in 3 bullets, checks Notion review rules, posts structured review comment, pings reviewer in Slack.

## Composio apps to connect (one OAuth click each)
- [ ] GitHub
- [ ] Notion
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
Set NOTION_PR_RULES_DB_ID and SLACK_PR_REVIEW_CHANNEL in .env
```

## Run

```bash
cd 08_pr_review
python pr_review.py
```

## Demo input baked into the script
PR #42 in repo Sandyyy123/groverjobapps

## Expected output
3-bullet summary, violations list (tests/docs/secrets), GitHub PR comment, Slack ping in #pr-reviews

## Files in this folder
- `pr_review.py` - the working LangGraph + Composio agent
- `README.md` - this file
- `graph.png` (auto-generated on first run) - the supervisor + workers diagram
- `pr_review.db` (auto-generated) - SQLite checkpoint store

## Architecture pattern

```
START -> supervisor -> [worker_1 | worker_2 | ...] -> supervisor -> ... -> END
```

Workers always return to the supervisor. The supervisor decides what's next based on state.
