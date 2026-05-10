# Project 6 - Insurance Claims Intake Agent

## Summary
Presidio PII redaction, extracts policy/date/value, looks up policy in Notion, custom @tool routes auto/human/reject, posts audit to Slack.

## Composio apps to connect (one OAuth click each)
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
pip install presidio-analyzer presidio-anonymizer && python -m spacy download en_core_web_lg
```

## Run

```bash
cd 06_insurance_claims
python insurance_claims.py
```

## Demo input baked into the script
Email: 'I am John Smith, policy P-90142, claim $3,250 for car repair on 2026-04-29'

## Expected output
Redacted body, extracted JSON, policy lookup, decision (auto_approve since <$5k + active), Slack #claims-ops post

## Files in this folder
- `insurance_claims.py` - the working LangGraph + Composio agent
- `README.md` - this file
- `graph.png` (auto-generated on first run) - the supervisor + workers diagram
- `insurance_claims.db` (auto-generated) - SQLite checkpoint store

## Architecture pattern

```
START -> supervisor -> [worker_1 | worker_2 | ...] -> supervisor -> ... -> END
```

Workers always return to the supervisor. The supervisor decides what's next based on state.
