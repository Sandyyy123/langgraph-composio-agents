# How to run all 10 LangGraph + Composio assignments

## One-time setup (15 min)

1. **Composio account** - sign up at https://composio.dev, copy API key.
2. **Add to .env**:
   ```
   OPENAI_API_KEY=sk-...
   COMPOSIO_API_KEY=cs-...
   ```
3. **Install Python deps** (one venv for all 10 projects):
   ```bash
   pip install langgraph langchain-openai composio composio_langgraph \
               python-dotenv langgraph-checkpoint-sqlite \
               presidio-analyzer presidio-anonymizer
   python -m spacy download en_core_web_lg
   ```
4. **Connect SaaS apps** in the Composio dashboard. One OAuth click each:
   - Gmail (projects 2, 3, 5, 9)
   - Google Calendar (projects 1, 3, 5, 9)
   - Google Drive (projects 7, 9)
   - Notion (projects 1-10 except 6 only Slack)
   - Slack (projects 1-10)
   - Linear (projects 7, 8, 10)
   - GitHub (project 8)
   - Tavily (projects 1, 3, 4)
   - HubSpot (project 1, optional)

## Per-project quickstart

| # | Project | Run command | Demo input baked in |
|---|---------|-------------|---------------------|
| 1 | Sales Lead Triage | `python 01_sales_lead_triage/sales_lead_triage.py` | vp.eng@acme-saas.com from Acme |
| 2 | Customer Support | `jupyter execute 02_support_resolver/support_resolver.ipynb` | Password reset ticket |
| 3 | Meeting Prep Briefer | `python 03_meeting_prep_briefer/meeting_prep_briefer.py` | Today's meetings |
| 4 | Competitive Intel | `python 04_competitive_intel/competitive_intel.py` | Weekly sweep |
| 5 | Recruiter Pipeline | `python 05_recruiter_pipeline/recruiter_pipeline.py` | Resume of Jane Q Developer |
| 6 | Insurance Claims | `python 06_insurance_claims/insurance_claims.py` | $3,250 auto claim |
| 7 | QA Bug Triage | `python 07_qa_bug_triage/qa_bug_triage.py` | Export-to-PDF bug |
| 8 | PR Review | `python 08_pr_review/pr_review.py` | PR #42 |
| 9 | Compliance Reviewer | `python 09_compliance_reviewer/compliance_reviewer.py` | MSA contract |
| 10 | IT Helpdesk | `python 10_it_helpdesk/it_helpdesk.py` | Password lockout |

## Per-project extra env vars

Some projects need IDs you must paste in once:

```
# Project 4 (Competitive Intel)
NOTION_COMPETITORS_DB_ID=... # from your Notion DB URL
NOTION_PARENT_PAGE_ID=...    # parent page where weekly reports go

# Project 8 (PR Review)
NOTION_PR_RULES_DB_ID=...
SLACK_PR_REVIEW_CHANNEL=#pr-reviews

# Project 9 (Compliance)
NOTION_COMPLIANCE_DB_ID=...

# Project 10 (IT Helpdesk)
LINEAR_IT_TEAM_ID=IT
```

## What you get on first run

Each project produces:
- Console trace of supervisor + worker calls
- Final state dump
- `graph.png` - the supervisor + workers diagram (Mermaid render)
- `<project>.db` - SQLite checkpoint store

## Troubleshooting

- **`COMPOSIO_API_KEY missing`** - sign up at composio.dev, add key to `.env`.
- **`Action.X does not exist`** - Composio renamed the action. Run `python -c "from composio_langgraph import Action; print([a for a in dir(Action) if 'GMAIL' in a])"` to find the new name.
- **OAuth fails** - go to Composio dashboard, click the app, re-authorise.
- **Project 6 import errors** - `pip install presidio-analyzer presidio-anonymizer && python -m spacy download en_core_web_lg`
- **Project 9 hangs at email_sender** - that's the HITL interrupt, by design. The script demonstrates the resume in the demo block.

## Submission checklist (Paras's rubric)

For each project:
- [ ] Supervisor + at least 3 worker nodes
- [ ] All workers use Composio toolkits (no hand-rolled API calls)
- [ ] README.md (in each project folder)
- [ ] `graph.png` saved (auto-generated)
- [ ] One end-to-end demo run output
- [ ] HITL on destructive actions (project 9 mandatory; others optional bonus)
- [ ] Checkpoint persistence (all 10 use SqliteSaver)

Push each project as its own GitHub repo or one mono-repo with 10 folders, then share links in the cohort channel.
