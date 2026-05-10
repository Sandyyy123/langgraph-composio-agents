# LangGraph + Composio AI Agents — 10 Production Workflows

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python) ![LangGraph](https://img.shields.io/badge/LangGraph-0.2-purple) ![Composio](https://img.shields.io/badge/Composio-SaaS%20integrations-green) ![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-green?logo=openai) ![License](https://img.shields.io/badge/license-CC%20BY--NC%204.0-lightgrey)

Ten production-grade agentic workflows built with **LangGraph** (supervisor + worker pattern) and **Composio** (one-click SaaS OAuth). Each agent solves a real business automation problem end-to-end.

---

## Architecture Pattern

Every agent uses the same LangGraph supervisor loop:

```
START
  └─► Supervisor (GPT-4o)
         ├─► Worker 1 (tool call via Composio)
         ├─► Worker 2 (tool call via Composio)
         └─► Worker N  ...
                └─► back to Supervisor
                        └─► END (when task complete)
```

State is persisted in SQLite via `langgraph-checkpoint-sqlite`. Workers never call each other directly — all routing goes through the supervisor.

---

## The 10 Agents

| # | Agent | Composio Apps | Use Case |
|---|-------|--------------|----------|
| 01 | Sales Lead Triage | Tavily, Calendar, Slack | Enrich leads, score ICP, book hot leads |
| 02 | Support Resolver | Gmail, Notion, Slack | Classify tickets, draft replies, escalate |
| 03 | Meeting Prep Briefer | Gmail, Calendar, Tavily, Notion | Pre-meeting dossier on attendees |
| 04 | Competitive Intel | Tavily, Notion, Slack | Summarise competitor moves on demand |
| 05 | Recruiter Pipeline | Gmail, Calendar, Notion, Slack | Screen candidates, schedule interviews |
| 06 | Insurance Claims | Drive, Slack | Validate claims, flag fraud signals |
| 07 | QA Bug Triage | Linear, GitHub, Slack | Route bugs by severity, assign sprint |
| 08 | PR Review | GitHub, Linear, Slack | Review PRs, post inline comments |
| 09 | Compliance Reviewer | Gmail, Drive, Notion | Flag policy violations in documents |
| 10 | IT Helpdesk | Gmail, Slack, Notion | Auto-resolve L1 tickets, escalate L2 |

---

## Quick Start

### 1. Install dependencies (one venv for all 10 agents)

```bash
pip install langgraph langchain-openai composio composio_langgraph \
            python-dotenv langgraph-checkpoint-sqlite \
            presidio-analyzer presidio-anonymizer
python -m spacy download en_core_web_lg
```

### 2. Add keys to `.env`

```
OPENAI_API_KEY=sk-...
COMPOSIO_API_KEY=cs-...
```

### 3. Connect SaaS apps in Composio dashboard

Sign up at https://composio.dev, then connect the apps your target agent needs (one OAuth click each):

- Gmail, Google Calendar, Google Drive
- Notion, Slack, Linear, GitHub
- Tavily (web search)
- HubSpot (optional, agent 01)

### 4. Run an agent

```bash
cd 01_sales_lead_triage
python sales_lead_triage.py
# or open the .ipynb in Jupyter
```

Each folder has its own `README.md` with exact setup steps and expected output.

---

## Project Structure

```
langgraph-composio-agents/
├── 00_template/               # Blank supervisor scaffold to fork
├── 01_sales_lead_triage/
│   ├── sales_lead_triage.py   # Runnable agent
│   ├── sales_lead_triage.ipynb
│   └── README.md
├── 02_support_resolver/ ...
...
├── 10_it_helpdesk/
└── RUN.md                     # End-to-end setup guide
```

---

## Tech Stack

| Component | Library |
|-----------|---------|
| Agent orchestration | LangGraph 0.2 |
| SaaS integrations | Composio |
| LLM backbone | OpenAI GPT-4o |
| State persistence | langgraph-checkpoint-sqlite |
| PII handling | Microsoft Presidio |
| Web search | Tavily API |

---

## Author

**Dr. Sandeep Grover** — PhD Data Science, independent ML researcher, Mössingen, Germany.

---

## License

MIT
