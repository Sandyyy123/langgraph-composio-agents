"""
Project #5 - Recruiter Pipeline Assistant
Workers: parser (LLM) -> scorer (Notion) -> scheduler (Calendar, score>=70) -> db_updater (Notion)
"""

import os, sqlite3, json, re, datetime as dt
from pathlib import Path
from typing import TypedDict, Annotated, Optional
from dotenv import load_dotenv

load_dotenv(".env")
assert os.getenv("OPENAI_API_KEY")
assert os.getenv("COMPOSIO_API_KEY"), "Connect Notion + Google Calendar in Composio"

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage
from composio_langgraph import Action, ComposioToolSet


class RecruiterState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    next_worker: str
    resume_text: str
    candidate_email: str
    candidate_name: Optional[str]
    parsed: Optional[dict]
    score: Optional[int]
    score_rationale: Optional[str]
    calendar_event: Optional[str]
    pipeline_status: Optional[str]


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
toolset = ComposioToolSet()
notion_query_tools = toolset.get_tools(actions=[Action.NOTION_QUERY_DATABASE])
calendar_tools = toolset.get_tools(actions=[Action.GOOGLECALENDAR_CREATE_EVENT])
notion_write_tools = toolset.get_tools(actions=[Action.NOTION_CREATE_NOTION_PAGE, Action.NOTION_APPEND_TEXT_BLOCKS])


PARSER_PROMPT = """Extract structured fields from the resume. Return ONLY JSON:
{"candidate_name": "...", "skills": ["..."], "years_experience": int, "education": "..."}"""

def parser_node(state: RecruiterState) -> dict:
    out = llm.invoke([SystemMessage(content=PARSER_PROMPT), HumanMessage(content=state["resume_text"])])
    raw = out.content.strip()
    if raw.startswith("```"): raw = raw.strip("`").lstrip("json").strip()
    try: parsed = json.loads(raw)
    except json.JSONDecodeError: parsed = {"candidate_name": "", "skills": [], "years_experience": 0, "education": ""}
    return {"parsed": {"skills": parsed.get("skills", []),
                      "years_experience": int(parsed.get("years_experience") or 0),
                      "education": parsed.get("education", "")},
            "candidate_name": parsed.get("candidate_name", ""),
            "messages": [AIMessage(content=f"Parsed: {len(parsed.get('skills',[]))} skills, {parsed.get('years_experience',0)} yrs", name="parser")]}


SCORER_PROMPT = """You are a recruiter scoring engine.
Step 1: NOTION_QUERY_DATABASE on 'Open Roles', filter Status=Active, take first row.
Read Title, Required Skills, Min Years, Education.
Step 2: Score 0-100: 50pts skill overlap, 30pts years_exp >= min_years, 20pts education match.
Step 3: Return JSON ONLY: {"score": int, "rationale": "...", "jd_title": "..."}"""

scorer_agent = create_react_agent(llm, notion_query_tools, prompt=SCORER_PROMPT)

def scorer_node(state: RecruiterState) -> dict:
    parsed = state["parsed"] or {}
    user_msg = (f"Candidate parsed:\n  skills: {parsed.get('skills', [])}\n"
               f"  years_experience: {parsed.get('years_experience', 0)}\n"
               f"  education: {parsed.get('education', '')}\n\n"
               f"Pull active JD from 'Open Roles' Notion DB and score.")
    result = scorer_agent.invoke({"messages": [HumanMessage(content=user_msg)]})
    final = result["messages"][-1].content.strip()
    if final.startswith("```"): final = final.strip("`").lstrip("json").strip()
    try:
        payload = json.loads(final)
        score = int(payload.get("score", 0))
        rationale = payload.get("rationale", "")
    except json.JSONDecodeError:
        m = re.search(r"\b(\d{1,3})\b", final)
        score = int(m.group(1)) if m else 0
        rationale = final[:200]
    score = max(0, min(100, score))
    return {"score": score, "score_rationale": rationale,
            "messages": [AIMessage(content=f"Score: {score}/100 - {rationale[:120]}", name="scorer")]}


def _next_tuesday_1400_iso() -> tuple[str, str]:
    today = dt.date.today()
    days_ahead = (1 - today.weekday()) % 7
    if days_ahead == 0: days_ahead = 7
    target = today + dt.timedelta(days=days_ahead)
    start = dt.datetime.combine(target, dt.time(14, 0, 0))
    end = start + dt.timedelta(minutes=30)
    return start.isoformat(), end.isoformat()


SCHEDULER_PROMPT = """Use GOOGLECALENDAR_CREATE_EVENT for a 30-min screening call.
- Summary: 'Screening call - {candidate_name}'
- Description: 'Initial 30-min screen for {jd_role}. Score: {score}/100.'
- Start/End: ISO 8601 provided
- Time zone: Europe/Berlin
- Attendees: candidate_email
After creating, return ONLY the htmlLink URL on the final line."""

scheduler_agent = create_react_agent(llm, calendar_tools, prompt=SCHEDULER_PROMPT)

def scheduler_node(state: RecruiterState) -> dict:
    if (state.get("score") or 0) < 70:
        return {"calendar_event": "SKIPPED", "pipeline_status": "Reject",
                "messages": [AIMessage(content="Score < 70 - skipping calendar invite", name="scheduler")]}
    start_iso, end_iso = _next_tuesday_1400_iso()
    user_msg = (f"candidate_name: {state.get('candidate_name', 'Candidate')}\n"
               f"candidate_email: {state['candidate_email']}\n"
               f"start_datetime: {start_iso}\nend_datetime: {end_iso}\n"
               f"jd_role: scored role\nscore: {state['score']}\n"
               "Create the event now and return the htmlLink URL.")
    result = scheduler_agent.invoke({"messages": [HumanMessage(content=user_msg)]})
    last = result["messages"][-1].content.strip()
    m = re.search(r"https?://\S+", last)
    link = m.group(0) if m else last[:300]
    return {"calendar_event": link, "pipeline_status": "Screening",
            "messages": [AIMessage(content=f"Calendar event: {link[:120]}", name="scheduler")]}


DB_UPDATER_PROMPT = """Step 1: NOTION_CREATE_NOTION_PAGE in 'Pipeline' DB with properties:
Name, Email, Score, Status, Calendar.
Step 2: NOTION_APPEND_TEXT_BLOCKS to append rationale as paragraph.
Return one-line confirmation with new page id."""

db_updater_agent = create_react_agent(llm, notion_write_tools, prompt=DB_UPDATER_PROMPT)

def db_updater_node(state: RecruiterState) -> dict:
    user_msg = (f"candidate_name: {state.get('candidate_name', 'Unknown')}\n"
               f"candidate_email: {state['candidate_email']}\n"
               f"score: {state.get('score', 0)}\nstatus: {state.get('pipeline_status', 'Reject')}\n"
               f"calendar_event: {state.get('calendar_event', 'SKIPPED')}\n"
               f"rationale: {state.get('score_rationale', '')}\n"
               "Create the Pipeline page and append the rationale block.")
    result = db_updater_agent.invoke({"messages": [HumanMessage(content=user_msg)]})
    return {"messages": [AIMessage(content=f"Pipeline updated: {result['messages'][-1].content[:200]}", name="db_updater")]}


def supervisor(state: RecruiterState) -> dict:
    if state.get("parsed") is None: return {"next_worker": "parser"}
    if state.get("score") is None: return {"next_worker": "scorer"}
    if state.get("calendar_event") is None: return {"next_worker": "scheduler"}
    if state.get("pipeline_status") and not any(getattr(m,"name","")=="db_updater" for m in state["messages"]):
        return {"next_worker": "db_updater"}
    return {"next_worker": "DONE"}

def route(state: RecruiterState) -> str:
    nxt = state["next_worker"]
    return nxt if nxt in {"parser","scorer","scheduler","db_updater"} else "__end__"


g = StateGraph(RecruiterState)
g.add_node("supervisor", supervisor)
g.add_node("parser", parser_node)
g.add_node("scorer", scorer_node)
g.add_node("scheduler", scheduler_node)
g.add_node("db_updater", db_updater_node)
g.add_edge(START, "supervisor")
g.add_conditional_edges("supervisor", route, {"parser":"parser","scorer":"scorer",
    "scheduler":"scheduler","db_updater":"db_updater","__end__": END})
for w in ("parser","scorer","scheduler","db_updater"): g.add_edge(w, "supervisor")

conn = sqlite3.connect("recruiter_pipeline.db", check_same_thread=False)
app = g.compile(checkpointer=SqliteSaver(conn))

if __name__ == "__main__":
    DEMO_RESUME = """Jane Q Developer
jane.dev@example.com | +1-415-555-0117

EXPERIENCE
Senior Backend Engineer, Acme Corp (2019 - 2025)
- Designed Python microservices on AWS (FastAPI, Postgres, Redis, Kafka)
- Led migration to Kubernetes, cut p95 latency by 40%

Backend Engineer, Startup Inc (2017 - 2019)
- Django + Celery payment pipeline; PCI-DSS compliance

EDUCATION
B.S. Computer Science, UC Berkeley, 2017

SKILLS
Python, FastAPI, Django, Postgres, Redis, Kafka, AWS, Kubernetes, Docker, CI/CD"""

    config = {"configurable": {"thread_id": "candidate-jane-001"}, "recursion_limit": 40}
    initial: RecruiterState = {
        "messages": [HumanMessage(content="New candidate: Jane Q Developer")],
        "next_worker": "", "resume_text": DEMO_RESUME, "candidate_email": "jane.dev@example.com",
        "candidate_name": None, "parsed": None, "score": None, "score_rationale": None,
        "calendar_event": None, "pipeline_status": None}
    result = app.invoke(initial, config=config)
    print("=== FINAL STATE ===")
    for k, v in result.items():
        if k != "messages": print(f"{k}: {str(v)[:140]}")
    print("\n=== MESSAGE TRACE ===")
    for m in result["messages"]:
        print(f"[{getattr(m,'name',m.type)}] {str(m.content)[:180]}")
    try:
        Path("recruiter_pipeline_graph.png").write_bytes(app.get_graph().draw_mermaid_png())
        print("\ngraph saved -> recruiter_pipeline_graph.png")
    except Exception:
        print(app.get_graph().draw_ascii())
