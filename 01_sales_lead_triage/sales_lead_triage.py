"""
Project #1 - Sales Lead Triage
Workers: enricher (Tavily) -> scorer (LLM only) -> booker (Calendar, hot only) -> notifier (Slack)
"""

import os, sqlite3, json
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(".env")

assert os.getenv("OPENAI_API_KEY")
assert os.getenv("COMPOSIO_API_KEY"), "Sign up at composio.dev and connect Tavily + Google Calendar + Slack"
print("env OK")

from typing import TypedDict, Annotated, Optional
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage

class LeadState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    next_worker: str
    lead_email: str
    lead_company: str
    enrichment_data: Optional[dict]
    icp_score: Optional[int]
    tier: Optional[str]
    calendar_link: Optional[str]

from langchain_openai import ChatOpenAI
from composio_langgraph import Action, ComposioToolSet
from langgraph.prebuilt import create_react_agent

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
toolset = ComposioToolSet()

tavily_tools = toolset.get_tools(actions=[Action.TAVILY_TAVILY_SEARCH, Action.TAVILY_TAVILY_EXTRACT])
calendar_tools = toolset.get_tools(actions=[Action.GOOGLECALENDAR_CREATE_EVENT])
slack_tools = toolset.get_tools(actions=[Action.SLACK_SENDS_A_MESSAGE_TO_A_SLACK_CHANNEL])

ENRICHER_PROMPT = """You are a sales lead enrichment specialist. Use Tavily search and extract
to gather facts about a company. Look up company size, industry, recent news (past 90 days).
Output ONLY a JSON object on the final line:
{"company_size": "...", "industry": "...", "recent_news": "..."}"""

enricher_agent = create_react_agent(llm, tavily_tools, prompt=ENRICHER_PROMPT)

def enricher_node(state: LeadState) -> dict:
    query = f"Research the company '{state['lead_company']}' (lead email: {state['lead_email']}). Return company_size, industry, recent_news as JSON."
    result = enricher_agent.invoke({"messages": [HumanMessage(content=query)]})
    final = result["messages"][-1].content
    enrichment = {"company_size": "unknown", "industry": "unknown", "recent_news": "none"}
    for line in reversed(final.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                enrichment = json.loads(line); break
            except json.JSONDecodeError:
                continue
    return {"enrichment_data": enrichment,
            "messages": [AIMessage(content=f"Enriched {state['lead_company']}: {json.dumps(enrichment)[:200]}", name="enricher")]}

SCORER_PROMPT = """You are an ICP scorer for a B2B SaaS sales team.
ICP: B2B SaaS, 50-500 employees, growth stage.
Scoring: 80-100 strong match, 50-79 partial, 0-49 mismatch.
Tier: hot if >= 80, warm if 50-79, cold if < 50.
Output JSON only: {"icp_score": int, "tier": "hot|warm|cold", "rationale": "..."}"""

def scorer_node(state: LeadState) -> dict:
    enrichment = state.get("enrichment_data") or {}
    out = llm.invoke([SystemMessage(content=SCORER_PROMPT),
                     HumanMessage(content=f"Lead company: {state['lead_company']}\nEnrichment: {json.dumps(enrichment)}")])
    raw = out.content.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"): raw = raw[4:]
        raw = raw.strip()
    try:
        parsed = json.loads(raw)
        score = int(parsed.get("icp_score", 0))
        tier = parsed.get("tier")
    except Exception:
        score, tier = 0, "cold"
    if tier not in ("hot", "warm", "cold"):
        tier = "hot" if score >= 80 else "warm" if score >= 50 else "cold"
    return {"icp_score": score, "tier": tier,
            "messages": [AIMessage(content=f"Scored {state['lead_company']}: {score}/100 -> {tier}", name="scorer")]}

BOOKER_PROMPT = """You are a meeting booker. Create a 30-min discovery call event for tomorrow 10:00.
Title: 'Discovery call - {company}'. Include lead's email as attendee. Return ONLY the htmlLink URL."""

booker_agent = create_react_agent(llm, calendar_tools, prompt=BOOKER_PROMPT)

def booker_node(state: LeadState) -> dict:
    if state.get("tier") != "hot":
        return {"messages": [AIMessage(content="Tier not hot; skipping booking.", name="booker")]}
    from datetime import datetime, timedelta, timezone
    tomorrow = datetime.now(timezone.utc) + timedelta(days=1)
    start = tomorrow.replace(hour=10, minute=0, second=0, microsecond=0)
    end = start + timedelta(minutes=30)
    instruction = (f"Create event 'Discovery call - {state['lead_company']}' "
                   f"start {start.isoformat()} end {end.isoformat()} "
                   f"attendee {state['lead_email']}. Return htmlLink.")
    result = booker_agent.invoke({"messages": [HumanMessage(content=instruction)]})
    final = result["messages"][-1].content
    link = "https://calendar.google.com/"
    for tok in final.split():
        tok = tok.strip(" .,;()[]<>\"'")
        if tok.startswith("http") and "calendar" in tok.lower():
            link = tok; break
    return {"calendar_link": link,
            "messages": [AIMessage(content=f"Booked: {link}", name="booker")]}

NOTIFIER_PROMPT = """Post a single message to #sales-leads with lead summary. Under 5 lines. Confirm 'POSTED'."""
notifier_agent = create_react_agent(llm, slack_tools, prompt=NOTIFIER_PROMPT)

def notifier_node(state: LeadState) -> dict:
    summary = (f"New lead triaged.\nCompany: {state['lead_company']}\nEmail: {state['lead_email']}\n"
               f"Tier: {state.get('tier','cold').upper()} (score {state.get('icp_score',0)}/100)\n"
               f"Calendar: {state.get('calendar_link') or 'not booked'}")
    result = notifier_agent.invoke({"messages": [HumanMessage(content=f"Post to #sales-leads:\n\n{summary}")]})
    return {"messages": [AIMessage(content=f"Notified Slack: {result['messages'][-1].content[:150]}", name="notifier")]}

def supervisor(state: LeadState) -> dict:
    if state.get("enrichment_data") is None: return {"next_worker": "enricher"}
    if state.get("icp_score") is None: return {"next_worker": "scorer"}
    if state.get("tier") == "hot" and state.get("calendar_link") is None: return {"next_worker": "booker"}
    last = state["messages"][-1].content if state["messages"] else ""
    if "Notified Slack" not in last: return {"next_worker": "notifier"}
    return {"next_worker": "DONE"}

def route(state: LeadState) -> str:
    nxt = state["next_worker"]
    return nxt if nxt in {"enricher", "scorer", "booker", "notifier"} else "__end__"

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

g = StateGraph(LeadState)
g.add_node("supervisor", supervisor)
for n, fn in [("enricher", enricher_node), ("scorer", scorer_node), ("booker", booker_node), ("notifier", notifier_node)]:
    g.add_node(n, fn); g.add_edge(n, "supervisor")
g.add_edge(START, "supervisor")
g.add_conditional_edges("supervisor", route, {"enricher":"enricher","scorer":"scorer","booker":"booker","notifier":"notifier","__end__":END})

conn = sqlite3.connect("sales_lead_triage.db", check_same_thread=False)
app = g.compile(checkpointer=SqliteSaver(conn))

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "lead-2001"}, "recursion_limit": 30}
    initial = {"lead_email": "vp.eng@acme-saas.com", "lead_company": "Acme SaaS Inc.",
               "messages": [HumanMessage(content="New inbound lead from Acme SaaS Inc.")]}
    result = app.invoke(initial, config=config)
    print("=== FINAL STATE ===")
    for k, v in result.items():
        if k != "messages": print(f"{k}: {str(v)[:160]}")
    print("\n=== MESSAGE TRACE ===")
    for m in result["messages"]:
        print(f"[{getattr(m,'name',m.type)}] {m.content[:200]}")
    try:
        Path("graph_lead.png").write_bytes(app.get_graph().draw_mermaid_png())
        print("\ngraph saved -> graph_lead.png")
    except Exception:
        print(app.get_graph().draw_ascii())