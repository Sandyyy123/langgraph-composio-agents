"""
Project #7 - QA Bug Triage Agent
Workers: classifier (LLM) -> dedup_checker (Linear) -> screenshot_fetcher (Drive) -> ticket_creator (Linear) -> notifier (Slack)
Branching: if duplicate found, skip screenshot + ticket creation, jump to notifier.
"""

import os, sqlite3, json, re
from pathlib import Path
from typing import TypedDict, Annotated, Optional
from dotenv import load_dotenv

load_dotenv(".env")
assert os.getenv("OPENAI_API_KEY")
assert os.getenv("COMPOSIO_API_KEY"), "Connect Linear + Google Drive + Slack in Composio"

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage
from composio_langgraph import Action, ComposioToolSet


class BugState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    next_worker: str
    bug_report: str
    reporter: str
    summary: Optional[str]
    severity: Optional[str]
    severity_reason: Optional[str]
    duplicate_id: Optional[str]
    screenshot_link: Optional[str]
    linear_ticket_id: Optional[str]
    notified: bool


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
toolset = ComposioToolSet()
linear_search_tools = toolset.get_tools(actions=[Action.LINEAR_SEARCH_ISSUES])
linear_create_tools = toolset.get_tools(actions=[Action.LINEAR_CREATE_LINEAR_ISSUE])
drive_tools = toolset.get_tools(actions=[Action.GOOGLEDRIVE_FIND_FILE])
slack_tools = toolset.get_tools(actions=[Action.SLACK_SENDS_A_MESSAGE_TO_A_SLACK_CHANNEL])


CLASSIFIER_PROMPT = """Severity rubric:
P0: critical - prod down, data loss, security breach
P1: high - major feature broken, blocks many users, no workaround
P2: medium - feature partially broken, workaround exists
P3: low - cosmetic, copy issue, minor edge case
Return ONLY JSON: {"summary": "<<= 100 chars>", "severity": "P0|P1|P2|P3", "reason": "..."}"""

def classifier_node(state: BugState) -> dict:
    out = llm.invoke([SystemMessage(content=CLASSIFIER_PROMPT), HumanMessage(content=state["bug_report"])])
    raw = out.content.strip()
    if raw.startswith("```"): raw = raw.strip("`").lstrip("json").strip()
    try: payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = {"summary": state["bug_report"][:100], "severity": "P2", "reason": "fallback"}
    sev = payload.get("severity", "P2")
    if sev not in {"P0","P1","P2","P3"}: sev = "P2"
    return {"summary": (payload.get("summary") or "")[:200], "severity": sev,
            "severity_reason": payload.get("reason", ""),
            "messages": [AIMessage(content=f"Classified {sev}: {payload.get('summary','')[:100]}", name="classifier")]}


DEDUP_PROMPT = """LINEAR_SEARCH_ISSUES for issues whose title/description matches keywords.
Filter to open issues (Backlog, Todo, In Progress).
If found, take FIRST. Return ONLY JSON on final line:
{"is_duplicate": bool, "duplicate_id": "<ENG-1234>", "duplicate_title": "..."}"""

dedup_agent = create_react_agent(llm, linear_search_tools, prompt=DEDUP_PROMPT)

def dedup_checker_node(state: BugState) -> dict:
    user_msg = f"New bug: {state.get('summary','')}\nFull: {state['bug_report'][:1000]}"
    result = dedup_agent.invoke({"messages": [HumanMessage(content=user_msg)]})
    last = result["messages"][-1].content.strip()
    if last.startswith("```"): last = last.strip("`").lstrip("json").strip()
    payload = {}
    try: payload = json.loads(last)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", last, flags=re.DOTALL)
        if m:
            try: payload = json.loads(m.group(0))
            except json.JSONDecodeError: pass
    is_dup = bool(payload.get("is_duplicate", False))
    dup_id = (payload.get("duplicate_id") or "").strip() or None
    return {"duplicate_id": dup_id if is_dup else None,
            "messages": [AIMessage(content=f"Dedup: {'duplicate of '+(dup_id or '?') if is_dup else 'no duplicate'}", name="dedup_checker")]}


SCREENSHOT_PROMPT = """GOOGLEDRIVE_FIND_FILE in 'Bug Reports' folder.
Match files containing reporter email local-part, full email, or recent date.
Return FIRST match. Output JSON: {"found": bool, "link": "...", "name": "..."}"""

screenshot_agent = create_react_agent(llm, drive_tools, prompt=SCREENSHOT_PROMPT)

def screenshot_fetcher_node(state: BugState) -> dict:
    user_msg = f"reporter: {state['reporter']}\nsummary: {state.get('summary','')}\nFind first screenshot."
    result = screenshot_agent.invoke({"messages": [HumanMessage(content=user_msg)]})
    last = result["messages"][-1].content.strip()
    if last.startswith("```"): last = last.strip("`").lstrip("json").strip()
    payload = {}
    try: payload = json.loads(last)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", last, flags=re.DOTALL)
        if m:
            try: payload = json.loads(m.group(0))
            except json.JSONDecodeError: pass
    link = (payload.get("link") or "").strip() if payload.get("found") else ""
    return {"screenshot_link": link or "NO_SCREENSHOT",
            "messages": [AIMessage(content=f"Screenshot: {link or 'none found'}", name="screenshot_fetcher")]}


ROUTING_RULES = {"P0": "qa-leads", "P1": "qa-leads", "P2": "qa-leads", "P3": "qa-leads"}

TICKET_PROMPT = """LINEAR_CREATE_LINEAR_ISSUE with:
- title: summary
- description: full bug + 'Screenshot:' line (or 'Screenshot: not provided' if NO_SCREENSHOT)
- labels: [severity]
- assignee: provided
Return ONLY JSON: {"ticket_id": "ENG-XXXX", "url": "..."}"""

ticket_agent = create_react_agent(llm, linear_create_tools, prompt=TICKET_PROMPT)

def ticket_creator_node(state: BugState) -> dict:
    severity = state.get("severity", "P2")
    assignee = ROUTING_RULES.get(severity, "qa-leads")
    user_msg = (f"summary: {state.get('summary', '')}\nseverity: {severity}\nassignee: {assignee}\n"
               f"reporter: {state['reporter']}\n"
               f"screenshot_link: {state.get('screenshot_link', 'NO_SCREENSHOT')}\n\n"
               f"Full bug:\n{state['bug_report']}\n")
    result = ticket_agent.invoke({"messages": [HumanMessage(content=user_msg)]})
    last = result["messages"][-1].content.strip()
    if last.startswith("```"): last = last.strip("`").lstrip("json").strip()
    payload = {}
    try: payload = json.loads(last)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", last, flags=re.DOTALL)
        if m:
            try: payload = json.loads(m.group(0))
            except json.JSONDecodeError: pass
    ticket_id = (payload.get("ticket_id") or "").strip() or "UNKNOWN"
    return {"linear_ticket_id": ticket_id,
            "messages": [AIMessage(content=f"Linear created: {ticket_id} (assignee={assignee})", name="ticket_creator")]}


NOTIFIER_PROMPT = """DM the reporter via Slack.
If duplicate_id: 'Thanks! Looks like a duplicate of <ID>, already tracked.'
Else: 'Thanks! Filed <TICKET_ID> at <SEVERITY>.'
Confirm sent."""

notifier_agent = create_react_agent(llm, slack_tools, prompt=NOTIFIER_PROMPT)

def notifier_node(state: BugState) -> dict:
    user_msg = (f"reporter: {state['reporter']}\nduplicate_id: {state.get('duplicate_id') or ''}\n"
               f"linear_ticket_id: {state.get('linear_ticket_id') or ''}\n"
               f"severity: {state.get('severity', 'P2')}\nSend DM now.")
    result = notifier_agent.invoke({"messages": [HumanMessage(content=user_msg)]})
    return {"notified": True,
            "messages": [AIMessage(content=f"Reporter notified: {result['messages'][-1].content[:200]}", name="notifier")]}


def supervisor(state: BugState) -> dict:
    if state.get("severity") is None: return {"next_worker": "classifier"}
    if state.get("duplicate_id") is None and "dedup_checker" not in [getattr(m,"name","") for m in state["messages"]]:
        return {"next_worker": "dedup_checker"}
    if state.get("duplicate_id"):
        if not state.get("notified"): return {"next_worker": "notifier"}
        return {"next_worker": "DONE"}
    if state.get("screenshot_link") is None: return {"next_worker": "screenshot_fetcher"}
    if state.get("linear_ticket_id") is None: return {"next_worker": "ticket_creator"}
    if not state.get("notified"): return {"next_worker": "notifier"}
    return {"next_worker": "DONE"}

def route(state: BugState) -> str:
    nxt = state["next_worker"]
    return nxt if nxt in {"classifier","dedup_checker","screenshot_fetcher","ticket_creator","notifier"} else "__end__"


g = StateGraph(BugState)
g.add_node("supervisor", supervisor)
g.add_node("classifier", classifier_node)
g.add_node("dedup_checker", dedup_checker_node)
g.add_node("screenshot_fetcher", screenshot_fetcher_node)
g.add_node("ticket_creator", ticket_creator_node)
g.add_node("notifier", notifier_node)
g.add_edge(START, "supervisor")
g.add_conditional_edges("supervisor", route, {
    "classifier":"classifier","dedup_checker":"dedup_checker",
    "screenshot_fetcher":"screenshot_fetcher","ticket_creator":"ticket_creator",
    "notifier":"notifier","__end__": END})
for w in ("classifier","dedup_checker","screenshot_fetcher","ticket_creator","notifier"):
    g.add_edge(w, "supervisor")

conn = sqlite3.connect("qa_bug_triage.db", check_same_thread=False)
app = g.compile(checkpointer=SqliteSaver(conn))

if __name__ == "__main__":
    DEMO_BUG = """When I click "Export to PDF" on the Reports page, the entire app freezes
for 30 seconds and then shows a blank white screen. I have to refresh to recover.

Steps:
1. Log in
2. Go to Dashboard > Reports
3. Click "Export to PDF" on any report

Browser: Chrome 124 on macOS 14.4
Tried: hard refresh, different report, incognito - same result.
This blocks our weekly report delivery to clients."""

    config = {"configurable": {"thread_id": "bug-rpt-2031"}, "recursion_limit": 40}
    initial: BugState = {
        "messages": [HumanMessage(content="New bug submitted")],
        "next_worker": "", "bug_report": DEMO_BUG, "reporter": "qa.tester@example.com",
        "summary": None, "severity": None, "severity_reason": None,
        "duplicate_id": None, "screenshot_link": None, "linear_ticket_id": None, "notified": False}
    result = app.invoke(initial, config=config)
    print("=== FINAL STATE ===")
    for k, v in result.items():
        if k != "messages": print(f"{k}: {str(v)[:140]}")
    print("\n=== MESSAGE TRACE ===")
    for m in result["messages"]:
        print(f"[{getattr(m,'name',m.type)}] {str(m.content)[:200]}")
    try:
        Path("qa_bug_triage_graph.png").write_bytes(app.get_graph().draw_mermaid_png())
        print("\ngraph saved -> qa_bug_triage_graph.png")
    except Exception:
        print(app.get_graph().draw_ascii())
