"""
Project #10 - Internal IT Helpdesk Agent
Workers: classifier (LLM) -> runbook_finder (Notion) -> [auto_resolver | ticket_escalator] -> status_notifier
"""

import os, sqlite3, json
from pathlib import Path
from typing import TypedDict, Annotated, Optional, Literal
from dotenv import load_dotenv

load_dotenv(".env")
assert os.getenv("OPENAI_API_KEY")
assert os.getenv("COMPOSIO_API_KEY"), "Connect Notion + Slack + Linear on composio.dev"
print("env OK")

from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage

Category = Literal["access", "reset", "install", "other"]

class HelpdeskState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    next_worker: str
    user_id: str
    user_slack_id: Optional[str]
    request_text: str
    category: Optional[Category]
    runbook_match: Optional[str]
    self_service: Optional[bool]
    linear_ticket_id: Optional[str]
    auto_resolved: bool
    notified: bool

from langchain_openai import ChatOpenAI
from composio_langgraph import Action, ComposioToolSet
from langgraph.prebuilt import create_react_agent

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
toolset = ComposioToolSet()
notion_tools = toolset.get_tools(actions=[Action.NOTION_SEARCH_NOTION_PAGE])
slack_tools = toolset.get_tools(actions=[Action.SLACK_SENDS_A_MESSAGE_TO_A_SLACK_CHANNEL])
linear_tools = toolset.get_tools(actions=[Action.LINEAR_CREATE_LINEAR_ISSUE])


CLASSIFIER_PROMPT = """Classify IT helpdesk request. JSON ONLY: {"category":"access|reset|install|other"}
- access: VPN, login, MFA, SSO, permissions
- reset: forgotten password, locked account, expired credential
- install: software/license/peripheral request
- other: anything else (HR, payroll, slow PC)"""

def classifier_node(state: HelpdeskState) -> dict:
    out = llm.invoke([SystemMessage(content=CLASSIFIER_PROMPT),
                     HumanMessage(content=f"User: {state['user_id']}\nRequest: {state['request_text']}")])
    raw = out.content.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"): raw = raw[4:].strip()
    try: category = json.loads(raw).get("category", "other")
    except json.JSONDecodeError: category = "other"
    if category not in {"access","reset","install","other"}: category = "other"
    return {"category": category,
            "messages": [AIMessage(content=f"Classified as '{category}'", name="classifier")]}


RUNBOOK_PROMPT = """1. NOTION_SEARCH_NOTION_PAGE in 'Runbooks' workspace, filter by category keyword.
2. Pull top match.
3. JSON ONLY: {"runbook":"<text or 'NO_RUNBOOK'>", "self_service":bool, "assignee":"<email or null>"}
self_service=true means user can fix themselves."""

runbook_agent = create_react_agent(llm, notion_tools, prompt=RUNBOOK_PROMPT)

def runbook_finder_node(state: HelpdeskState) -> dict:
    query = f"Search Notion runbooks for category='{state['category']}'. Request: {state['request_text'][:400]}"
    result = runbook_agent.invoke({"messages": [HumanMessage(content=query)]})
    raw = (result["messages"][-1].content or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"): raw = raw[4:].strip()
    try: parsed = json.loads(raw)
    except json.JSONDecodeError: parsed = {"runbook":"NO_RUNBOOK","self_service":False,"assignee":None}
    runbook_text = parsed.get("runbook", "NO_RUNBOOK") or "NO_RUNBOOK"
    self_service = bool(parsed.get("self_service", False)) and runbook_text != "NO_RUNBOOK"
    assignee = parsed.get("assignee")
    return {"runbook_match": runbook_text, "self_service": self_service,
            "messages": [AIMessage(content=f"Runbook {'found' if runbook_text!='NO_RUNBOOK' else 'not found'}; self_service={self_service}; assignee={assignee}", name="runbook_finder")]}


AUTO_RESOLVER_PROMPT = """SLACK_SENDS_A_MESSAGE_TO_A_SLACK_CHANNEL to user's Slack DM with steps verbatim.
On success: 'DM_SENT'. On fail: 'DM_FAILED: <reason>'."""

auto_resolver_agent = create_react_agent(llm, slack_tools, prompt=AUTO_RESOLVER_PROMPT)

def auto_resolver_node(state: HelpdeskState) -> dict:
    if not state.get("self_service"):
        return {"auto_resolved": False,
                "messages": [AIMessage(content="auto_resolver invoked but self_service=False - skipping.", name="auto_resolver")]}
    runbook = state.get("runbook_match") or ""
    text = (f"Hi - your IT request was classified as '{state['category']}' and matches a self-service runbook. Steps:\n\n{runbook}\n\n"
           f"If these don't resolve in 15 min, reply and we'll open a ticket.")
    channel = state.get("user_slack_id") or state["user_id"]
    query = f"DM channel '{channel}'. Text:\n{text}\nUse SLACK_SENDS_A_MESSAGE_TO_A_SLACK_CHANNEL."
    result = auto_resolver_agent.invoke({"messages": [HumanMessage(content=query)]})
    confirm = (result["messages"][-1].content or "").strip()
    sent = "DM_SENT" in confirm or "sent" in confirm.lower()
    return {"auto_resolved": sent,
            "messages": [AIMessage(content=f"Auto-resolution DM {'sent' if sent else 'failed'} to {channel}", name="auto_resolver")]}


ESCALATOR_PROMPT = """LINEAR_CREATE_LINEAR_ISSUE with team_id (env LINEAR_IT_TEAM_ID), title, description, label, assignee_email.
On success: 'TICKET_CREATED: <id>'. On fail: 'TICKET_FAILED: <reason>'."""

escalator_agent = create_react_agent(llm, linear_tools, prompt=ESCALATOR_PROMPT)

def ticket_escalator_node(state: HelpdeskState) -> dict:
    team_id = os.getenv("LINEAR_IT_TEAM_ID", "IT")
    title = f"[{state['category'].upper()}] {state['request_text'][:80]}"
    description = (f"Reporter: {state['user_id']}\nCategory: {state['category']}\n\n"
                  f"Request:\n{state['request_text']}\n\n"
                  f"Runbook: {(state.get('runbook_match') or 'NO_RUNBOOK')[:1500]}")
    assignee = "it-oncall@example.com"
    for m in reversed(state["messages"]):
        if getattr(m, "name", None) == "runbook_finder" and "assignee=" in m.content:
            tail = m.content.split("assignee=", 1)[1].strip()
            if tail and tail.lower() not in {"none", "null"}: assignee = tail
            break
    query = (f"Create Linear issue:\nteam_id='{team_id}'\ntitle='{title}'\n"
            f"label='{state['category']}'\nassignee_email='{assignee}'\ndescription:\n{description}")
    result = escalator_agent.invoke({"messages": [HumanMessage(content=query)]})
    confirm = (result["messages"][-1].content or "").strip()
    ticket_id = None
    if "TICKET_CREATED" in confirm:
        try: ticket_id = confirm.split("TICKET_CREATED:", 1)[1].strip().split()[0]
        except Exception: ticket_id = "unknown"
    return {"linear_ticket_id": ticket_id,
            "messages": [AIMessage(content=f"Linear ticket {'opened '+str(ticket_id) if ticket_id else 'failed'}", name="ticket_escalator")]}


status_notifier_agent = create_react_agent(llm, slack_tools, prompt=AUTO_RESOLVER_PROMPT)

def status_notifier_node(state: HelpdeskState) -> dict:
    channel = state.get("user_slack_id") or state["user_id"]
    if state.get("auto_resolved"):
        text = f"Status: your '{state['category']}' request was handled via the self-service runbook above. Reply if still broken."
    elif state.get("linear_ticket_id"):
        text = f"Status: your '{state['category']}' request was escalated. Ticket: {state['linear_ticket_id']}. IT will reach out."
    else:
        text = f"Status: received your '{state['category']}' request but couldn't auto-resolve or escalate. On-call paged."
    query = f"DM channel '{channel}'. Text:\n{text}\nUse SLACK_SENDS_A_MESSAGE_TO_A_SLACK_CHANNEL."
    result = status_notifier_agent.invoke({"messages": [HumanMessage(content=query)]})
    confirm = (result["messages"][-1].content or "").strip()
    sent = "DM_SENT" in confirm or "sent" in confirm.lower()
    return {"notified": sent,
            "messages": [AIMessage(content=f"Status DM {'sent' if sent else 'failed'} to {channel}", name="status_notifier")]}


def supervisor(state: HelpdeskState) -> dict:
    if state.get("category") is None: return {"next_worker": "classifier"}
    if state.get("runbook_match") is None: return {"next_worker": "runbook_finder"}
    if state.get("self_service"):
        if not state.get("auto_resolved"): return {"next_worker": "auto_resolver"}
    else:
        if state.get("linear_ticket_id") is None: return {"next_worker": "ticket_escalator"}
    if not state.get("notified"): return {"next_worker": "status_notifier"}
    return {"next_worker": "DONE"}

def route(state: HelpdeskState) -> str:
    nxt = state["next_worker"]
    return nxt if nxt in {"classifier","runbook_finder","auto_resolver","ticket_escalator","status_notifier"} else "__end__"


from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

g = StateGraph(HelpdeskState)
g.add_node("supervisor", supervisor)
g.add_node("classifier", classifier_node)
g.add_node("runbook_finder", runbook_finder_node)
g.add_node("auto_resolver", auto_resolver_node)
g.add_node("ticket_escalator", ticket_escalator_node)
g.add_node("status_notifier", status_notifier_node)
g.add_edge(START, "supervisor")
g.add_conditional_edges("supervisor", route, {
    "classifier":"classifier","runbook_finder":"runbook_finder","auto_resolver":"auto_resolver",
    "ticket_escalator":"ticket_escalator","status_notifier":"status_notifier","__end__": END})
for w in ("classifier","runbook_finder","auto_resolver","ticket_escalator","status_notifier"):
    g.add_edge(w, "supervisor")

conn = sqlite3.connect("it_helpdesk.db", check_same_thread=False)
app = g.compile(checkpointer=SqliteSaver(conn))

try:
    Path("it_helpdesk_graph.png").write_bytes(app.get_graph().draw_mermaid_png())
    print("graph saved -> it_helpdesk_graph.png")
except Exception:
    print(app.get_graph().draw_ascii())

if __name__ == "__main__":
    # Case 1 - password reset (auto_resolver)
    cfg1 = {"configurable": {"thread_id": "ticket-itd-001"}, "recursion_limit": 30}
    init1 = {"user_id": "alice@example.com", "user_slack_id": "U07ALICE01",
            "request_text": "I forgot my SSO password and my account is locked out.",
            "auto_resolved": False, "notified": False,
            "messages": [HumanMessage(content="New IT request from alice")]}
    r1 = app.invoke(init1, config=cfg1)
    print("=== CASE 1 (self-service) ===")
    for k, v in r1.items():
        if k != "messages": print(f"{k}: {str(v)[:160]}")

    # Case 2 - VPN issue (escalate)
    cfg2 = {"configurable": {"thread_id": "ticket-itd-002"}, "recursion_limit": 30}
    init2 = {"user_id": "bob@example.com", "user_slack_id": "U07BOB0002",
            "request_text": ("VPN keeps dropping every 90 seconds when I try to reach the staging cluster "
                            "from the new office subnet 10.42.x.x. Already restarted client and reinstalled."),
            "auto_resolved": False, "notified": False,
            "messages": [HumanMessage(content="New IT request from bob")]}
    r2 = app.invoke(init2, config=cfg2)
    print("\n=== CASE 2 (escalated) ===")
    for k, v in r2.items():
        if k != "messages": print(f"{k}: {str(v)[:160]}")
    print("\n=== MESSAGE TRACE ===")
    for m in r2["messages"]:
        print(f"[{getattr(m,'name',m.type)}] {m.content[:200]}")
