"""
Project #9 - Compliance Document Reviewer
Workers: doc_reader (Drive) -> checklist_matcher (Notion) -> email_drafter (LLM) -> [HITL] -> email_sender (Gmail) -> reminder_creator (Calendar)
MANDATORY HITL: graph compiled with interrupt_before=['email_sender'].
"""

import os, sqlite3, json
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import TypedDict, Annotated, Optional
from dotenv import load_dotenv

load_dotenv(".env")
assert os.getenv("OPENAI_API_KEY")
assert os.getenv("COMPOSIO_API_KEY"), "Connect Drive + Notion + Gmail + Calendar on composio.dev"
print("env OK")

from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage

class ComplianceState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    next_worker: str
    drive_file_id: str
    doc_name: Optional[str]
    reviewer_email: str
    doc_text: Optional[str]
    flagged_clauses: Optional[list]
    draft_email: Optional[str]
    reminder_event_id: Optional[str]
    approved: bool
    email_sent: bool

from langchain_openai import ChatOpenAI
from composio_langgraph import Action, ComposioToolSet
from langgraph.prebuilt import create_react_agent

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
toolset = ComposioToolSet()
drive_tools = toolset.get_tools(actions=[Action.GOOGLEDRIVE_FIND_FILE])
notion_tools = toolset.get_tools(actions=[Action.NOTION_QUERY_DATABASE])
gmail_tools = toolset.get_tools(actions=[Action.GMAIL_SEND_EMAIL])
calendar_tools = toolset.get_tools(actions=[Action.GOOGLECALENDAR_CREATE_EVENT])


DOC_READER_PROMPT = """1. GOOGLEDRIVE_FIND_FILE with file_id.
2. If Google Doc, export as 'text/plain'. PDF/DOCX same way.
3. Return ONLY plain-text body. No commentary.
4. On fail: 'DRIVE_FETCH_FAILED: <reason>'."""

doc_reader_agent = create_react_agent(llm, drive_tools, prompt=DOC_READER_PROMPT)

def doc_reader_node(state: ComplianceState) -> dict:
    query = (f"Fetch plain-text body of Drive file id '{state['drive_file_id']}'. "
            f"GOOGLEDRIVE_FIND_FILE with id, export to 'text/plain'. Return text only.")
    result = doc_reader_agent.invoke({"messages": [HumanMessage(content=query)]})
    body = (result["messages"][-1].content or "").strip()
    doc_name = state.get("doc_name") or f"drive::{state['drive_file_id']}"
    return {"doc_text": body, "doc_name": doc_name,
            "messages": [AIMessage(content=f"Drive doc loaded ({len(body)} chars) - {doc_name}", name="doc_reader")]}


CHECKLIST_PROMPT = """1. NOTION_QUERY_DATABASE 'Compliance Checklist' (env NOTION_COMPLIANCE_DB_ID).
2. If unreachable, fallback to 13-item MSA checklist:
   Confidentiality, IP assignment, Indemnification, Limitation of liability, Warranty disclaimer,
   Termination for convenience, Termination for cause + cure, Governing law, Dispute resolution,
   Data protection (GDPR/CCPA), Force majeure, Assignment restrictions, Insurance.
3. For each: present | missing | risky.
4. JSON ONLY: {"flagged":[{"clause":"...","status":"missing|risky","severity":"low|medium|high","note":"..."}]}
   Only include missing/risky. Empty list if all good."""

checklist_agent = create_react_agent(llm, notion_tools, prompt=CHECKLIST_PROMPT)

def checklist_matcher_node(state: ComplianceState) -> dict:
    doc_text = state.get("doc_text") or ""
    trimmed = doc_text[:8000]
    db_hint = os.getenv("NOTION_COMPLIANCE_DB_ID", "<'Compliance Checklist' DB>")
    query = (f"Database hint: {db_hint}\nDoc: {state.get('doc_name')}\n"
            f"Text:\n```\n{trimmed}\n```\nReturn flagged clauses JSON.")
    result = checklist_agent.invoke({"messages": [HumanMessage(content=query)]})
    raw = (result["messages"][-1].content or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"): raw = raw[4:].strip()
    try: flagged = json.loads(raw).get("flagged", [])
    except json.JSONDecodeError: flagged = []
    return {"flagged_clauses": flagged,
            "messages": [AIMessage(content=f"Checklist: {len(flagged)} flagged.", name="checklist_matcher")]}


EMAIL_DRAFTER_PROMPT = """Draft compliance review email. Format:
Subject: <line>

Greeting (use reviewer first name from email local-part)
Paragraph: doc name + what was reviewed
Bullets: - <Clause> [<severity>] - <recommended action>
Closing: ask for sign-off (approve / request changes / escalate)
Sign off as 'Compliance Bot'. Factual, neutral, no marketing language."""

def email_drafter_node(state: ComplianceState) -> dict:
    flagged = state.get("flagged_clauses") or []
    if flagged:
        flagged_render = "\n".join(
            f"- {f.get('clause','?')} [{f.get('severity','medium')}] - {f.get('note','no note')}"
            for f in flagged)
    else:
        flagged_render = "(no flagged clauses)"
    user_block = (f"Reviewer: {state['reviewer_email']}\nDocument: {state.get('doc_name')}\n"
                 f"Flagged:\n{flagged_render}\n"
                 f"Action: 'request changes' if any high severity, else 'approve'.")
    out = llm.invoke([SystemMessage(content=EMAIL_DRAFTER_PROMPT), HumanMessage(content=user_block)])
    draft = out.content.strip()
    return {"draft_email": draft,
            "messages": [AIMessage(content=f"Draft ready ({len(draft)} chars) - awaiting approval.", name="email_drafter")]}


EMAIL_SENDER_PROMPT = """GMAIL_SEND_EMAIL with recipient, subject, body verbatim.
On success: 'EMAIL_SENT'. On fail: 'EMAIL_FAILED: <reason>'."""

email_sender_agent = create_react_agent(llm, gmail_tools, prompt=EMAIL_SENDER_PROMPT)

def _split_subject_body(draft: str) -> tuple[str, str]:
    lines = draft.splitlines()
    subject = "Compliance review"
    body_start = 0
    if lines and lines[0].lower().startswith("subject:"):
        subject = lines[0].split(":", 1)[1].strip() or subject
        body_start = 1
        if body_start < len(lines) and lines[body_start].strip() == "":
            body_start += 1
    body = "\n".join(lines[body_start:]).strip()
    return subject, body

def email_sender_node(state: ComplianceState) -> dict:
    if not state.get("approved"):
        return {"email_sent": False,
                "messages": [AIMessage(content="email_sender reached without approval - no send.", name="email_sender")]}
    subject, body = _split_subject_body(state.get("draft_email") or "")
    query = (f"Send email. recipient='{state['reviewer_email']}', subject='{subject}', "
            f"body:\n{body}\nGMAIL_SEND_EMAIL.")
    result = email_sender_agent.invoke({"messages": [HumanMessage(content=query)]})
    confirm = (result["messages"][-1].content or "").strip()
    sent = "EMAIL_SENT" in confirm or "sent" in confirm.lower()
    return {"email_sent": sent,
            "messages": [AIMessage(content=f"Email {'sent' if sent else 'failed'} to {state['reviewer_email']}", name="email_sender")]}


REMINDER_PROMPT = """GOOGLECALENDAR_CREATE_EVENT with summary, start_datetime, end_datetime (ISO 8601),
description, attendees. On success: 'EVENT_CREATED: <id>'. On fail: 'EVENT_FAILED: <reason>'."""

reminder_agent = create_react_agent(llm, calendar_tools, prompt=REMINDER_PROMPT)

def reminder_creator_node(state: ComplianceState) -> dict:
    start = (datetime.now(timezone.utc) + timedelta(days=7)).replace(microsecond=0)
    end = start + timedelta(minutes=30)
    summary = f"Follow up on {state.get('doc_name','contract')} compliance review"
    description = (f"Auto-scheduled. Reviewer: {state['reviewer_email']}.\n"
                  f"Flagged: {len(state.get('flagged_clauses') or [])}.\n"
                  f"Original: drive_file_id={state['drive_file_id']}.")
    query = (f"Create event:\nsummary='{summary}'\nstart='{start.isoformat()}'\n"
            f"end='{end.isoformat()}'\nattendees=['{state['reviewer_email']}']\n"
            f"description:\n{description}")
    result = reminder_agent.invoke({"messages": [HumanMessage(content=query)]})
    confirm = (result["messages"][-1].content or "").strip()
    event_id = None
    if "EVENT_CREATED" in confirm:
        try: event_id = confirm.split("EVENT_CREATED:", 1)[1].strip().split()[0]
        except Exception: event_id = "unknown"
    return {"reminder_event_id": event_id,
            "messages": [AIMessage(content=f"Reminder {'created '+str(event_id) if event_id else 'creation failed'}", name="reminder_creator")]}


def supervisor(state: ComplianceState) -> dict:
    if state.get("doc_text") is None: return {"next_worker": "doc_reader"}
    if state.get("flagged_clauses") is None: return {"next_worker": "checklist_matcher"}
    if state.get("draft_email") is None: return {"next_worker": "email_drafter"}
    if not state.get("email_sent"): return {"next_worker": "email_sender"}
    if state.get("reminder_event_id") is None: return {"next_worker": "reminder_creator"}
    return {"next_worker": "DONE"}

def route(state: ComplianceState) -> str:
    nxt = state["next_worker"]
    return nxt if nxt in {"doc_reader","checklist_matcher","email_drafter","email_sender","reminder_creator"} else "__end__"


from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

g = StateGraph(ComplianceState)
g.add_node("supervisor", supervisor)
g.add_node("doc_reader", doc_reader_node)
g.add_node("checklist_matcher", checklist_matcher_node)
g.add_node("email_drafter", email_drafter_node)
g.add_node("email_sender", email_sender_node)
g.add_node("reminder_creator", reminder_creator_node)
g.add_edge(START, "supervisor")
g.add_conditional_edges("supervisor", route, {
    "doc_reader":"doc_reader","checklist_matcher":"checklist_matcher",
    "email_drafter":"email_drafter","email_sender":"email_sender",
    "reminder_creator":"reminder_creator","__end__": END})
for w in ("doc_reader","checklist_matcher","email_drafter","email_sender","reminder_creator"):
    g.add_edge(w, "supervisor")

conn = sqlite3.connect("compliance_reviewer.db", check_same_thread=False)
app = g.compile(checkpointer=SqliteSaver(conn), interrupt_before=["email_sender"])

try:
    Path("compliance_graph.png").write_bytes(app.get_graph().draw_mermaid_png())
    print("graph saved -> compliance_graph.png")
except Exception:
    print(app.get_graph().draw_ascii())

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "compliance-msa-001"}, "recursion_limit": 40}
    initial = {"drive_file_id": "1A2B3C4D5E6F7G8H9I0J", "doc_name": "AcmeCorp_MSA_v3.docx",
              "reviewer_email": "legal-reviewer@example.com", "approved": False, "email_sent": False,
              "messages": [HumanMessage(content="Run compliance review on new MSA draft")]}

    pre = app.invoke(initial, config=config)
    print("=== PAUSED FOR HUMAN APPROVAL ===")
    print("Draft preview:")
    print((pre.get("draft_email") or "<none>")[:1000])
    print("\nFlagged clauses:")
    for f in pre.get("flagged_clauses", []) or []:
        print(f"  - {f}")

    approval = "approved"  # In production, from UI/Slack
    if approval == "approved":
        app.update_state(config, {"approved": True})
        post = app.invoke(Command(resume="approved"), config=config)
        print("\n=== AFTER RESUME ===")
        for k, v in post.items():
            if k != "messages": print(f"{k}: {str(v)[:160]}")
        print("\n=== MESSAGE TRACE ===")
        for m in post["messages"]:
            print(f"[{getattr(m,'name',m.type)}] {m.content[:200]}")
    else:
        print("Declined - workflow paused, no email sent.")
