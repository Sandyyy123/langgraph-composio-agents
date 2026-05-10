"""
Project #6 - Insurance Claims Intake Agent
Workers: redactor (Presidio) -> extractor (LLM) -> policy_lookup (Notion) -> router (custom @tool) -> notifier (Slack)
"""

import os, sqlite3, json, re
from pathlib import Path
from typing import TypedDict, Annotated, Optional
from dotenv import load_dotenv

load_dotenv(".env")
assert os.getenv("OPENAI_API_KEY")
assert os.getenv("COMPOSIO_API_KEY"), "Connect Notion + Slack in Composio"

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from composio_langgraph import Action, ComposioToolSet

# pip install presidio-analyzer presidio-anonymizer
# python -m spacy download en_core_web_lg
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine


class ClaimState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    next_worker: str
    raw_email_id: str
    raw_email_body: str
    raw_email_body_redacted: Optional[str]
    policy_number: Optional[str]
    incident_date: Optional[str]
    claim_value: Optional[float]
    claim_type: Optional[str]
    policy_state: Optional[str]
    policyholder_found: bool
    decision: Optional[str]
    decision_reason: Optional[str]


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
toolset = ComposioToolSet()
notion_tools = toolset.get_tools(actions=[Action.NOTION_QUERY_DATABASE])
slack_tools = toolset.get_tools(actions=[Action.SLACK_SENDS_A_MESSAGE_TO_A_SLACK_CHANNEL])

_analyzer = AnalyzerEngine()
_anonymizer = AnonymizerEngine()
_PII_ENTITIES = ["PERSON", "US_SSN", "PHONE_NUMBER", "EMAIL_ADDRESS", "LOCATION"]


def redactor_node(state: ClaimState) -> dict:
    body = state["raw_email_body"] or ""
    findings = _analyzer.analyze(text=body, entities=_PII_ENTITIES, language="en")
    redacted = _anonymizer.anonymize(text=body, analyzer_results=findings).text
    return {"raw_email_body_redacted": redacted,
            "messages": [AIMessage(content=f"Redacted {len(findings)} PII spans ({sorted({f.entity_type for f in findings})})", name="redactor")]}


EXTRACTOR_PROMPT = """Read redacted email and extract:
- policy_number (alphanumeric, e.g. P-90142)
- incident_date (YYYY-MM-DD)
- claim_value (float, no currency symbols)
- claim_type ("auto"|"home"|"health"|"life"|"other")
Return ONLY JSON, no fences."""

def extractor_node(state: ClaimState) -> dict:
    out = llm.invoke([SystemMessage(content=EXTRACTOR_PROMPT),
                     HumanMessage(content=state.get("raw_email_body_redacted") or state["raw_email_body"])])
    raw = out.content.strip()
    if raw.startswith("```"): raw = raw.strip("`").lstrip("json").strip()
    try: payload = json.loads(raw)
    except json.JSONDecodeError: payload = {"policy_number": "", "incident_date": "", "claim_value": 0.0, "claim_type": "other"}
    try: claim_value = float(payload.get("claim_value") or 0.0)
    except (TypeError, ValueError): claim_value = 0.0
    return {"policy_number": (payload.get("policy_number") or "").strip(),
            "incident_date": (payload.get("incident_date") or "").strip(),
            "claim_value": claim_value,
            "claim_type": (payload.get("claim_type") or "other").strip().lower(),
            "messages": [AIMessage(content=f"Extracted: policy={payload.get('policy_number','')} date={payload.get('incident_date','')} value=${claim_value:.2f} type={payload.get('claim_type','')}", name="extractor")]}


POLICY_LOOKUP_PROMPT = """NOTION_QUERY_DATABASE on 'Policies' DB filtered by PolicyNumber.
Read Status (active|lapsed|cancelled) and Holder.
Return ONLY JSON on final line: {"found": bool, "status": "active|lapsed|cancelled|unknown", "holder": "..."}"""

policy_agent = create_react_agent(llm, notion_tools, prompt=POLICY_LOOKUP_PROMPT)

def policy_lookup_node(state: ClaimState) -> dict:
    pn = state.get("policy_number") or ""
    if not pn:
        return {"policyholder_found": False, "policy_state": "unknown",
                "messages": [AIMessage(content="No policy number extracted", name="policy_lookup")]}
    user_msg = f"Look up policy_number = '{pn}' in the Policies DB."
    result = policy_agent.invoke({"messages": [HumanMessage(content=user_msg)]})
    last = result["messages"][-1].content.strip()
    payload = {}
    json_match = re.search(r"\{[^{}]*\}", last[::-1])
    if json_match:
        try:
            candidate = last[len(last) - json_match.end():].strip()
            payload = json.loads(candidate)
        except (json.JSONDecodeError, ValueError): payload = {}
    if not payload:
        try: payload = json.loads(last)
        except json.JSONDecodeError: payload = {"found": False, "status": "unknown", "holder": ""}
    found = bool(payload.get("found", False))
    status = (payload.get("status") or "unknown").lower()
    return {"policyholder_found": found, "policy_state": status,
            "messages": [AIMessage(content=f"Policy {pn}: found={found}, status={status}", name="policy_lookup")]}


@tool
def evaluate_claim_threshold(claim_value: float, policy_state: str, claim_type: str) -> dict:
    """Apply business rules to decide claim routing."""
    state = (policy_state or "").lower()
    ctype = (claim_type or "").lower()
    if state != "active":
        return {"decision": "reject", "reason": f"Policy state is '{state}', not active."}
    if ctype in {"health", "life"}:
        return {"decision": "human_adjuster", "reason": f"{ctype} claims always require human review."}
    if claim_value < 5000:
        return {"decision": "auto_approve", "reason": f"Active policy and ${claim_value:.2f} below $5,000 threshold."}
    return {"decision": "human_adjuster", "reason": f"Active policy but ${claim_value:.2f} >= $5,000 threshold."}


def router_node(state: ClaimState) -> dict:
    if not state.get("policyholder_found"):
        decision = "reject"
        reason = f"Policy '{state.get('policy_number','')}' not found in Policies DB."
    else:
        out = evaluate_claim_threshold.invoke({
            "claim_value": float(state.get("claim_value") or 0.0),
            "policy_state": state.get("policy_state") or "unknown",
            "claim_type": state.get("claim_type") or "other"})
        decision = out["decision"]
        reason = out["reason"]
    return {"decision": decision, "decision_reason": reason,
            "messages": [AIMessage(content=f"Decision: {decision} - {reason}", name="router")]}


NOTIFIER_PROMPT = """Post to #claims-ops with audit trail. Format:
*Claim intake decision*
- Email ID, Policy, Holder found, Policy state, Claim type, Claim value, Incident date
- *Decision*: <UPPERCASE>
- Reason: <reason>
After posting, confirm channel and timestamp."""

notifier_agent = create_react_agent(llm, slack_tools, prompt=NOTIFIER_PROMPT)

def notifier_node(state: ClaimState) -> dict:
    user_msg = (f"raw_email_id: {state['raw_email_id']}\n"
               f"policy_number: {state.get('policy_number', '')}\n"
               f"holder_found: {'yes' if state.get('policyholder_found') else 'no'}\n"
               f"policy_state: {state.get('policy_state', 'unknown')}\n"
               f"claim_type: {state.get('claim_type', 'other')}\n"
               f"claim_value: {float(state.get('claim_value') or 0.0):.2f}\n"
               f"incident_date: {state.get('incident_date', '')}\n"
               f"decision: {state.get('decision', '').upper()}\n"
               f"reason: {state.get('decision_reason', '')}\n"
               "Post the audit-trail message to #claims-ops now.")
    result = notifier_agent.invoke({"messages": [HumanMessage(content=user_msg)]})
    return {"messages": [AIMessage(content=f"Slack posted: {result['messages'][-1].content[:200]}", name="notifier")]}


def supervisor(state: ClaimState) -> dict:
    if state.get("raw_email_body_redacted") is None: return {"next_worker": "redactor"}
    if state.get("policy_number") is None: return {"next_worker": "extractor"}
    if state.get("policyholder_found") is None and state.get("policy_state") is None:
        return {"next_worker": "policy_lookup"}
    if state.get("decision") is None: return {"next_worker": "router"}
    if not any(getattr(m,"name","")=="notifier" for m in state["messages"]):
        return {"next_worker": "notifier"}
    return {"next_worker": "DONE"}

def route(state: ClaimState) -> str:
    nxt = state["next_worker"]
    return nxt if nxt in {"redactor","extractor","policy_lookup","router","notifier"} else "__end__"


g = StateGraph(ClaimState)
g.add_node("supervisor", supervisor)
g.add_node("redactor", redactor_node)
g.add_node("extractor", extractor_node)
g.add_node("policy_lookup", policy_lookup_node)
g.add_node("router", router_node)
g.add_node("notifier", notifier_node)
g.add_edge(START, "supervisor")
g.add_conditional_edges("supervisor", route, {
    "redactor":"redactor","extractor":"extractor","policy_lookup":"policy_lookup",
    "router":"router","notifier":"notifier","__end__": END})
for w in ("redactor","extractor","policy_lookup","router","notifier"):
    g.add_edge(w, "supervisor")

conn = sqlite3.connect("claims_intake.db", check_same_thread=False)
app = g.compile(checkpointer=SqliteSaver(conn))

if __name__ == "__main__":
    DEMO_EMAIL = """From: John A. Smith <john.smith@example.com>
Phone: +1 (415) 555-2042
SSN: 123-45-6789
Address: 1414 Oak Street, Palo Alto, CA

Hello Claims Team,

I am submitting a claim under policy P-90142. On 2026-04-29 my parked Honda Civic was hit
by a delivery van in the parking lot of my apartment building. The estimated repair cost
from the body shop is $3,250.

Please process this auto claim at your earliest convenience.

Thanks,
John"""

    config = {"configurable": {"thread_id": "claim-EML-7821"}, "recursion_limit": 40}
    initial: ClaimState = {
        "messages": [HumanMessage(content="New claim: EML-7821")],
        "next_worker": "", "raw_email_id": "EML-7821", "raw_email_body": DEMO_EMAIL,
        "raw_email_body_redacted": None, "policy_number": None, "incident_date": None,
        "claim_value": None, "claim_type": None, "policy_state": None,
        "policyholder_found": None, "decision": None, "decision_reason": None}  # type: ignore
    result = app.invoke(initial, config=config)
    print("=== FINAL STATE ===")
    for k, v in result.items():
        if k != "messages": print(f"{k}: {str(v)[:140]}")
    print("\n=== MESSAGE TRACE ===")
    for m in result["messages"]:
        print(f"[{getattr(m,'name',m.type)}] {str(m.content)[:200]}")
    try:
        Path("claims_intake_graph.png").write_bytes(app.get_graph().draw_mermaid_png())
        print("\ngraph saved -> claims_intake_graph.png")
    except Exception:
        print(app.get_graph().draw_ascii())
