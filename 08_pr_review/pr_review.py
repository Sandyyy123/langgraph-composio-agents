"""
Project #8 - PR Review Assistant
Workers: diff_fetcher (GitHub) -> summariser (LLM) -> rule_checker (Notion) -> commenter (GitHub) -> notifier (Slack)
"""

import os, sqlite3, json
from pathlib import Path
from typing import TypedDict, Annotated, Optional
from dotenv import load_dotenv

load_dotenv(".env")
assert os.getenv("OPENAI_API_KEY")
assert os.getenv("COMPOSIO_API_KEY"), "Connect GitHub + Notion + Slack on composio.dev"
print("env OK")

from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage

class PRState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    next_worker: str
    pr_number: int
    repo: str
    reviewer_slack_id: Optional[str]
    diff_text: Optional[str]
    summary: Optional[str]
    rule_violations: Optional[list]
    review_posted: bool

from langchain_openai import ChatOpenAI
from composio_langgraph import Action, ComposioToolSet
from langgraph.prebuilt import create_react_agent

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
toolset = ComposioToolSet()

github_diff_tools = toolset.get_tools(actions=[Action.GITHUB_GET_PR_DIFF])
notion_tools = toolset.get_tools(actions=[Action.NOTION_QUERY_DATABASE])
github_comment_tools = toolset.get_tools(actions=[Action.GITHUB_CREATE_PR_COMMENT])
slack_tools = toolset.get_tools(actions=[Action.SLACK_SENDS_A_MESSAGE_TO_A_SLACK_CHANNEL])


DIFF_FETCHER_PROMPT = """Call GITHUB_GET_PR_DIFF with the args. Return raw unified diff.
Do not paraphrase. On error: 'DIFF_FETCH_FAILED: <reason>'."""

diff_agent = create_react_agent(llm, github_diff_tools, prompt=DIFF_FETCHER_PROMPT)

def diff_fetcher_node(state: PRState) -> dict:
    owner, name = state["repo"].split("/", 1)
    query = (f"Fetch unified diff for PR #{state['pr_number']} in {owner}/{name}. "
            f"GITHUB_GET_PR_DIFF with owner='{owner}', repo='{name}', pull_number={state['pr_number']}.")
    result = diff_agent.invoke({"messages": [HumanMessage(content=query)]})
    diff_text = result["messages"][-1].content or ""
    return {"diff_text": diff_text,
            "messages": [AIMessage(content=f"Diff fetched ({len(diff_text)} chars) for {state['repo']}#{state['pr_number']}", name="diff_fetcher")]}


SUMMARISER_PROMPT = """Output exactly 3 bullets, plain English:
- bullet 1: what changed (1 sentence)
- bullet 2: which files/modules
- bullet 3: why (inferred)
Be concrete. No 'this PR' opener."""

def summariser_node(state: PRState) -> dict:
    diff_text = state.get("diff_text") or ""
    trimmed = diff_text[:6000]
    out = llm.invoke([SystemMessage(content=SUMMARISER_PROMPT),
                     HumanMessage(content=f"Repo: {state['repo']}\nPR: #{state['pr_number']}\nDiff:\n```\n{trimmed}\n```")])
    summary = out.content.strip()
    return {"summary": summary,
            "messages": [AIMessage(content=f"Summary ready: {summary[:160]}", name="summariser")]}


RULE_CHECKER_PROMPT = """1. NOTION_QUERY_DATABASE 'PR Review Rules' (id from env or hint).
2. Common rules to check if Notion empty:
   - Tests added when src/ changed
   - Docs updated when public APIs change
   - No console.log/print/debugger in prod paths
   - No hard-coded secrets
   - Conventional Commits title (feat:, fix:, etc.)
3. Return JSON ONLY: {"violations":[{"rule":"...","severity":"low|medium|high","evidence":"..."}]}"""

rule_agent = create_react_agent(llm, notion_tools, prompt=RULE_CHECKER_PROMPT)

def rule_checker_node(state: PRState) -> dict:
    diff_text = state.get("diff_text") or ""
    trimmed = diff_text[:6000]
    db_hint = os.getenv("NOTION_PR_RULES_DB_ID", "<query 'PR Review Rules' DB>")
    query = (f"Database hint: {db_hint}\nPR: {state['repo']}#{state['pr_number']}\n"
            f"Diff:\n```\n{trimmed}\n```\nReturn violations JSON.")
    result = rule_agent.invoke({"messages": [HumanMessage(content=query)]})
    raw = (result["messages"][-1].content or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"): raw = raw[4:].strip()
    try: violations = json.loads(raw).get("violations", [])
    except json.JSONDecodeError: violations = []
    return {"rule_violations": violations,
            "messages": [AIMessage(content=f"Rule check: {len(violations)} violation(s).", name="rule_checker")]}


COMMENTER_PROMPT = """GITHUB_CREATE_PR_COMMENT with owner, repo, issue_number=<pr_number>, body=verbatim.
On success: 'COMMENT_POSTED'. On fail: 'COMMENT_FAILED: <reason>'."""

comment_agent = create_react_agent(llm, github_comment_tools, prompt=COMMENTER_PROMPT)

def _build_review_body(summary: str, violations: list) -> str:
    lines = ["## Automated PR Review", "", "### Summary", summary or "_no summary_", ""]
    if violations:
        lines.append(f"### Violations ({len(violations)})")
        for v in violations:
            sev = v.get("severity", "medium").upper()
            lines.append(f"- **[{sev}] {v.get('rule', '?')}** - {v.get('evidence', '')}")
        lines.append("")
        lines.append("Please address before merging.")
    else:
        lines.append("### Verdict")
        lines.append(":white_check_mark: **LGTM** - no rule violations.")
    lines.append("")
    lines.append("_posted by pr-review-assistant_")
    return "\n".join(lines)

def commenter_node(state: PRState) -> dict:
    body = _build_review_body(state.get("summary") or "", state.get("rule_violations") or [])
    owner, name = state["repo"].split("/", 1)
    query = (f"Post on PR #{state['pr_number']} of {owner}/{name}. "
            f"GITHUB_CREATE_PR_COMMENT with owner='{owner}', repo='{name}', "
            f"issue_number={state['pr_number']}. Body:\n\n{body}")
    result = comment_agent.invoke({"messages": [HumanMessage(content=query)]})
    confirm = (result["messages"][-1].content or "").strip()
    posted = "COMMENT_POSTED" in confirm or "posted" in confirm.lower()
    return {"review_posted": posted,
            "messages": [AIMessage(content=f"Review {'posted' if posted else 'failed'} on PR #{state['pr_number']}", name="commenter")]}


NOTIFIER_PROMPT = """SLACK_SENDS_A_MESSAGE_TO_A_SLACK_CHANNEL with channel and text verbatim.
On success: 'SLACK_OK'. On fail: 'SLACK_FAIL: <reason>'."""

notifier_agent = create_react_agent(llm, slack_tools, prompt=NOTIFIER_PROMPT)

def notifier_node(state: PRState) -> dict:
    violations = state.get("rule_violations") or []
    reviewer = state.get("reviewer_slack_id") or "@channel"
    pr_url = f"https://github.com/{state['repo']}/pull/{state['pr_number']}"
    text = (f"<{reviewer}> automated review ready for {state['repo']}#{state['pr_number']} "
           f"({len(violations)} violation{'s' if len(violations)!=1 else ''}). {pr_url}")
    channel = os.getenv("SLACK_PR_REVIEW_CHANNEL", "#pr-reviews")
    query = (f"Send to channel '{channel}'. Text:\n{text}\n"
            f"SLACK_SENDS_A_MESSAGE_TO_A_SLACK_CHANNEL with channel='{channel}'.")
    result = notifier_agent.invoke({"messages": [HumanMessage(content=query)]})
    confirm = (result["messages"][-1].content or "").strip()
    return {"messages": [AIMessage(content=f"Slack ping {'sent' if 'SLACK_OK' in confirm else 'attempt failed'}: {text[:140]}", name="notifier")]}


def supervisor(state: PRState) -> dict:
    if state.get("diff_text") is None: return {"next_worker": "diff_fetcher"}
    if state.get("summary") is None: return {"next_worker": "summariser"}
    if state.get("rule_violations") is None: return {"next_worker": "rule_checker"}
    if not state.get("review_posted"): return {"next_worker": "commenter"}
    last = state["messages"][-1].content if state["messages"] else ""
    if "Slack ping" not in last: return {"next_worker": "notifier"}
    return {"next_worker": "DONE"}

def route(state: PRState) -> str:
    nxt = state["next_worker"]
    return nxt if nxt in {"diff_fetcher","summariser","rule_checker","commenter","notifier"} else "__end__"


from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

g = StateGraph(PRState)
g.add_node("supervisor", supervisor)
g.add_node("diff_fetcher", diff_fetcher_node)
g.add_node("summariser", summariser_node)
g.add_node("rule_checker", rule_checker_node)
g.add_node("commenter", commenter_node)
g.add_node("notifier", notifier_node)
g.add_edge(START, "supervisor")
g.add_conditional_edges("supervisor", route, {
    "diff_fetcher":"diff_fetcher","summariser":"summariser","rule_checker":"rule_checker",
    "commenter":"commenter","notifier":"notifier","__end__": END})
for w in ("diff_fetcher","summariser","rule_checker","commenter","notifier"):
    g.add_edge(w, "supervisor")

conn = sqlite3.connect("pr_review.db", check_same_thread=False)
app = g.compile(checkpointer=SqliteSaver(conn))

try:
    Path("pr_review_graph.png").write_bytes(app.get_graph().draw_mermaid_png())
    print("graph saved -> pr_review_graph.png")
except Exception:
    print(app.get_graph().draw_ascii())

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "pr-42"}, "recursion_limit": 40}
    initial = {"pr_number": 42, "repo": "Sandyyy123/groverjobapps",
              "reviewer_slack_id": "U07ABC123", "review_posted": False,
              "messages": [HumanMessage(content="New PR opened, run review")]}
    result = app.invoke(initial, config=config)
    print("=== FINAL STATE ===")
    for k, v in result.items():
        if k != "messages": print(f"{k}: {str(v)[:160]}")
    print("\n=== MESSAGE TRACE ===")
    for m in result["messages"]:
        print(f"[{getattr(m,'name',m.type)}] {m.content[:200]}")
