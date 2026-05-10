"""
Project #4 - Competitive Intelligence
Walk Notion competitors DB, scan each, write findings to weekly Notion page, broadcast highlights to Slack.
Workers: list_reader -> scanner -> writer -> broadcaster (fan-out per competitor)
"""

import os, sqlite3, json
from pathlib import Path
from datetime import date
from dotenv import load_dotenv
load_dotenv(".env")

assert os.getenv("OPENAI_API_KEY")
assert os.getenv("COMPOSIO_API_KEY"), "Connect Notion + Tavily + Slack on composio.dev"
print("env OK")

NOTION_COMPETITORS_DB_ID = os.getenv("NOTION_COMPETITORS_DB_ID", "REPLACE_WITH_YOUR_NOTION_DB_ID")
NOTION_PARENT_PAGE_ID = os.getenv("NOTION_PARENT_PAGE_ID", "REPLACE_WITH_YOUR_PARENT_PAGE_ID")

from typing import TypedDict, Annotated, Optional
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage

class CIState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    next_worker: str
    competitors: Optional[list[str]]
    pending: Optional[list[str]]
    current_competitor: Optional[str]
    findings: dict
    notion_page_id: Optional[str]
    broadcast_done: bool

from langchain_openai import ChatOpenAI
from composio_langgraph import Action, ComposioToolSet
from langgraph.prebuilt import create_react_agent

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
toolset = ComposioToolSet()

notion_db_tools = toolset.get_tools(actions=[Action.NOTION_QUERY_DATABASE])
tavily_tools = toolset.get_tools(actions=[Action.TAVILY_TAVILY_SEARCH, Action.TAVILY_TAVILY_EXTRACT])
notion_write_tools = toolset.get_tools(actions=[Action.NOTION_CREATE_NOTION_PAGE, Action.NOTION_APPEND_TEXT_BLOCKS])
slack_tools = toolset.get_tools(actions=[Action.SLACK_SENDS_A_MESSAGE_TO_A_SLACK_CHANNEL])

LIST_READER_PROMPT = """Query the Notion competitors database. Extract Name (or Title) per row.
Output ONLY a JSON array of strings on the final line: ["Acme", "Globex", "Initech"]"""

list_reader_agent = create_react_agent(llm, notion_db_tools, prompt=LIST_READER_PROMPT)

def list_reader_node(state: CIState) -> dict:
    instruction = f"Query Notion database id '{NOTION_COMPETITORS_DB_ID}'. Return all competitor names as JSON array."
    result = list_reader_agent.invoke({"messages": [HumanMessage(content=instruction)]})
    final = result["messages"][-1].content.strip()
    competitors: list[str] = []
    for line in reversed(final.splitlines()):
        line = line.strip()
        if line.startswith("[") and line.endswith("]"):
            try: competitors = json.loads(line); break
            except json.JSONDecodeError: continue
    if not competitors:
        try: competitors = json.loads(final)
        except Exception: competitors = ["Acme Corp", "Globex Inc", "Initech"]
    competitors = [c for c in competitors if isinstance(c, str) and c.strip()]
    return {"competitors": competitors, "pending": list(competitors), "findings": {},
            "messages": [AIMessage(content=f"Loaded {len(competitors)} competitors.", name="list_reader")]}

SCANNER_PROMPT = """For one competitor, search news from past 7 days about: launches, pricing,
funding, hires, layoffs, M&A. Output EXACTLY 5 bullet lines (each "- ..."). No preamble.
Each bullet 1 short sentence with date and source domain in parens."""

scanner_agent = create_react_agent(llm, tavily_tools, prompt=SCANNER_PROMPT)

def scanner_node(state: CIState) -> dict:
    name = state.get("current_competitor")
    if not name:
        return {"messages": [AIMessage(content="No competitor selected.", name="scanner")]}
    query = f'"{name}" launches OR pricing OR funding OR layoffs past 7 days'
    result = scanner_agent.invoke({"messages": [HumanMessage(content=query)]})
    raw = result["messages"][-1].content.strip()
    bullets = [ln.strip() for ln in raw.splitlines() if ln.strip().startswith("- ")]
    while len(bullets) < 5: bullets.append("- (no additional finding)")
    bullets = bullets[:5]
    findings = dict(state.get("findings") or {})
    findings[name] = bullets
    pending = [c for c in (state.get("pending") or []) if c != name]
    return {"findings": findings, "pending": pending, "current_competitor": None,
            "messages": [AIMessage(content=f"Scanned {name}: 5 bullets.", name="scanner")]}

WRITER_PROMPT = """Create a Notion page titled "Weekly CI - <date>" under the parent page given,
then append text blocks for each competitor (heading + 5 bullets).
After writing, output ONLY the new page id on the final line."""

writer_agent = create_react_agent(llm, notion_write_tools, prompt=WRITER_PROMPT)

def writer_node(state: CIState) -> dict:
    findings = state.get("findings") or {}
    if not findings:
        return {"messages": [AIMessage(content="No findings to write.", name="writer")]}
    today = date.today().isoformat()
    body_blocks: list[str] = []
    for comp, bullets in findings.items():
        body_blocks.append(f"## {comp}")
        body_blocks.extend(bullets)
        body_blocks.append("")
    body_text = "\n".join(body_blocks)
    instruction = (f"Step 1: Create Notion page under parent '{NOTION_PARENT_PAGE_ID}' titled 'Weekly CI - {today}'.\n"
                  f"Step 2: Append these blocks:\n\n{body_text}\n\n"
                  f"Return only the new page id on the last line.")
    result = writer_agent.invoke({"messages": [HumanMessage(content=instruction)]})
    final = result["messages"][-1].content.strip()
    page_id = final.splitlines()[-1].strip().strip("`'\"")
    return {"notion_page_id": page_id,
            "messages": [AIMessage(content=f"Wrote weekly CI page: {page_id}", name="writer")]}

BROADCAST_PROMPT = """Post to #competitive-intel: 'Weekly CI - top highlights' followed by 5 bullets.
After posting, confirm 'POSTED'."""

broadcast_agent = create_react_agent(llm, slack_tools, prompt=BROADCAST_PROMPT)

def _rank_top_highlights(findings: dict, k: int = 5) -> list[str]:
    rounds: list[list[str]] = []
    max_len = max((len(v) for v in findings.values()), default=0)
    for i in range(max_len):
        for comp, bullets in findings.items():
            if i < len(bullets):
                rounds.append([f"- [{comp}] {bullets[i].lstrip('- ').strip()}"])
    flat = [item for sub in rounds for item in sub]
    return flat[:k]

def broadcaster_node(state: CIState) -> dict:
    findings = state.get("findings") or {}
    top = _rank_top_highlights(findings, k=5)
    page_id = state.get("notion_page_id") or "n/a"
    body = f"Weekly CI - top highlights\nFull report: Notion page {page_id}\n" + "\n".join(top)
    instruction = f"Post to #competitive-intel:\n\n{body}"
    result = broadcast_agent.invoke({"messages": [HumanMessage(content=instruction)]})
    return {"broadcast_done": True,
            "messages": [AIMessage(content=f"Broadcast: {result['messages'][-1].content[:140]}", name="broadcaster")]}

def supervisor(state: CIState) -> dict:
    if state.get("competitors") is None: return {"next_worker": "list_reader"}
    pending = state.get("pending") or []
    if pending:
        if state.get("current_competitor") is None: return {"next_worker": "set_current"}
        return {"next_worker": "scanner"}
    if state.get("notion_page_id") is None: return {"next_worker": "writer"}
    if not state.get("broadcast_done"): return {"next_worker": "broadcaster"}
    return {"next_worker": "DONE"}

def set_current_node(state: CIState) -> dict:
    pending = state.get("pending") or []
    if not pending: return {"current_competitor": None}
    return {"current_competitor": pending[0],
            "messages": [AIMessage(content=f"Now scanning: {pending[0]}", name="supervisor")]}

def route(state: CIState) -> str:
    nxt = state["next_worker"]
    return nxt if nxt in {"list_reader","set_current","scanner","writer","broadcaster"} else "__end__"

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

g = StateGraph(CIState)
g.add_node("supervisor", supervisor)
g.add_node("list_reader", list_reader_node)
g.add_node("set_current", set_current_node)
g.add_node("scanner", scanner_node)
g.add_node("writer", writer_node)
g.add_node("broadcaster", broadcaster_node)
g.add_edge(START, "supervisor")
g.add_conditional_edges("supervisor", route, {
    "list_reader":"list_reader","set_current":"set_current","scanner":"scanner",
    "writer":"writer","broadcaster":"broadcaster","__end__": END})
for w in ("list_reader","set_current","scanner","writer","broadcaster"):
    g.add_edge(w, "supervisor")

conn = sqlite3.connect("competitive_intel.db", check_same_thread=False)
app = g.compile(checkpointer=SqliteSaver(conn))

if __name__ == "__main__":
    config = {"configurable": {"thread_id": f"ci-{date.today().isoformat()}"}, "recursion_limit": 80}
    initial = {"messages": [HumanMessage(content="Run weekly competitive intelligence sweep.")],
              "findings": {}, "broadcast_done": False}
    result = app.invoke(initial, config=config)
    print("=== FINAL STATE ===")
    print(f"Competitors: {len(result.get('competitors') or [])}")
    print(f"Pending: {len(result.get('pending') or [])}")
    print(f"Notion page id: {result.get('notion_page_id')}")
    print(f"Broadcast done: {result.get('broadcast_done')}")
    for comp, bullets in (result.get("findings") or {}).items():
        print(f"  {comp}: {len(bullets)} bullets")
        if bullets: print(f"    {bullets[0][:140]}")
    print("\n=== MESSAGE TRACE ===")
    for m in result["messages"]:
        print(f"[{getattr(m,'name',m.type)}] {m.content[:160]}")
    try:
        Path("graph_ci.png").write_bytes(app.get_graph().draw_mermaid_png())
        print("\ngraph saved -> graph_ci.png")
    except Exception:
        print(app.get_graph().draw_ascii())
