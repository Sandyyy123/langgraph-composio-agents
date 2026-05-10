"""
Project #3 - Meeting Prep Briefer
Loop over today's meetings; for each: research attendees, pull email history, compose brief, DM via Slack.
Workers: calendar_reader -> attendee_researcher -> history_puller -> composer (loops per meeting)
"""

import os, sqlite3, json
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(".env")

assert os.getenv("OPENAI_API_KEY")
assert os.getenv("COMPOSIO_API_KEY"), "Connect Calendar + Tavily + Notion + Gmail + Slack on composio.dev"
print("env OK")

from typing import TypedDict, Annotated, Optional
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage

class MeetingState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    next_worker: str
    meetings: list[dict]
    current_meeting: Optional[dict]
    attendee_research: Optional[str]
    history_summary: Optional[str]
    brief: Optional[str]
    completed_meetings: list[dict]

from langchain_openai import ChatOpenAI
from composio_langgraph import Action, ComposioToolSet
from langgraph.prebuilt import create_react_agent

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
toolset = ComposioToolSet()

calendar_tools = toolset.get_tools(actions=[Action.GOOGLECALENDAR_LIST_EVENTS])
research_tools = toolset.get_tools(actions=[Action.TAVILY_TAVILY_SEARCH, Action.NOTION_SEARCH_NOTION_PAGE])
gmail_tools = toolset.get_tools(actions=[Action.GMAIL_FETCH_EMAILS])
slack_tools = toolset.get_tools(actions=[Action.SLACK_SENDS_A_MESSAGE_TO_A_SLACK_CHANNEL])

CAL_PROMPT = """List the user's events for today (next 16 hours from now).
Output one JSON object per line with keys: time, attendees (list of emails), title.
JSON lines only, no prose, no markdown fences."""

calendar_agent = create_react_agent(llm, calendar_tools, prompt=CAL_PROMPT)

def calendar_reader_node(state: MeetingState) -> dict:
    from datetime import datetime, timedelta, timezone
    now = datetime.now(timezone.utc)
    end = now + timedelta(hours=16)
    instruction = f"List events from {now.isoformat()} to {end.isoformat()}. Output JSON-per-line."
    result = calendar_agent.invoke({"messages": [HumanMessage(content=instruction)]})
    final = result["messages"][-1].content
    meetings: list[dict] = []
    for line in final.splitlines():
        line = line.strip().strip(",")
        if not line.startswith("{"): continue
        try:
            obj = json.loads(line)
            if "title" in obj:
                meetings.append({"time": obj.get("time", "unknown"),
                                "attendees": obj.get("attendees", []) or [],
                                "title": obj.get("title", "Untitled")})
        except json.JSONDecodeError:
            continue
    if not meetings:
        meetings = [{"time": (now + timedelta(hours=2)).isoformat(),
                    "attendees": ["partner@example.com"],
                    "title": "Sync with partner"}]
    return {"meetings": meetings,
            "messages": [AIMessage(content=f"Loaded {len(meetings)} meetings.", name="calendar_reader")]}

RESEARCH_PROMPT = """For each attendee, search the web (Tavily) AND search internal Notion pages.
Return a concise plaintext brief (max 6 sentences total) covering: role, company, recent activity, prior notes.
No JSON, no markdown."""

research_agent = create_react_agent(llm, research_tools, prompt=RESEARCH_PROMPT)

def attendee_researcher_node(state: MeetingState) -> dict:
    meeting = state["current_meeting"] or {}
    attendees = meeting.get("attendees", [])
    if not attendees:
        return {"attendee_research": "No external attendees.",
                "messages": [AIMessage(content="No attendees to research.", name="researcher")]}
    instruction = (f"Meeting: {meeting.get('title')}\nAttendees: {', '.join(attendees)}\n"
                  f"Research each via Tavily and Notion. Concatenate findings.")
    result = research_agent.invoke({"messages": [HumanMessage(content=instruction)]})
    text = result["messages"][-1].content.strip()
    return {"attendee_research": text,
            "messages": [AIMessage(content=f"Research done ({len(text)} chars).", name="researcher")]}

HISTORY_PROMPT = """Fetch the most recent 5 emails between the user and any of those attendees,
then summarise in EXACTLY 3 bullet points (lines starting with "- "). No preamble."""

history_agent = create_react_agent(llm, gmail_tools, prompt=HISTORY_PROMPT)

def history_puller_node(state: MeetingState) -> dict:
    meeting = state["current_meeting"] or {}
    attendees = meeting.get("attendees", [])
    if not attendees:
        return {"history_summary": "- No prior history.\n- N/A\n- N/A",
                "messages": [AIMessage(content="Skipped history.", name="history")]}
    query = " OR ".join([f"from:{a} OR to:{a}" for a in attendees])
    instruction = f"Fetch up to 5 most recent emails matching: {query}. Summarise in 3 bullets."
    result = history_agent.invoke({"messages": [HumanMessage(content=instruction)]})
    summary = result["messages"][-1].content.strip()
    if summary.count("- ") < 3:
        bullets = [b for b in summary.splitlines() if b.strip().startswith("- ")]
        while len(bullets) < 3: bullets.append("- (no further history)")
        summary = "\n".join(bullets[:3])
    return {"history_summary": summary,
            "messages": [AIMessage(content=f"History summarised.", name="history")]}

COMPOSER_SYS = """You compose meeting prep briefs. Sections:
- Meeting: title and time
- Attendees: bullet list
- Background: 2-3 sentences
- Recent history: 3 bullets
- Suggested talking points: 3 bullets
Under 300 words total."""

SLACK_DM_TARGET = os.getenv("SLACK_DM_TARGET", "@me")
slack_dm_agent = create_react_agent(llm, slack_tools,
    prompt=f"DM the given text to {SLACK_DM_TARGET}. After sending, confirm 'SENT'.")

def composer_node(state: MeetingState) -> dict:
    meeting = state["current_meeting"] or {}
    composed = llm.invoke([
        SystemMessage(content=COMPOSER_SYS),
        HumanMessage(content=(
            f"Meeting: {meeting.get('title')}\nTime: {meeting.get('time')}\n"
            f"Attendees: {', '.join(meeting.get('attendees', []))}\n"
            f"Attendee research:\n{state.get('attendee_research', '')}\n\n"
            f"Email history summary:\n{state.get('history_summary', '')}"))])
    brief = composed.content.strip()
    slack_dm_agent.invoke({"messages": [HumanMessage(content=f"Send this brief:\n\n{brief}")]})

    archived = list(state.get("completed_meetings") or [])
    archived.append({"meeting": meeting, "brief": brief})
    remaining = list(state.get("meetings") or [])
    if remaining and remaining[0] == meeting: remaining = remaining[1:]

    return {"brief": brief, "completed_meetings": archived, "meetings": remaining,
            "current_meeting": None, "attendee_research": None, "history_summary": None,
            "messages": [AIMessage(content=f"Brief sent for '{meeting.get('title')}'.", name="composer")]}

def supervisor(state: MeetingState) -> dict:
    if state.get("meetings") is None: return {"next_worker": "calendar_reader"}
    if state.get("current_meeting") is None:
        return {"next_worker": "set_current"} if state.get("meetings") else {"next_worker": "DONE"}
    if state.get("attendee_research") is None: return {"next_worker": "researcher"}
    if state.get("history_summary") is None: return {"next_worker": "history"}
    if not state.get("brief"): return {"next_worker": "composer"}
    return {"next_worker": "set_current"}

def set_current_node(state: MeetingState) -> dict:
    queue = state.get("meetings") or []
    if not queue:
        return {"current_meeting": None, "messages": [AIMessage(content="All meetings done.", name="supervisor")]}
    return {"current_meeting": queue[0], "brief": None,
            "attendee_research": None, "history_summary": None,
            "messages": [AIMessage(content=f"Now: {queue[0].get('title')}", name="supervisor")]}

def route(state: MeetingState) -> str:
    nxt = state["next_worker"]
    return nxt if nxt in {"calendar_reader", "set_current", "researcher", "history", "composer"} else "__end__"

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

g = StateGraph(MeetingState)
g.add_node("supervisor", supervisor)
g.add_node("calendar_reader", calendar_reader_node)
g.add_node("set_current", set_current_node)
g.add_node("researcher", attendee_researcher_node)
g.add_node("history", history_puller_node)
g.add_node("composer", composer_node)
g.add_edge(START, "supervisor")
g.add_conditional_edges("supervisor", route, {
    "calendar_reader":"calendar_reader", "set_current":"set_current",
    "researcher":"researcher", "history":"history", "composer":"composer", "__end__": END})
for w in ("calendar_reader","set_current","researcher","history","composer"):
    g.add_edge(w, "supervisor")

conn = sqlite3.connect("meeting_briefer.db", check_same_thread=False)
app = g.compile(checkpointer=SqliteSaver(conn))

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "meeting-day-1"}, "recursion_limit": 60}
    initial = {"messages": [HumanMessage(content="Prep me for today's meetings.")],
              "completed_meetings": []}
    result = app.invoke(initial, config=config)
    print("=== FINAL STATE ===")
    print(f"Meetings remaining: {len(result.get('meetings') or [])}")
    print(f"Meetings completed: {len(result.get('completed_meetings') or [])}")
    for entry in (result.get("completed_meetings") or []):
        print(f"\n--- {entry['meeting'].get('title')} ---\n{entry['brief'][:200]}...")
    print("\n=== MESSAGE TRACE ===")
    for m in result["messages"]:
        print(f"[{getattr(m,'name',m.type)}] {m.content[:160]}")
    try:
        Path("graph_meeting.png").write_bytes(app.get_graph().draw_mermaid_png())
        print("\ngraph saved -> graph_meeting.png")
    except Exception:
        print(app.get_graph().draw_ascii())
