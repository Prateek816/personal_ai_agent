from langchain_groq import ChatGroq

from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, SystemMessage,AIMessage
from langchain_core.prompts.chat import ChatPromptTemplate,MessagesPlaceholder
from typing_extensions import TypedDict, Annotated
from langgraph.graph import MessagesState, END
from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver
from langchain_classic.tools import StructuredTool
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.prebuilt import create_react_agent
from langchain.agents import create_agent 

from dotenv import load_dotenv
import os
from datetime import datetime
from typing import  Literal
from pydantic import BaseModel, Field, validator , constr

from langchain_tavily import TavilySearch
from langchain.tools import tool
from langchain_community.agent_toolkits import GmailToolkit
from langchain_google_community.gmail.utils import (
    build_gmail_service,     
    get_google_credentials  
)
from langchain_google_community import CalendarToolkit
from langchain_google_community.calendar.utils import (
    build_calendar_service,
    get_google_credentials,
)


load_dotenv()

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="openai/gpt-oss-120b",
    temperature=0,
)
#Gmail Toolkit
credentials = get_google_credentials(
    token_file="token.json",
    scopes=["https://mail.google.com/"],
    client_secrets_file="credentials.json",
)
api_resource = build_gmail_service(credentials=credentials)
gmail_toolkit = GmailToolkit(api_resource=api_resource)
gmail_tools = gmail_toolkit.get_tools()

#Calendar Toolkit
credentials = get_google_credentials(
    token_file="token.json",
    scopes=["https://www.googleapis.com/auth/calendar"],
    client_secrets_file="credentials.json",
)
api_resource = build_calendar_service(credentials=credentials)
calendar_toolkit = CalendarToolkit(api_resource=api_resource)
calendar_tools = calendar_toolkit.get_tools()
print(calendar_tools)

#Web search toolkit
web_toolkit = []

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEYnot foundin environment variables.")
tavily_tool = TavilySearch(
    api_key=TAVILY_API_KEY,
    max_results=5,
    topic="general",
    include_answer=True,
    include_images=False,
    include_raw_content=True,
)
web_toolkit.append(tavily_tool)

#web agent , search agent , email agent , calendar agent , coding agent

class AgentState(BaseModel):
    messages : Annotated[list[HumanMessage | AIMessage], Field(description="The conversation history between the user and the agent.")]
    current_worker: Annotated[str, Field(description="The worker currently handling the conversation.")]
    user_id: Annotated[str, Field(description="The unique identifier for the user.")]
    completed_stage: Annotated[list[str], Field(description="List of completed stages in the workflow.")] = []


from langgraph.types import Command

"""def content_planner(state:AgentState)->Command(Literal['supervisor']):

    print("Content Planner is processing the conversation history to determine the next worker.")

    messages = state.messages+[HumanMessage(content="Based on the above conversation history, which worker should handle the next part of the conversation? The options are: 'web_agent', 'search_agent', 'email_agent', 'calendar_agent', 'coding_agent'. Please respond with only the name of the worker.")]
    plan = llm.invoke(messages)
    plan = plan.content
    print("Content planner : planning completed")

    return Command(
        goto="supervisor",
        update={
            "messages":[plan]
        }
    )"""


def supervisor_node(state:AgentState)->Command:
    """LLM-based supervisor that decides which agent to call next"""
    print("Supervisor is evaluating the current state to decide the next worker.")
    messages = state.messages+[HumanMessage(content="Based on the conversation history and the completed stages, which worker should handle the next part of the conversation? The options are: 'web_agent', 'search_agent', 'email_agent', 'calendar_agent', 'coding_agent'. Please respond with only the name of the worker.")]
    next_worker = llm.invoke(messages)
    next_worker = next_worker.content.strip().lower()
    print(f"Supervisor decided to call: {next_worker}")
    
    if next_worker not in ['web_agent', 'search_agent', 'email_agent', 'calendar_agent', 'coding_agent']:
        print(f"Supervisor received an invalid worker name: {next_worker}. Defaulting to 'web_agent'.")
        next_worker = 'web_agent'
    
    return Command(
        goto=next_worker,
        update={
            "current_worker": next_worker
        }
    )

def make_dummy_agent(
    agent_name: str,
    description: str,
    next_node: str = "supervisor",
):
    """
    Returns a node function that:
      - Simulates the agent doing work
      - Appends a dummy AIMessage to state
      - Returns a Command routing back to supervisor (or END)
    """
    def agent_node(state: AgentState) -> Command:
        user_input = state["messages"][0].content          # original query
        print(f"\n🤖  [{agent_name}] working on: {user_input[:72]}…")
        print(f"    Role: {description}")

        dummy_output = (
            f"[{agent_name}] ✅  Task complete.\n"
            f"Simulated result for query: '{user_input[:60]}'\n"
            f"Agent capability: {description}"
        )
        print(f"    Output: {dummy_output[:80]}…")

        return Command(
            goto=next_node,
            update={
                "messages": AIMessage(content=dummy_output),
            },
        )

    agent_node.__name__ = agent_name
    return agent_node

web_agent = make_dummy_agent(
    agent_name  = "web_agent",
    description = "Browses websites, scrapes pages, and fetches live web content.",
    next_node   = "supervisor",     # hand back control to supervisor
)

search_agent = make_dummy_agent(
    agent_name  = "search_agent",
    description = "Runs search-engine queries and ranks the most relevant results.",
    next_node   = "supervisor",
)

email_agent = make_dummy_agent(
    agent_name  = "email_agent",
    description = "Reads, drafts, and sends emails via connected mail accounts.",
    next_node   = "supervisor",
)

calendar_agent = make_dummy_agent(
    agent_name  = "calendar_agent",
    description = "Checks availability, creates events, and manages calendar entries.",
    next_node   = "supervisor",
)

coding_agent = make_dummy_agent(
    agent_name  = "coding_agent",
    description = "Writes, reviews, and executes Python/JavaScript code snippets.",
    next_node   = "supervisor",
)
def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # ── Register nodes ────────────────────────
    graph.add_node("supervisor",     supervisor_node)
    graph.add_node("web_agent",      web_agent)
    graph.add_node("search_agent",   search_agent)
    graph.add_node("email_agent",    email_agent)
    graph.add_node("calendar_agent", calendar_agent)
    graph.add_node("coding_agent",   coding_agent)

    # ── Entry point ───────────────────────────
    graph.set_entry_point("supervisor")

    # NOTE: No explicit add_edges needed here.
    # Command(goto=...) handles all routing dynamically.
    # LangGraph requires us to declare possible destinations
    # so it can validate the graph — done via Command[WorkerName].

    return graph.compile()


app = build_graph()

from IPython.display import Image, display

# Option 1 — display inline (Jupyter / VS Code notebook)
display(Image(app.get_graph().draw_mermaid_png()))

# Option 2 — save to disk
png_bytes = app.get_graph().draw_mermaid_png()
with open("workflow.png", "wb") as f:
    f.write(png_bytes)
print("Graph saved to workflow.png")