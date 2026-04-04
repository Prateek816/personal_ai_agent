from __future__ import annotations
import os
from datetime import datetime
from typing import Annotated, Literal
from typing_extensions import TypedDict

from dotenv import load_dotenv
from pydantic import BaseModel, Field, validator, constr

# LangGraph
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.types import Command
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import tools_condition, ToolNode, create_react_agent

# LangChain Core
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnableConfig

# LLM
from langchain_groq import ChatGroq

# Tools
from langchain.tools import tool
from langchain_classic.tools import StructuredTool
from langchain_tavily import TavilySearch

# Google Integrations
from langchain_community.agent_toolkits import GmailToolkit
from langchain_google_community.gmail.utils import (
    build_gmail_service,
    get_google_credentials,
)

from langchain_google_community import CalendarToolkit
from langchain_google_community.calendar.utils import (
    build_calendar_service,
    get_google_credentials,
)
#Agents
from agents.calender_agent import calendar_agent
from agents.web_agent import web_agent
from agents.gmail_agent import gmail_agent

load_dotenv()

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="openai/gpt-oss-120b",
    temperature=0,
)

class AgentState(BaseModel):
    messages : Annotated[list,add_messages] = Field(description="The conversation history as a list of messages.")
    current_worker: Annotated[str, Field(description="The worker currently handling the conversation.")]
    user_id: Annotated[str, Field(description="The unique identifier for the user.")]
    completed_stage: Annotated[list[str], Field(description="List of completed stages in the workflow.")] = []

class WorkerName(BaseModel):
    worker: Annotated[Literal['web_agent','email_agent', 'summarize_agent', 'calender_agent'], Field(description="The name of the worker to handle the next part of the conversation.")]

prompt = """You are a Supervisor (Orchestrator) Agent responsible for controlling the workflow between specialized agents.

Your role is to decide which agent should act next based on:
- the user’s request
- the current conversation state
- the progress of the task

---

### Available Agents

1. web_agent  
→ Use for:
- fetching external information
- searching the internet
- gathering facts or data

2. gmail_agent  
→ Use for:
- sending emails
- reading/searching emails
- drafting or replying

3. calendar_agent  
→ Use for:
- scheduling events
- checking availability
- updating or deleting events

4. summarize_agent  
→ Use ONLY when:
- the task is fully completed
- all required actions have been executed
- no further tool calls are needed

---

### Core Decision Rules

1. ROUTE BASED ON INTENT
- If task involves information gathering → web_agent
- If task involves email actions → gmail_agent
- If task involves scheduling → calendar_agent

2. MULTI-STEP TASKS
- Break tasks logically into steps
- Call agents sequentially as needed
- Do NOT jump to summarize_agent prematurely

3. TERMINATION LOGIC (VERY IMPORTANT)
- When ALL required steps are completed:
  → call summarize_agent
- summarize_agent MUST be the FINAL step

4. NEVER LOOP UNNECESSARILY
- Do not call the same agent repeatedly without new purpose

5. DO NOT EXECUTE TASKS YOURSELF
- You ONLY decide which agent runs next

---

### Output Rules (STRICT)

- Respond with ONLY one of:
  web_agent
  gmail_agent
  calendar_agent
  summarize_agent

- Do NOT include explanations
- Do NOT include extra text

---

### Examples

User: "Find AI events and email me"
→ web_agent (first step)

After web results:
→ gmail_agent

After email sent:
→ summarize_agent

---

### Absolute Rules

- NEVER call summarize_agent before task completion
- ALWAYS ensure logical step progression
- ALWAYS end workflow using summarize_agent"""

def supervisor_node(state:AgentState)->Command:
    """LLM-based supervisor that decides which agent to call next."""
    print("\n🔀  Supervisor is evaluating the current state to decide the next worker.")

    messages = state.messages + [
        HumanMessage(
            content=(
                prompt
            )
        )
    ]
    response = llm.invoke(messages).content.strip()

    print(f"👉 Decision: {response}")

    valid_workers = [
        "web_agent",
        "email_agent",
        "calendar_agent",
        "summarize_agent"
    ]

    if response not in valid_workers:
        print("⚠️ Invalid response, defaulting to summarize_agent")
        response = "summarize_agent"

    return Command(
        goto=response,
        update={
            "messages": AIMessage(content=f"Supervisor chose: {response}")
        }
    )

def summarize_agent(state:AgentState)->Command:
    messages = state.messages + [
        HumanMessage(
            content=(
                "Based on the conversation history and the completed stages, "
                "please provide a summary of the key points discussed so far."
            )
        )
    ]
    response = llm.invoke(messages)
    response = response['messages'][-1].content

    return Command(
        update={
            "messages": AIMessage(content=response)
        }
    )

def build_graph() :
    graph = StateGraph(AgentState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("web_agent", web_agent)
    graph.add_node("email_agent", gmail_agent)
    graph.add_node("calendar_agent", calendar_agent)
    graph.add_node("summarize_agent", summarize_agent)

    graph.add_edge(START, "supervisor")
    

    graph.add_edge("summarize_agent", END)

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

if __name__ == "__main__":
    input_state = {
        "messages": [
            HumanMessage(content="Find AI events this week and email me the details")
        ],
        "current_worker": "",
        "user_id": "user_1",
        "completed_stage": []
    }

    result = app.invoke(input_state)

    print("\n=== FINAL OUTPUT ===")
    for msg in result["messages"]:
        print(msg.content)