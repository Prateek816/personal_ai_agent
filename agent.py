from langchain_groq import ChatGroq
from datetime import datetime
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, SystemMessage,AIMessage
from langchain_core.prompts.chat import ChatPromptTemplate,MessagesPlaceholder
from pydantic import BaseModel, Field, validator , constr
from typing_extensions import TypedDict, Annotated
from langgraph.graph import MessagesState, END
from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver
from langchain_classic.tools import StructuredTool
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_community.tools import TavilySearchResults
from langgraph.prebuilt import create_react_agent
from typing import  Literal
from langchain_core.tools import tool

from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="openai/gpt-oss-120b",
    temperature=0,
)

