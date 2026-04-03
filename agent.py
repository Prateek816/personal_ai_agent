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
from langchain_community.tools import TavilySearchResults
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

from dotenv import load_dotenv
import os
from datetime import datetime
from typing import  Literal
from pydantic import BaseModel, Field, validator , constr


load_dotenv()

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="openai/gpt-oss-120b",
    temperature=0,
)

from langchain.tools import tool
from langchain_community.agent_toolkits import GmailToolkit
from langchain_google_community.gmail.utils import (
    build_gmail_service,     
    get_google_credentials  
)

credentials = get_google_credentials(
    token_file="token.json",
    scopes=["https://mail.google.com/"],
    client_secrets_file="credentials.json",
)

api_resource = build_gmail_service(credentials=credentials)
gmail_toolkit = GmailToolkit(api_resource=api_resource)
gmail_toolkit = gmail_toolkit.get_tools()

import os
from dotenv import load_dotenv
load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEYnot foundin environment variables.")
from langchain_tavily import TavilySearch
tavily_tool = TavilySearch(
    api_key=TAVILY_API_KEY,
    max_results=5,
    topic="general",
    include_answer=True,
    include_images=False,
    include_raw_content=True,
)

