from __future__ import annotations
from typing import Literal
import os
from dotenv import load_dotenv

from langchain_core.tools import tool
from langgraph.graph import MessagesState
from langgraph.types import Command
from langchain_tavily import TavilySearch
from langchain_groq import ChatGroq

load_dotenv()

_tavily = TavilySearch(
    api_key=os.getenv("TAVILY_API_KEY"),
    max_results=5,
    topic="general",
    include_answer=True,
    include_images=False,
    include_raw_content=True,
)

# Web search tool for gathering research data
@tool
def web_search(user_query: str) -> str:
    """Search the web for up-to-date, factual information based on a user query."""

    if _tavily is None:
        return "Web search unavailable. Set TAVILY_API_KEY to enable this feature."

    # Refine the query to make it more search-friendly
    refined_query = f"{user_query} latest facts statistics sources"

    # Call Tavily search
    search_results = _tavily.invoke(
        input=refined_query,
        search_depth="basic",
        max_results=2
    )

    return str(search_results)


# Conducts research and gathers information for the blog post
def web_agent(state: MessagesState) -> Command[Literal["supervisor"]]:
    """Fetches and summarizes relevant information from the web.

    Uses web search to gather current, factual data and returns a concise,
    structured summary useful for downstream tasks.
    """

    print("🌐 Web Agent: Fetching information...")
    
    model = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model="openai/gpt-oss-120b",
        temperature=0,
    )

    system = (
        "You are a web intelligence agent. Your job is to extract and summarize useful, accurate, "
        "and up-to-date information from web search results. Focus on key facts, important details, "
        "and reliable insights. Avoid fluff and keep the output structured and concise."
    )

    messages = state["messages"] + [{"role": "system", "content": system}]
    
    # Build search query from latest user intent
    search_query = f"{state['messages'][-1].content if state['messages'] else 'general topic'} latest information facts"
    
    # Perform web search
    search_results = web_search.invoke(search_query)['answer']
    
    # Summarize results
    summary_prompt = (
        "Extract key insights from the following web search results. Provide:\n"
        "- Key facts\n"
        "- Important insights\n"
        "- Any relevant statistics or data\n\n"
        f"{search_results}"
    )

    web_summary = model.invoke([{"role": "user", "content": summary_prompt}])
    
    print("🌐 Web Agent: Completed!")
    
    return Command(
        goto="supervisor",
        update={
            "messages": [web_summary, {"role": "user", "content": "Web"}],
        },
    )