import os
from langchain.tools import  tool
from langchain_tavily import TavilySearch
from core.config import AppConfig
from dotenv import load_dotenv
load_dotenv()
@tool
def WebSearchTool(query:str):
    """tool for performing web search """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("Missing Tavily config: TAVILY_API_KEY not set.")
    tavily_tool = TavilySearch(
    api_key=api_key,
    max_results=3,
    topic="general",
    include_answer=True,          # <-- REQUIRED
    include_images=False,
    include_raw_content=True,
    )
    response = tavily_tool.invoke({"query": query})


from bs4 import BeautifulSoup as bs
    

import requests
@tool
def WebScrapingTool(url:str):
    """tool for performing web scraping """
    response = requests.get(url)
    soup = bs(response.text, 'html.parser')
    
    

    