"""
Agent builder.

This module exposes a single factory function, :func:`build_agent`, which
constructs a LangChain agent capable of executing
a specified task. The agent is built using a language model provider and a set of tools, which are defined in the configuration.
"""

from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.tools import BaseTool
from langgraph.checkpoint.memory import InMemorySaver

from .core.config import AgentConfig
from .core.state import BaseAgentState
from .llm.base import get_llm
from .middleware import (

)
from .tools import 

async def build_agent():

    llm = get_llm()

    tools : list[BaseTool] = []

    #Agent Definition
    agent  = create_agent(
        model = llm,
        tools=tools,
        system_prompt = cfg.system_prompt,
        checkpointer=InMemorySaver(),
        middleware= [

        ],

    )
    return agent