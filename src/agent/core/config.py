"""
Configuration module for all the components in the Personal AI Agent. 
This module defines the configuration classes and settings for various components of the agent, including the language model provider, state management, and other relevant configurations. 
It provides a centralized location for managing and updating the configurations as needed for the agent's operation.
"""

import os
from dataclasses import dataclass , field
from typing import Dict, Any , List , Optional
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

@dataclass
class LLMConfig:
    """
    Configuration class for the Language Model Provider (LLM). 
    This class includes attributes for specifying the provider name, model name, and any additional parameters required for initializing the LLM client.
    """

@dataclass
class AgentConfig:

    llm: LLMConfig = field(default_factory=LLMConfig)
    @classmethod
    def from_env(cls) -> "AppConfig":
        """Construct an AppConfig instance from environment variables.

        Returns:
            AppConfig: A configuration object with defaults resolved from
            the current process environment.
        """
        return cls()

    @property
    def system_prompt(self) -> str:
        """Expose the system prompt derived from the active policy."""
        return self.shell_policy.system_prompt
     