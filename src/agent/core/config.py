"""
Configuration module for all the components in the Personal AI Agent. 
This module defines the configuration classes and settings for various components of the agent, including the language model provider, state management, and other relevant configurations. 
It provides a centralized location for managing and updating the configurations as needed for the agent's operation.
"""

import os
from dataclasses import dataclass , field
from typing import Dict, Any , List , Optional
from pathlib import Path
from enum import Enum
from dotenv import load_dotenv

load_dotenv()

class LLMProvider(str, Enum):
    """Enumerates supported Large Language Model providers."""

    
    GROQ = "groq"
    GEMINI = "gemini"

@dataclass
class LLMConfig:
    """
    Configuration class for the Language Model Provider (LLM). 
    This class includes attributes for specifying the provider name, model name, and any additional parameters required for initializing the LLM client.
    """
    provider : LLMProvider = LLMProvider(os.getenv("LLM_Provider", "groq"))
    model : str = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
    groq_api_key : Optional[str] = os.getenv("GROQ_API_KEY")
    google_api_key : Optional[str] = os.getenv("GOOGLE_API_KEY")
    temperature : float = float(os.getenv("LLM_TEMPERATURE", "0.3"))
    max_tokens : int = int(os.getenv("LLM_MAX_TOKENS", "512"))
    

@dataclass
class AppConfig:
    
    llm: LLMConfig = field(default_factory=LLMConfig) # making it immutable by default
    @classmethod
    def from_env(cls) -> "AppConfig":
        """Construct an AppConfig instance from environment variables.

        Returns:
            AppConfig: A configuration object with defaults resolved from
            the current process environment.
        """
        return cls()
    
     