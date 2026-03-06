"""
This module defines the State class, which represents the state of an agent in a personal AI system. 
The State class includes attributes for storing the agent's name, goals, and other relevant information. 
It also provides methods for updating and retrieving the state information as needed by the agent's decision-making processes.
"""

from typing import List, Dict, Any , Optional
from pydantic import BaseModel , Field

class BaseAgentState(BaseModel):
    """
    Base class for agent state , providing common attributes and methods for all agent states.
    """
    session_id: str = Field(..., description="Unique identifier for the agent's session.")

    metadata: Optional[Dict[str, Any]] = Field(default = None, description="Additional metadata related to the agent's state.")
    