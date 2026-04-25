"""
Data models for the ClarifyRL Environment.

This environment uses MCP (Model Context Protocol) for tool-based interactions.
The agent discovers tools via ListToolsAction and invokes them via CallToolAction.
"""

from typing import Optional

from pydantic import Field

from openenv.core.env_server.mcp_types import (
    CallToolAction,
    CallToolObservation,
    ListToolsAction,
    ListToolsObservation,
)
from openenv.core.env_server.types import Action, Observation, State


ClarifyAction = CallToolAction
ClarifyObservation = CallToolObservation


class ClarifyState(State):
    """Extended state for ClarifyRL episodes."""

    task_id: str = Field(default="", description="Current task identifier (easy/medium/hard)")
    task_title: str = Field(default="", description="Human-readable title for the current task")
    questions_asked: list[str] = Field(
        default_factory=list,
        description="Clarifying questions the agent has asked so far",
    )
    questions_remaining: int = Field(
        default=6,
        description="Number of questions the agent can still ask",
    )
    answers_received: list[str] = Field(
        default_factory=list,
        description="Answers received from the simulated user",
    )
    fields_revealed: list[str] = Field(
        default_factory=list,
        description="Profile fields that have been revealed through questions",
    )
    plan_submitted: bool = Field(default=False, description="Whether the agent has submitted a plan")
    episode_done: bool = Field(default=False, description="Whether the episode has ended")
    final_score: Optional[float] = Field(
        default=None,
        description="Final rubric score once the episode completes",
    )
    score_breakdown: Optional[dict] = Field(
        default=None,
        description="Per-component rubric breakdown",
    )


__all__ = [
    "ClarifyAction",
    "ClarifyObservation",
    "ClarifyState",
    "CallToolAction",
    "CallToolObservation",
    "ListToolsAction",
    "ListToolsObservation",
    "Action",
    "Observation",
    "State",
]
