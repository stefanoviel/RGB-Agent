"""ArcGym agent package."""

from .claude_code_action_agent import ClaudeCodeActionAgent
from .rgb_agent import RGBAgent

AVAILABLE_AGENTS = {
    "claude_code_action_agent": ClaudeCodeActionAgent,
}

__all__ = [
    "ClaudeCodeActionAgent",
    "RGBAgent",
    "AVAILABLE_AGENTS",
]
