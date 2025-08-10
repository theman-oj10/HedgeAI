"""
Investment Agents Module

Contains Warren Buffett and Cathie Wood investment agents.
"""

from .warren_buffett_agent import WarrenBuffettAgent, WarrenBuffettSignal
from .cathie_wood_agent import CathieWoodAgent, CathieWoodSignal
from .moderator_agent import ModeratorAgent, DebatePoint, ConsensusDecision

__all__ = [
    "WarrenBuffettAgent",
    "WarrenBuffettSignal", 
    "CathieWoodAgent",
    "CathieWoodSignal",
    "ModeratorAgent",
    "DebatePoint", 
    "ConsensusDecision"
]
