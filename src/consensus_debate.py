"""
Agent Consensus Debate System - Moderated Discussion Between Investment Agents
"""
import os
import sys
from typing import List, Dict, Any
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate

from agents.warren_buffett_agent import WarrenBuffettAgent, WarrenBuffettSignal
from agents.cathie_wood_agent import CathieWoodAgent, CathieWoodSignal
from agents.moderator_agent import ModeratorAgent, DebatePoint, ConsensusDecision
from utils import call_llm

# Load environment variables
load_dotenv()

console = Console()


class DebateResponse(BaseModel):
    response: str
    updated_signal: str
    updated_confidence: float


class AgentDebateSystem:
    """Orchestrates moderated debates between investment agents"""
    
    def __init__(self):
        self.warren_buffett = WarrenBuffettAgent()
        self.cathie_wood = CathieWoodAgent()
        self.moderator = ModeratorAgent()
        self.debate_history = []
    
    def conduct_debate(self, ticker: str, end_date: str = "2024-12-31") -> ConsensusDecision:
        """Conduct a full moderated debate and reach consensus"""
        
        console.print(f"\n[bold blue]🎯 Investment Debate for {ticker}[/bold blue]")
        console.print("=" * 60)
        
        # Phase 1: Initial Presentations
        console.print("\n[bold yellow]📋 Phase 1: Initial Presentations[/bold yellow]")
        
        buffett_initial = self.warren_buffett.analyze_stock(ticker, end_date)
        wood_initial = self.cathie_wood.analyze_stock(ticker, end_date)
        
        self._display_initial_presentations(ticker, buffett_initial, wood_initial)
        
        # Phase 2: Moderator Identifies Disagreements
        console.print("\n[bold yellow]🔍 Phase 2: Identifying Key Disagreements[/bold yellow]")
        
        debate_point = self.moderator.identify_disagreements(
            ticker,
            {
                "signal": buffett_initial.signal,
                "confidence": buffett_initial.confidence,
                "reasoning": buffett_initial.reasoning
            },
            {
                "signal": wood_initial.signal,
                "confidence": wood_initial.confidence,
                "reasoning": wood_initial.reasoning
            }
        )
        
        self._display_disagreements(debate_point)
        
        # Phase 3: Cross-Examination Round
        console.print("\n[bold yellow]⚔️  Phase 3: Cross-Examination[/bold yellow]")
        
        # Buffett responds to Wood's challenge
        buffett_rebuttal = self.warren_buffett.respond_to_debate(
            ticker,
            buffett_initial,
            wood_initial,
            debate_point.question_for_buffett
        )
        
        # Wood responds to Buffett's challenge
        wood_rebuttal = self.cathie_wood.respond_to_debate(
            ticker,
            wood_initial,
            buffett_initial,
            debate_point.question_for_wood
        )
        
        self._display_rebuttals(buffett_rebuttal, wood_rebuttal)
        
        # Phase 4: Final Consensus
        console.print("\n[bold yellow]🤝 Phase 4: Moderator Synthesis[/bold yellow]")
        
        # Compile debate history
        self.debate_history = [
            {"phase": "initial", "agent": "warren_buffett", "content": buffett_initial.model_dump()},
            {"phase": "initial", "agent": "cathie_wood", "content": wood_initial.model_dump()},
            {"phase": "disagreements", "moderator": debate_point.model_dump()},
            {"phase": "rebuttal", "agent": "warren_buffett", "content": buffett_rebuttal.model_dump()},
            {"phase": "rebuttal", "agent": "cathie_wood", "content": wood_rebuttal.model_dump()}
        ]
        
        # Generate consensus
        consensus = self.moderator.synthesize_consensus(ticker, self.debate_history)
        
        self._display_consensus(ticker, consensus)
        
        return consensus
    
    def _display_initial_presentations(self, ticker: str, buffett_result: WarrenBuffettSignal, wood_result: CathieWoodSignal):
        """Display initial presentations from both agents"""
        
        # Warren Buffett's presentation
        buffett_content = Text()
        buffett_content.append(f"Signal: ", style="bold")
        buffett_content.append(f"{buffett_result.signal.upper()}", style=f"bold {'green' if buffett_result.signal == 'bullish' else 'red' if buffett_result.signal == 'bearish' else 'yellow'}")
        buffett_content.append(f"\nConfidence: {buffett_result.confidence:.1f}%\n\n", style="bold")
        buffett_content.append(f"{buffett_result.reasoning}")
        
        console.print(Panel(
            buffett_content,
            title="[bold blue]🏛️  Warren Buffett's Opening Statement[/bold blue]",
            border_style="blue"
        ))
        
        # Cathie Wood's presentation
        wood_content = Text()
        wood_content.append(f"Signal: ", style="bold")
        wood_content.append(f"{wood_result.signal.upper()}", style=f"bold {'green' if wood_result.signal == 'bullish' else 'red' if wood_result.signal == 'bearish' else 'yellow'}")
        wood_content.append(f"\nConfidence: {wood_result.confidence:.1f}%\n\n", style="bold")
        wood_content.append(f"{wood_result.reasoning}")
        
        console.print(Panel(
            wood_content,
            title="[bold purple]🚀 Cathie Wood's Opening Statement[/bold purple]",
            border_style="purple"
        ))
    
    def _display_disagreements(self, debate_point: DebatePoint):
        """Display the moderator's analysis of disagreements"""
        
        content = Text()
        content.append("Key Disagreement:\n", style="bold red")
        content.append(f"{debate_point.key_disagreement}\n\n")
        content.append("Buffett's Position:\n", style="bold blue")
        content.append(f"{debate_point.buffett_position}\n\n")
        content.append("Wood's Position:\n", style="bold purple")
        content.append(f"{debate_point.wood_position}\n\n")
        content.append("Questions for Cross-Examination:\n", style="bold yellow")
        content.append(f"To Buffett: {debate_point.question_for_buffett}\n")
        content.append(f"To Wood: {debate_point.question_for_wood}")
        
        console.print(Panel(
            content,
            title="[bold yellow]⚖️  Moderator's Analysis[/bold yellow]",
            border_style="yellow"
        ))
    

    
    def _display_rebuttals(self, buffett_rebuttal: DebateResponse, wood_rebuttal: DebateResponse):
        """Display the rebuttal responses from both agents"""
        
        # Buffett's rebuttal
        buffett_content = Text()
        buffett_content.append(f"Updated Signal: ", style="bold")
        buffett_content.append(f"{buffett_rebuttal.updated_signal.upper()}", style=f"bold {'green' if buffett_rebuttal.updated_signal == 'bullish' else 'red' if buffett_rebuttal.updated_signal == 'bearish' else 'yellow'}")
        buffett_content.append(f"\nUpdated Confidence: {buffett_rebuttal.updated_confidence:.1f}%\n\n", style="bold")
        buffett_content.append(f"{buffett_rebuttal.response}")
        
        console.print(Panel(
            buffett_content,
            title="[bold blue]🏛️  Warren Buffett's Rebuttal[/bold blue]",
            border_style="blue"
        ))
        
        # Cathie Wood's rebuttal
        wood_content = Text()
        wood_content.append(f"Updated Signal: ", style="bold")
        wood_content.append(f"{wood_rebuttal.updated_signal.upper()}", style=f"bold {'green' if wood_rebuttal.updated_signal == 'bullish' else 'red' if wood_rebuttal.updated_signal == 'bearish' else 'yellow'}")
        wood_content.append(f"\nUpdated Confidence: {wood_rebuttal.updated_confidence:.1f}%\n\n", style="bold")
        wood_content.append(f"{wood_rebuttal.response}")
        
        console.print(Panel(
            wood_content,
            title="[bold purple]🚀 Cathie Wood's Rebuttal[/bold purple]",
            border_style="purple"
        ))
    
    def _display_consensus(self, ticker: str, consensus: ConsensusDecision):
        """Display the final consensus decision"""
        
        content = Text()
        content.append(f"Consensus Signal: ", style="bold")
        content.append(f"{consensus.signal.upper()}", style=f"bold {'green' if consensus.signal == 'bullish' else 'red' if consensus.signal == 'bearish' else 'yellow'}")
        content.append(f"\nConsensus Confidence: {consensus.confidence:.1f}%\n", style="bold")
        content.append(f"Buffett Influence: {consensus.buffett_weight:.1%} | ", style="bold blue")
        content.append(f"Wood Influence: {consensus.wood_weight:.1%}\n\n", style="bold purple")
        content.append("Consensus Reasoning:\n", style="bold")
        content.append(f"{consensus.reasoning}")
        
        console.print(Panel(
            content,
            title=f"[bold magenta]🤝 Final Consensus for {ticker}[/bold magenta]",
            border_style="magenta"
        ))


def main():
    """Main CLI interface for the debate system"""
    console.print(Panel.fit(
        "[bold blue]Investment Agent Debate System[/bold blue]\n"
        "[dim]Warren Buffett vs Cathie Wood - Moderated Consensus[/dim]",
        border_style="blue"
    ))
    
    # Check for required environment variables
    required_vars = ["OPENAI_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var) or os.getenv(var).startswith("your_")]
    
    if missing_vars:
        console.print(f"[red]Missing required environment variables: {', '.join(missing_vars)}[/red]")
        console.print("[yellow]Please set up your .env file with your OpenAI API key[/yellow]")
        return
    
    if len(sys.argv) < 2:
        console.print("[yellow]Usage: python consensus_debate.py <TICKER1> [TICKER2] ...[/yellow]")
        console.print("[yellow]Example: python consensus_debate.py AAPL[/yellow]")
        console.print("[yellow]Example: python consensus_debate.py AAPL TSLA NVDA[/yellow]")
        return
    
    debate_system = AgentDebateSystem()
    tickers = [ticker.upper() for ticker in sys.argv[1:]]
    
    if len(tickers) == 1:
        # Single stock debate
        consensus = debate_system.conduct_debate(tickers[0])
        
    else:
        # Multiple stock debates with summary table
        consensus_results = {}
        
        for ticker in tickers:
            console.print(f"\n[bold magenta]{'='*20} DEBATE FOR {ticker} {'='*20}[/bold magenta]")
            consensus = debate_system.conduct_debate(ticker)
            consensus_results[ticker] = consensus
        
        # Display summary table
        console.print(f"\n[bold blue]📊 Consensus Summary[/bold blue]")
        table = Table(title="Investment Debate Results")
        
        table.add_column("Ticker", style="cyan", no_wrap=True)
        table.add_column("Consensus Signal", style="bold")
        table.add_column("Confidence", style="bold")
        table.add_column("Buffett Weight", style="blue")
        table.add_column("Wood Weight", style="purple")
        
        for ticker, consensus in consensus_results.items():
            signal_color = {"bullish": "green", "bearish": "red", "neutral": "yellow"}.get(consensus.signal, "white")
            
            table.add_row(
                ticker,
                f"[{signal_color}]{consensus.signal.upper()}[/{signal_color}]",
                f"{consensus.confidence:.1f}%",
                f"{consensus.buffett_weight:.1%}",
                f"{consensus.wood_weight:.1%}"
            )
        
        console.print(table)


if __name__ == "__main__":
    main()
