#!/usr/bin/env python3
"""
Investment Debate System

Conducts moderated debates between Warren Buffett and Cathie Wood investment agents.
"""

import sys
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate

from agents.warren_buffett_agent import WarrenBuffettAgent, WarrenBuffettSignal
from agents.cathie_wood_agent import CathieWoodAgent, CathieWoodSignal
from agents.moderator_agent import ModeratorAgent, DebatePoint, ConsensusDecision
from utils import call_llm


@dataclass
class DebateResult:
    """Result of a complete debate between agents"""
    ticker: str
    buffett_initial: WarrenBuffettSignal
    wood_initial: CathieWoodSignal
    buffett_response: Optional[Any] = None
    wood_response: Optional[Any] = None
    buffett_challenge: Optional[Any] = None
    wood_challenge: Optional[Any] = None
    final_decision: Optional[ConsensusDecision] = None

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
    
    def conduct_debate_with_initial_analyses(self, ticker: str, buffett_initial, wood_initial, end_date: str = "2024-12-31", current_weight: float = None, portfolio_context: dict = None) -> ConsensusDecision:
        """Conduct debate using pre-computed initial analyses (for parallel optimization)"""
        
        console = Console()
        console.print(f"\n[bold blue]🎯 Investment Debate for {ticker}[/bold blue]")
        console.print("=" * 60)
        
        # Skip Phase 1 since we have pre-computed initial analyses
        console.print("\n[bold yellow]📋 Phase 1: Using Pre-computed Initial Analyses[/bold yellow]")
        
        self._display_initial_presentations(ticker, buffett_initial, wood_initial)
        
        # Phase 2: Cross-examination
        console.print("\n[bold yellow]🔍 Phase 2: Cross-Examination[/bold yellow]")
        
        # Buffett challenges Wood
        buffett_challenge = self.moderator.generate_challenge_question(
            ticker, buffett_initial, wood_initial, "buffett_to_wood"
        )
        console.print(f"\n[cyan]Warren Buffett's Challenge:[/cyan]")
        console.print(f"[dim]{buffett_challenge.question}[/dim]")
        
        wood_response = self.cathie_wood.respond_to_debate(
            ticker, wood_initial, buffett_initial, buffett_challenge.question
        )
        console.print(f"\n[green]Cathie Wood's Response:[/green]")
        console.print(f"[dim]{wood_response.response}[/dim]")
        
        # Wood challenges Buffett
        wood_challenge = self.moderator.generate_challenge_question(
            ticker, wood_initial, buffett_initial, "wood_to_buffett"
        )
        console.print(f"\n[green]Cathie Wood's Challenge:[/green]")
        console.print(f"[dim]{wood_challenge.question}[/dim]")
        
        buffett_response = self.warren_buffett.respond_to_debate(
            ticker, buffett_initial, wood_initial, wood_challenge.question
        )
        console.print(f"\n[cyan]Warren Buffett's Response:[/cyan]")
        console.print(f"[dim]{buffett_response.response}[/dim]")
        
        # Phase 3: Final consensus
        console.print("\n[bold yellow]⚖️ Phase 3: Moderator's Final Decision[/bold yellow]")
        
        final_decision = self.moderator.make_final_decision(
            ticker,
            buffett_initial, wood_initial,
            buffett_response, wood_response,
            buffett_challenge, wood_challenge
        )
        
        self._display_consensus(ticker, final_decision)
        
        # Create result object (for internal tracking if needed)
        result = DebateResult(
            ticker=ticker,
            buffett_initial=buffett_initial,
            wood_initial=wood_initial,
            buffett_response=buffett_response,
            wood_response=wood_response,
            buffett_challenge=buffett_challenge,
            wood_challenge=wood_challenge,
            final_decision=final_decision
        )
        
        # Return the ConsensusDecision which has the signal attribute
        return final_decision
    
    def conduct_batch_debate(self, tickers: List[str], end_date: str = "2024-12-31") -> Dict[str, 'DebateResult']:
        """Conduct batch debates for multiple tickers"""
        
        results = {}
        
        for ticker in tickers:
            buffett_initial = self.warren_buffett.analyze_stock(ticker, end_date)
            wood_initial = self.cathie_wood.analyze_stock(ticker, end_date)
            
            result = self.conduct_debate_with_initial_analyses(ticker, buffett_initial, wood_initial, end_date)
            
            results[ticker] = result
        
        return results
    
    def conduct_debate(self, ticker: str, end_date: str = "2024-12-31", current_weight: float = None, portfolio_context: dict = None) -> ConsensusDecision:
        """Conduct a full moderated debate and reach consensus"""
        
        console.print(f"\n[bold blue]🎯 Investment Debate for {ticker}[/bold blue]")
        console.print("=" * 60)
        
        # Phase 1: Initial presentations from both agents (run in parallel)
        console.print("\n[bold yellow]📋 Phase 1: Initial Agent Presentations[/bold yellow]")
        console.print(f"[dim]Running parallel analysis for {ticker}...[/dim]")
        
        # Run both agent analyses in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both analysis tasks
            buffett_future = executor.submit(
                self.warren_buffett.analyze_stock, 
                ticker, end_date, current_weight, portfolio_context
            )
            wood_future = executor.submit(
                self.cathie_wood.analyze_stock, 
                ticker, end_date, current_weight, portfolio_context
            )
            
            # Collect results as they complete
            futures = {buffett_future: "Warren Buffett", wood_future: "Cathie Wood"}
            results = {}
            
            for future in as_completed(futures):
                agent_name = futures[future]
                try:
                    result = future.result()
                    results[agent_name] = result
                    console.print(f"[green]✅ {agent_name} analysis complete[/green]")
                except Exception as e:
                    console.print(f"[red]❌ {agent_name} analysis failed: {str(e)}[/red]")
                    raise e
        
        # Extract results
        buffett_initial = results["Warren Buffett"]
        wood_initial = results["Cathie Wood"]
        
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
        """Display initial agent presentations"""
        
        # Warren Buffett's analysis
        buffett_content = (
            f"[bold]Signal:[/bold] {buffett_result.signal.upper()}\n"
            f"[bold]Confidence:[/bold] {buffett_result.confidence:.1f}%\n"
        )
        
        if buffett_result.suggested_weight is not None:
            buffett_content += f"[bold]Suggested Weight:[/bold] {buffett_result.suggested_weight:.1%}\n"
            if buffett_result.weight_reasoning:
                buffett_content += f"[bold]Weight Reasoning:[/bold] {buffett_result.weight_reasoning}\n"
        
        buffett_content += f"\n[bold]Analysis:[/bold]\n{buffett_result.reasoning}"
        
        buffett_panel = Panel(
            buffett_content,
            title=f"💼 Warren Buffett's Analysis of {ticker}",
            border_style="blue"
        )
        console.print(buffett_panel)
        
        # Cathie Wood's analysis
        wood_content = (
            f"[bold]Signal:[/bold] {wood_result.signal.upper()}\n"
            f"[bold]Confidence:[/bold] {wood_result.confidence:.1f}%\n"
        )
        
        if wood_result.suggested_weight is not None:
            wood_content += f"[bold]Suggested Weight:[/bold] {wood_result.suggested_weight:.1%}\n"
            if wood_result.weight_reasoning:
                wood_content += f"[bold]Weight Reasoning:[/bold] {wood_result.weight_reasoning}\n"
        
        wood_content += f"\n[bold]Analysis:[/bold]\n{wood_result.reasoning}"
        
        wood_panel = Panel(
            wood_content,
            title=f"🚀 Cathie Wood's Analysis of {ticker}",
            border_style="green"
        )
        console.print(wood_panel)
    
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
