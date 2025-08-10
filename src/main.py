"""
Main application for running Warren Buffett and Cathie Wood investment agents
"""
import os
import sys
from typing import List
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from dotenv import load_dotenv

from agents.warren_buffett_agent import WarrenBuffettAgent
from agents.cathie_wood_agent import CathieWoodAgent
from utils import print_analysis_result

# Load environment variables
load_dotenv()

console = Console()


class InvestmentAnalyzer:
    """Main class for running investment analysis with both agents"""
    
    def __init__(self): # add all the agents here
        self.warren_buffett = WarrenBuffettAgent()
        self.cathie_wood = CathieWoodAgent()
    
    def analyze_single_stock(self, ticker: str, end_date: str = "2024-12-31"):
        """Analyze a single stock with both agents"""
        console.print(f"\n[bold blue]Analyzing {ticker}[/bold blue]")
        console.print("=" * 50)
        
        # Warren Buffett Analysis
        console.print("\n[bold green]🏛️  Warren Buffett Analysis[/bold green]")
        try:
            buffett_result = self.warren_buffett.analyze_stock(ticker, end_date)
            print_analysis_result(ticker, "Warren Buffett", {
                "signal": buffett_result.signal,
                "confidence": buffett_result.confidence,
                "reasoning": buffett_result.reasoning
            })
        except Exception as e:
            console.print(f"[red]Error in Buffett analysis: {e}[/red]")
        
        # Cathie Wood Analysis
        console.print("\n[bold purple]🚀 Cathie Wood Analysis[/bold purple]")
        try:
            wood_result = self.cathie_wood.analyze_stock(ticker, end_date)
            print_analysis_result(ticker, "Cathie Wood", {
                "signal": wood_result.signal,
                "confidence": wood_result.confidence,
                "reasoning": wood_result.reasoning
            })
        except Exception as e:
            console.print(f"[red]Error in Wood analysis: {e}[/red]")
    
    def analyze_multiple_stocks(self, tickers: List[str], end_date: str = "2024-12-31"):
        """Analyze multiple stocks and show comparison table"""
        results = {}
        
        for ticker in tickers:
            console.print(f"\n[bold cyan]Processing {ticker}...[/bold cyan]")
            
            try:
                buffett_result = self.warren_buffett.analyze_stock(ticker, end_date)
                wood_result = self.cathie_wood.analyze_stock(ticker, end_date)
                
                results[ticker] = {
                    "buffett": {
                        "signal": buffett_result.signal,
                        "confidence": buffett_result.confidence,
                        "reasoning": buffett_result.reasoning[:100] + "..." if len(buffett_result.reasoning) > 100 else buffett_result.reasoning
                    },
                    "wood": {
                        "signal": wood_result.signal,
                        "confidence": wood_result.confidence,
                        "reasoning": wood_result.reasoning[:100] + "..." if len(wood_result.reasoning) > 100 else wood_result.reasoning
                    }
                }
            except Exception as e:
                console.print(f"[red]Error analyzing {ticker}: {e}[/red]")
                results[ticker] = {
                    "buffett": {"signal": "error", "confidence": 0, "reasoning": "Analysis failed"},
                    "wood": {"signal": "error", "confidence": 0, "reasoning": "Analysis failed"}
                }
        
        # Display comparison table
        self._display_comparison_table(results)
    
    def _display_comparison_table(self, results: dict):
        """Display results in a comparison table"""
        table = Table(title="Investment Agent Comparison")
        
        table.add_column("Ticker", style="cyan", no_wrap=True)
        table.add_column("Warren Buffett", style="green")
        table.add_column("Cathie Wood", style="purple")
        
        for ticker, data in results.items():
            buffett_data = data["buffett"]
            wood_data = data["wood"]
            
            # Format signals with colors
            buffett_signal = self._format_signal(buffett_data["signal"], buffett_data["confidence"])
            wood_signal = self._format_signal(wood_data["signal"], wood_data["confidence"])
            
            table.add_row(ticker, buffett_signal, wood_signal)
        
        console.print("\n")
        console.print(table)
    
    def _format_signal(self, signal: str, confidence: float) -> str:
        """Format signal with appropriate colors"""
        color_map = {
            "bullish": "green",
            "bearish": "red",
            "neutral": "yellow",
            "error": "dim"
        }
        color = color_map.get(signal, "white")
        return f"[{color}]{signal.upper()}[/{color}] ({confidence:.1f}%)"


def main():
    """Main CLI interface"""
    console.print(Panel.fit(
        "[bold blue]Investment Agents[/bold blue]\n"
        "[dim]Warren Buffett & Cathie Wood AI Analysis[/dim]",
        border_style="blue"
    ))
    
    # Check for required environment variables
    required_vars = ["OPENAI_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var) or os.getenv(var).startswith("your_")]
    
    if missing_vars:
        console.print(f"[red]Missing required environment variables: {', '.join(missing_vars)}[/red]")
        console.print("[yellow]Please set up your .env file based on .env.example[/yellow]")
        return
    
    analyzer = InvestmentAnalyzer()
    
    if len(sys.argv) < 2:
        console.print("[yellow]Usage: python -m src.main <TICKER1> [TICKER2] ...[/yellow]")
        console.print("[yellow]Example: python -m src.main AAPL TSLA NVDA[/yellow]")
        return
    
    tickers = [ticker.upper() for ticker in sys.argv[1:]]
    
    if len(tickers) == 1:
        analyzer.analyze_single_stock(tickers[0])
    else:
        analyzer.analyze_multiple_stocks(tickers)


if __name__ == "__main__":
    main()
