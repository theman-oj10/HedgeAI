#!/usr/bin/env python3
"""
Portfolio Debate System

Conducts investment debates for entire portfolios with weighted analysis.
"""

import sys
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from dotenv import load_dotenv

from portfolio_config import Portfolio, PortfolioHolding, SAMPLE_PORTFOLIOS, create_custom_portfolio
from consensus_debate import AgentDebateSystem
from agents.moderator_agent import ConsensusDecision

# Load environment variables
load_dotenv()

console = Console()


class PortfolioDebateResult:
    """Results from portfolio-wide debate analysis"""
    
    def __init__(self, portfolio: Portfolio):
        self.portfolio = portfolio
        self.individual_results: Dict[str, ConsensusDecision] = {}
        self.portfolio_score: float = 0.0
        self.weighted_confidence: float = 0.0
        self.overall_signal: str = "neutral"
        self.rebalancing_suggestions: List[str] = []
        self.suggested_weights: Dict[str, float] = {}
        self.rebalanced_portfolio: Portfolio = None
    
    def add_stock_result(self, ticker: str, result: ConsensusDecision):
        """Add individual stock debate result"""
        self.individual_results[ticker] = result
        self._calculate_portfolio_metrics()
    
    def _calculate_portfolio_metrics(self):
        """Calculate overall portfolio metrics from individual results"""
        if not self.individual_results:
            return
        
        total_weighted_score = 0.0
        total_weighted_confidence = 0.0
        
        for holding in self.portfolio.holdings:
            ticker = holding.ticker
            weight = holding.weight
            
            if ticker in self.individual_results:
                result = self.individual_results[ticker]
                
                # Convert signal to numeric score
                signal_score = {
                    "bullish": 1.0,
                    "neutral": 0.0,
                    "bearish": -1.0
                }.get(result.signal.lower(), 0.0)
                
                total_weighted_score += signal_score * weight
                total_weighted_confidence += result.confidence * weight
        
        self.portfolio_score = total_weighted_score
        self.weighted_confidence = total_weighted_confidence
        
        # Determine overall signal
        if self.portfolio_score > 0.2:
            self.overall_signal = "bullish"
        elif self.portfolio_score < -0.2:
            self.overall_signal = "bearish"
        else:
            self.overall_signal = "neutral"
        
        # Calculate suggested rebalancing weights
        self._calculate_rebalanced_weights()
    
    def _calculate_rebalanced_weights(self):
        """Calculate new portfolio weights based on consensus results"""
        if not self.individual_results:
            return
        
        # Calculate signal scores for each holding
        signal_scores = {}
        confidence_scores = {}
        
        for holding in self.portfolio.holdings:
            ticker = holding.ticker
            if ticker in self.individual_results:
                result = self.individual_results[ticker]
                
                # Convert signal to numeric score
                signal_score = {
                    "bullish": 1.0,
                    "neutral": 0.0,
                    "bearish": -1.0
                }.get(result.signal.lower(), 0.0)
                
                # Adjust score by confidence (higher confidence = more weight)
                adjusted_score = signal_score * (result.confidence / 100.0)
                
                signal_scores[ticker] = adjusted_score
                confidence_scores[ticker] = result.confidence / 100.0
        
        if not signal_scores:
            return
        
        # Calculate base weights using confidence-adjusted signals
        # Start with equal base allocation, then adjust based on signals
        base_weight = 0.9 / len(signal_scores)  # Reserve 10% for cash
        
        # Calculate adjustment factors
        total_positive_adjustment = 0.0
        total_negative_adjustment = 0.0
        adjustments = {}
        
        for ticker, score in signal_scores.items():
            confidence = confidence_scores[ticker]
            
            # Bullish stocks get increased weight, bearish get decreased
            if score > 0.1:  # Bullish with decent confidence
                adjustment = score * confidence * 0.3  # Max 30% boost
                adjustments[ticker] = adjustment
                total_positive_adjustment += adjustment
            elif score < -0.1:  # Bearish with decent confidence
                adjustment = score * confidence * 0.3  # Max 30% reduction
                adjustments[ticker] = adjustment
                total_negative_adjustment += abs(adjustment)
            else:  # Neutral
                adjustments[ticker] = 0.0
        
        # Apply adjustments while maintaining total weight = 0.9
        suggested_weights = {}
        weight_adjustment_pool = 0.0
        
        for ticker in signal_scores.keys():
            current_weight = base_weight
            adjustment = adjustments[ticker]
            
            if adjustment > 0:  # Increase weight for bullish stocks
                new_weight = current_weight + adjustment
            elif adjustment < 0:  # Decrease weight for bearish stocks
                reduction = abs(adjustment)
                new_weight = max(0.01, current_weight - reduction)  # Minimum 1%
                weight_adjustment_pool += (current_weight - new_weight)
            else:  # Neutral stocks
                new_weight = current_weight
            
            suggested_weights[ticker] = new_weight
        
        # Redistribute weight from bearish stocks to bullish/neutral stocks
        if weight_adjustment_pool > 0:
            bullish_neutral_tickers = [t for t, s in signal_scores.items() if s >= 0]
            if bullish_neutral_tickers:
                redistribution_per_stock = weight_adjustment_pool / len(bullish_neutral_tickers)
                for ticker in bullish_neutral_tickers:
                    suggested_weights[ticker] += redistribution_per_stock
        
        # Normalize to ensure total weight = 0.9 (leaving 10% cash)
        total_weight = sum(suggested_weights.values())
        if total_weight > 0:
            normalization_factor = 0.9 / total_weight
            for ticker in suggested_weights:
                suggested_weights[ticker] *= normalization_factor
        
        self.suggested_weights = suggested_weights
        
        # Create rebalanced portfolio object
        new_holdings = []
        for ticker, weight in suggested_weights.items():
            new_holdings.append(PortfolioHolding(ticker=ticker, weight=weight))
        
        self.rebalanced_portfolio = Portfolio(
            name=f"{self.portfolio.name} (Rebalanced)",
            holdings=new_holdings,
            cash_weight=0.10
        )


class PortfolioDebateSystem:
    """Orchestrates debates for entire portfolios"""
    
    def __init__(self):
        self.debate_system = AgentDebateSystem()
    
    def analyze_portfolio(self, portfolio: Portfolio, end_date: str = "2024-12-31", parallel: bool = True) -> PortfolioDebateResult:
        """Conduct debates for all holdings in a portfolio"""
        
        console.print(f"\n[bold blue]🎯 Portfolio Analysis: {portfolio.name}[/bold blue]")
        console.print("=" * 80)
        
        # Display portfolio composition
        self._display_portfolio_composition(portfolio)
        
        result = PortfolioDebateResult(portfolio)
        
        if parallel and len(portfolio.holdings) > 1:
            # Run portfolio analysis in parallel
            console.print(f"\n[bold cyan]🚀 Running parallel analysis for {len(portfolio.holdings)} holdings...[/bold cyan]")
            result = self._analyze_portfolio_parallel(portfolio, end_date, result)
        else:
            # Run portfolio analysis sequentially
            result = self._analyze_portfolio_sequential(portfolio, end_date, result)
        
        # Display portfolio-level results
        self._display_portfolio_results(result)
        
        return result
    
    def _analyze_portfolio_sequential(self, portfolio: Portfolio, end_date: str, result: PortfolioDebateResult) -> PortfolioDebateResult:
        """Analyze portfolio holdings sequentially (original method)"""
        
        # Conduct individual stock debates
        for i, holding in enumerate(portfolio.holdings, 1):
            console.print(f"\n[bold yellow]📊 Analyzing {holding.ticker} ({holding.weight:.1%} allocation)[/bold yellow]")
            console.print(f"[dim]Stock {i} of {len(portfolio.holdings)}[/dim]")
            
            try:
                # Prepare portfolio context for this stock
                portfolio_context = {
                    "portfolio_name": portfolio.name,
                    "total_holdings": len(portfolio.holdings),
                    "cash_allocation": portfolio.cash_weight,
                    "other_holdings": [
                        {"ticker": h.ticker, "weight": h.weight} 
                        for h in portfolio.holdings if h.ticker != holding.ticker
                    ]
                }
                
                stock_result = self.debate_system.conduct_debate(
                    holding.ticker, 
                    end_date, 
                    current_weight=holding.weight,
                    portfolio_context=portfolio_context
                )
                result.add_stock_result(holding.ticker, stock_result)
                
                console.print(f"[green]✅ {holding.ticker} analysis complete[/green]")
                
            except Exception as e:
                console.print(f"[red]❌ Error analyzing {holding.ticker}: {str(e)}[/red]")
                continue
        
        return result
    
    def _analyze_portfolio_parallel(self, portfolio: Portfolio, end_date: str, result: PortfolioDebateResult) -> PortfolioDebateResult:
        """Analyze portfolio holdings with parallel initial analysis and sequential debates"""
        
        console.print(f"[bold cyan]🚀 Phase 1: Parallel Initial Analysis for All Holdings[/bold cyan]")
        
        # Step 1: Run initial analysis for all stocks in parallel
        initial_analyses = self._run_parallel_initial_analysis(portfolio, end_date)
        
        console.print(f"\n[bold yellow]🎯 Phase 2: Sequential Debates for Each Holding[/bold yellow]")
        
        # Step 2: Run debates sequentially using the pre-computed initial analyses
        for i, holding in enumerate(portfolio.holdings, 1):
            console.print(f"\n[bold yellow]📊 Debate for {holding.ticker} ({holding.weight:.1%} allocation)[/bold yellow]")
            console.print(f"[dim]Stock {i} of {len(portfolio.holdings)}[/dim]")
            
            ticker = holding.ticker
            if ticker in initial_analyses:
                buffett_initial, wood_initial = initial_analyses[ticker]
                
                try:
                    # Prepare portfolio context for this stock
                    portfolio_context = {
                        "portfolio_name": portfolio.name,
                        "total_holdings": len(portfolio.holdings),
                        "cash_allocation": portfolio.cash_weight,
                        "other_holdings": [
                            {"ticker": h.ticker, "weight": h.weight} 
                            for h in portfolio.holdings if h.ticker != holding.ticker
                        ]
                    }
                    
                    # Run the debate using pre-computed initial analyses
                    stock_result = self.debate_system.conduct_debate_with_initial_analyses(
                        ticker, 
                        buffett_initial, 
                        wood_initial, 
                        end_date, 
                        current_weight=holding.weight,
                        portfolio_context=portfolio_context
                    )
                    
                    result.add_stock_result(ticker, stock_result)
                    console.print(f"[green]✅ {ticker} debate complete[/green]")
                    
                except Exception as e:
                    console.print(f"[red]❌ Error in debate for {ticker}: {str(e)}[/red]")
                    continue
            else:
                console.print(f"[red]❌ No initial analysis found for {ticker}[/red]")
                continue
        
        return result
    
    def _run_parallel_initial_analysis(self, portfolio: Portfolio, end_date: str) -> Dict:
        """Run initial analysis for all stocks in parallel"""
        
        def analyze_stock_initial(holding):
            """Run initial analysis for a single stock"""
            try:
                # Prepare portfolio context for this stock
                portfolio_context = {
                    "portfolio_name": portfolio.name,
                    "total_holdings": len(portfolio.holdings),
                    "cash_allocation": portfolio.cash_weight,
                    "other_holdings": [
                        {"ticker": h.ticker, "weight": h.weight} 
                        for h in portfolio.holdings if h.ticker != holding.ticker
                    ]
                }
                
                # Run both agents' initial analysis in parallel
                with ThreadPoolExecutor(max_workers=2) as executor:
                    buffett_future = executor.submit(
                        self.debate_system.warren_buffett.analyze_stock,
                        holding.ticker, end_date, holding.weight, portfolio_context
                    )
                    wood_future = executor.submit(
                        self.debate_system.cathie_wood.analyze_stock,
                        holding.ticker, end_date, holding.weight, portfolio_context
                    )
                    
                    buffett_result = buffett_future.result()
                    wood_result = wood_future.result()
                
                return holding.ticker, (buffett_result, wood_result), None
                
            except Exception as e:
                return holding.ticker, None, str(e)
        
        # Run initial analysis for all holdings in parallel
        max_workers = min(len(portfolio.holdings), 4)  # Limit concurrent analyses
        initial_analyses = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all initial analysis tasks
            future_to_holding = {
                executor.submit(analyze_stock_initial, holding): holding 
                for holding in portfolio.holdings
            }
            
            # Process results as they complete
            for future in as_completed(future_to_holding):
                holding = future_to_holding[future]
                ticker, analyses, error = future.result()
                
                if error:
                    console.print(f"[red]❌ Initial analysis failed for {ticker}: {error}[/red]")
                else:
                    initial_analyses[ticker] = analyses
                    console.print(f"[green]✅ Initial analysis complete for {ticker}[/green]")
        
        console.print(f"[bold green]🎉 Initial analysis complete for {len(initial_analyses)}/{len(portfolio.holdings)} holdings[/bold green]")
        return initial_analyses
    
    def _display_portfolio_composition(self, portfolio: Portfolio):
        """Display portfolio composition table"""
        table = Table(title=f"Portfolio Composition: {portfolio.name}")
        table.add_column("Ticker", style="cyan", no_wrap=True)
        table.add_column("Weight", style="magenta")
        table.add_column("Allocation", style="green")
        
        for holding in portfolio.holdings:
            table.add_row(
                holding.ticker,
                f"{holding.weight:.1%}",
                f"{holding.weight:.3f}"
            )
        
        if portfolio.cash_weight > 0:
            table.add_row("CASH", f"{portfolio.cash_weight:.1%}", f"{portfolio.cash_weight:.3f}")
        
        console.print(table)
    
    def _display_portfolio_results(self, result: PortfolioDebateResult):
        """Display comprehensive portfolio analysis results"""
        
        # Individual stock results table
        table = Table(title="Individual Stock Analysis Results")
        table.add_column("Ticker", style="cyan", no_wrap=True)
        table.add_column("Weight", style="blue")
        table.add_column("Signal", style="bold")
        table.add_column("Confidence", style="magenta")
        table.add_column("Buffett Weight", style="yellow")
        table.add_column("Wood Weight", style="green")
        
        for holding in result.portfolio.holdings:
            ticker = holding.ticker
            if ticker in result.individual_results:
                stock_result = result.individual_results[ticker]
                
                # Color code the signal
                signal_color = {
                    "bullish": "green",
                    "bearish": "red",
                    "neutral": "yellow"
                }.get(stock_result.signal.lower(), "white")
                
                table.add_row(
                    ticker,
                    f"{holding.weight:.1%}",
                    f"[{signal_color}]{stock_result.signal.upper()}[/{signal_color}]",
                    f"{stock_result.confidence:.1f}%",
                    f"{stock_result.buffett_weight:.1f}%",
                    f"{stock_result.wood_weight:.1f}%"
                )
        
        console.print(table)
        
        # Portfolio-level summary
        overall_color = {
            "bullish": "green",
            "bearish": "red",
            "neutral": "yellow"
        }.get(result.overall_signal, "white")
        
        summary_panel = Panel(
            f"""[bold]Portfolio-Level Analysis Summary[/bold]

[bold]Overall Signal:[/bold] [{overall_color}]{result.overall_signal.upper()}[/{overall_color}]
[bold]Portfolio Score:[/bold] {result.portfolio_score:.3f} (Range: -1.0 to +1.0)
[bold]Weighted Confidence:[/bold] {result.weighted_confidence:.1f}%

[bold]Interpretation:[/bold]
• Portfolio Score > 0.2: Bullish outlook
• Portfolio Score < -0.2: Bearish outlook  
• Portfolio Score -0.2 to 0.2: Neutral outlook

[bold]Portfolio Composition:[/bold] {len(result.portfolio.holdings)} holdings
[bold]Cash Allocation:[/bold] {result.portfolio.cash_weight:.1%}""",
            title="🎯 Portfolio Investment Thesis",
            border_style=overall_color
        )
        
        console.print(summary_panel)
        
        # Display rebalanced portfolio suggestions
        self._display_rebalanced_portfolio(result)
    
    def _display_rebalanced_portfolio(self, result: PortfolioDebateResult):
        """Display suggested portfolio rebalancing based on consensus results"""
        
        if not result.suggested_weights:
            return
        
        console.print("\n[bold blue]🔄 Consensus-Based Portfolio Rebalancing[/bold blue]")
        console.print("=" * 80)
        
        # Comparison table: Current vs Suggested weights
        comparison_table = Table(title="Portfolio Rebalancing Recommendations")
        comparison_table.add_column("Ticker", style="cyan", no_wrap=True)
        comparison_table.add_column("Current Weight", style="blue")
        comparison_table.add_column("Suggested Weight", style="green")
        comparison_table.add_column("Change", style="bold")
        comparison_table.add_column("Signal", style="magenta")
        comparison_table.add_column("Confidence", style="yellow")
        comparison_table.add_column("Rationale", style="dim")
        
        for holding in result.portfolio.holdings:
            ticker = holding.ticker
            current_weight = holding.weight
            suggested_weight = result.suggested_weights.get(ticker, current_weight)
            
            # Calculate change
            weight_change = suggested_weight - current_weight
            change_pct = (weight_change / current_weight * 100) if current_weight > 0 else 0
            
            # Format change with color
            if abs(change_pct) < 1:
                change_str = f"[dim]{change_pct:+.1f}%[/dim]"
                change_color = "dim"
            elif change_pct > 0:
                change_str = f"[green]+{change_pct:.1f}%[/green]"
                change_color = "green"
            else:
                change_str = f"[red]{change_pct:.1f}%[/red]"
                change_color = "red"
            
            # Get signal and confidence
            if ticker in result.individual_results:
                stock_result = result.individual_results[ticker]
                signal = stock_result.signal.upper()
                confidence = f"{stock_result.confidence:.0f}%"
                
                # Rationale for weight change
                if change_pct > 5:
                    rationale = "Increase (bullish signal)"
                elif change_pct < -5:
                    rationale = "Decrease (bearish signal)"
                else:
                    rationale = "Maintain (neutral signal)"
            else:
                signal = "N/A"
                confidence = "N/A"
                rationale = "No analysis"
            
            comparison_table.add_row(
                ticker,
                f"{current_weight:.1%}",
                f"{suggested_weight:.1%}",
                change_str,
                signal,
                confidence,
                rationale
            )
        
        # Add cash row
        comparison_table.add_row(
            "CASH",
            f"{result.portfolio.cash_weight:.1%}",
            "10.0%",
            "[dim]0.0%[/dim]",
            "N/A",
            "N/A",
            "Fixed allocation"
        )
        
        console.print(comparison_table)
        
        # Rebalancing summary
        total_changes = sum(abs(result.suggested_weights.get(h.ticker, h.weight) - h.weight) 
                          for h in result.portfolio.holdings)
        
        rebalancing_summary = Panel(
            f"""[bold]Rebalancing Summary[/bold]

[bold]Total Portfolio Changes:[/bold] {total_changes:.1%}
[bold]Rebalancing Threshold:[/bold] {result.portfolio.rebalance_threshold:.1%}

[bold]Recommendation:[/bold] {'[green]REBALANCE RECOMMENDED[/green]' if total_changes > result.portfolio.rebalance_threshold else '[yellow]MINOR ADJUSTMENTS[/yellow]'}

[bold]Rebalancing Logic:[/bold]
• Bullish signals → Increase allocation (up to +30%)
• Bearish signals → Decrease allocation (down to 1% minimum)
• Neutral signals → Maintain current allocation
• Adjustments weighted by confidence levels
• Cash allocation maintained at 10%

[bold]Next Steps:[/bold]
1. Review individual stock analysis reasoning
2. Consider market conditions and timing
3. Implement changes gradually if desired
4. Monitor performance post-rebalancing""",
            title="📊 Portfolio Rebalancing Guidance",
            border_style="blue"
        )
        
        console.print(rebalancing_summary)


def main():
    """Main CLI interface for portfolio debate analysis"""
    
    if len(sys.argv) < 2:
        console.print("[bold red]Usage Options:[/bold red]")
        console.print("1. Use predefined portfolio:")
        console.print("   python src/portfolio_debate.py <portfolio_name>")
        console.print("   Available portfolios: tech_growth, value_dividend, balanced_mix, ai_innovation")
        console.print("\n2. Create custom portfolio:")
        console.print("   python src/portfolio_debate.py custom AAPL:0.3 TSLA:0.4 NVDA:0.3")
        console.print("\nExamples:")
        console.print("   python src/portfolio_debate.py tech_growth")
        console.print("   python src/portfolio_debate.py custom AAPL:0.5 MSFT:0.3 GOOGL:0.2")
        return
    
    portfolio_system = PortfolioDebateSystem()
    
    portfolio_name = sys.argv[1].lower()
    
    if portfolio_name == "custom":
        # Parse custom portfolio from command line
        if len(sys.argv) < 3:
            console.print("[red]Error: Custom portfolio requires ticker:weight pairs[/red]")
            console.print("Example: python src/portfolio_debate.py custom AAPL:0.4 TSLA:0.6")
            return
        
        try:
            holdings_dict = {}
            for arg in sys.argv[2:]:
                ticker, weight_str = arg.split(':')
                holdings_dict[ticker.upper()] = float(weight_str)
            
            # Validate weights sum to 1.0
            total_weight = sum(holdings_dict.values())
            if abs(total_weight - 1.0) > 0.001:
                console.print(f"[red]Error: Weights must sum to 1.0, got {total_weight}[/red]")
                return
            
            portfolio = create_custom_portfolio("Custom Portfolio", holdings_dict)
            
        except ValueError as e:
            console.print(f"[red]Error parsing custom portfolio: {e}[/red]")
            console.print("Format: TICKER:WEIGHT (e.g., AAPL:0.3)")
            return
    
    elif portfolio_name in SAMPLE_PORTFOLIOS:
        portfolio = SAMPLE_PORTFOLIOS[portfolio_name]
    
    else:
        console.print(f"[red]Unknown portfolio: {portfolio_name}[/red]")
        console.print(f"Available portfolios: {', '.join(SAMPLE_PORTFOLIOS.keys())}")
        return
    
    try:
        # Conduct portfolio analysis
        result = portfolio_system.analyze_portfolio(portfolio)
        
        console.print(f"\n[bold green]✅ Portfolio analysis complete for {portfolio.name}![/bold green]")
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Analysis interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"[red]Error during portfolio analysis: {e}[/red]")


if __name__ == "__main__":
    main()
