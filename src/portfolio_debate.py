#!/usr/bin/env python3
"""
Portfolio Debate System

Conducts investment debates for entire portfolios with weighted analysis.
"""

import sys
import json
from typing import Dict, List, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from dotenv import load_dotenv

from portfolio_config import Portfolio, PortfolioHolding, SAMPLE_PORTFOLIOS, create_custom_portfolio
from consensus_debate import AgentDebateSystem
from agents.moderator_agent import ConsensusDecision
from quarterly_data_extractor import QuarterlyDataExtractor, QuarterlyPortfolioContext

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
        self.timestamp = datetime.now()  # Add timestamp for JSON output
    
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
        self.quarterly_extractor = QuarterlyDataExtractor()
    
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
        
        # Generate and display JSON trading actions
        json_output = self.generate_trading_actions_json(result)
        console.print(f"\n[bold cyan]📄 JSON Trading Actions Output:[/bold cyan]")
        console.print(json_output)
        
        return result, json_output

    def analyze_portfolio_quarterly(self, portfolio: Portfolio, reference_date: str = None, parallel: bool = True) -> Tuple[PortfolioDebateResult, str]:
        """
        Conduct quarterly portfolio debates with historical data context
        
        Args:
            portfolio: Portfolio to analyze
            reference_date: Reference date for quarter calculation (defaults to today)
            parallel: Whether to run analysis in parallel
            
        Returns:
            Tuple of (PortfolioDebateResult, json_output)
        """
        console.print(f"\n[bold blue]📊 QUARTERLY Portfolio Analysis: {portfolio.name}[/bold blue]")
        console.print("=" * 80)
        
        # Extract quarterly context
        console.print("[bold cyan]📈 Extracting quarterly performance data...[/bold cyan]")
        tickers = [holding.ticker for holding in portfolio.holdings]
        quarterly_context = self.quarterly_extractor.extract_quarterly_context(tickers, reference_date)
        
        # Display quarterly summary
        quarterly_summary = self.quarterly_extractor.format_quarterly_summary(quarterly_context)
        console.print(Panel(
            quarterly_summary,
            title="📊 Quarterly Performance Context",
            border_style="cyan"
        ))
        
        # Display current portfolio composition
        self._display_portfolio_composition(portfolio)
        
        result = PortfolioDebateResult(portfolio)
        
        # Add quarterly context to the debate system
        if parallel and len(portfolio.holdings) > 1:
            console.print(f"\n[bold cyan]🚀 Running quarterly-informed parallel analysis for {len(portfolio.holdings)} holdings...[/bold cyan]")
            result = self._analyze_portfolio_parallel_with_quarterly_data(portfolio, quarterly_context, result)
        else:
            console.print(f"\n[bold cyan]🔄 Running quarterly-informed sequential analysis for {len(portfolio.holdings)} holdings...[/bold cyan]")
            result = self._analyze_portfolio_sequential_with_quarterly_data(portfolio, quarterly_context, result)
        
        # Display portfolio-level results with quarterly context
        self._display_quarterly_portfolio_results(result, quarterly_context)
        
        # Generate and display JSON trading actions
        json_output = self.generate_trading_actions_json(result)
        console.print(f"\n[bold cyan]📄 JSON Trading Actions Output:[/bold cyan]")
        console.print(json_output)
        
        return result, json_output
    
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

    def _analyze_portfolio_sequential_with_quarterly_data(self, portfolio: Portfolio, quarterly_context: QuarterlyPortfolioContext, result: PortfolioDebateResult) -> PortfolioDebateResult:
        """Analyze portfolio holdings sequentially with quarterly data context"""
        
        # Conduct individual stock debates with quarterly context
        for i, holding in enumerate(portfolio.holdings, 1):
            console.print(f"\n[bold yellow]📊 Analyzing {holding.ticker} ({holding.weight:.1%} allocation) with Q data[/bold yellow]")
            console.print(f"[dim]Stock {i} of {len(portfolio.holdings)}[/dim]")
            
            try:
                # Get quarterly performance for this stock
                quarterly_perf = quarterly_context.individual_performance.get(holding.ticker)
                
                # Prepare enhanced portfolio context with quarterly data
                portfolio_context = {
                    "portfolio_name": portfolio.name,
                    "total_holdings": len(portfolio.holdings),
                    "cash_allocation": portfolio.cash_weight,
                    "other_holdings": [
                        {"ticker": h.ticker, "weight": h.weight} 
                        for h in portfolio.holdings if h.ticker != holding.ticker
                    ],
                    "quarterly_context": {
                        "quarter_period": f"{quarterly_context.quarter_start} to {quarterly_context.quarter_end}",
                        "portfolio_return": quarterly_context.portfolio_return,
                        "market_benchmarks": quarterly_context.market_context,
                        "stock_performance": {
                            "quarterly_return": quarterly_perf.quarterly_return if quarterly_perf else 0.0,
                            "volatility": quarterly_perf.volatility if quarterly_perf else 0.0,
                            "max_drawdown": quarterly_perf.max_drawdown if quarterly_perf else 0.0,
                            "news_sentiment": quarterly_perf.positive_news_ratio if quarterly_perf else 0.0,
                            "insider_activity": quarterly_perf.net_insider_buying if quarterly_perf else 0.0,
                            "major_events": quarterly_perf.major_events if quarterly_perf else []
                        }
                    }
                }
                
                stock_result = self.debate_system.conduct_debate(
                    holding.ticker, 
                    quarterly_context.quarter_end, 
                    current_weight=holding.weight,
                    portfolio_context=portfolio_context
                )
                result.add_stock_result(holding.ticker, stock_result)
                
                console.print(f"[green]✅ {holding.ticker} quarterly analysis complete[/green]")
                
            except Exception as e:
                console.print(f"[red]❌ Error analyzing {holding.ticker}: {str(e)}[/red]")
                continue
        
        return result

    def _analyze_portfolio_parallel_with_quarterly_data(self, portfolio: Portfolio, quarterly_context: QuarterlyPortfolioContext, result: PortfolioDebateResult) -> PortfolioDebateResult:
        """Analyze portfolio holdings in parallel with quarterly data context"""
        
        # First, run initial analysis in parallel
        initial_results = self._run_parallel_initial_analysis_with_quarterly_data(portfolio, quarterly_context)
        
        # Then conduct debates sequentially (debates require sequential processing)
        for i, holding in enumerate(portfolio.holdings, 1):
            console.print(f"\n[bold yellow]🎭 Conducting quarterly debate for {holding.ticker} ({holding.weight:.1%})[/bold yellow]")
            console.print(f"[dim]Debate {i} of {len(portfolio.holdings)}[/dim]")
            
            try:
                # Get quarterly performance for this stock
                quarterly_perf = quarterly_context.individual_performance.get(holding.ticker)
                
                # Enhanced portfolio context with quarterly data
                portfolio_context = {
                    "portfolio_name": portfolio.name,
                    "total_holdings": len(portfolio.holdings),
                    "cash_allocation": portfolio.cash_weight,
                    "other_holdings": [
                        {"ticker": h.ticker, "weight": h.weight} 
                        for h in portfolio.holdings if h.ticker != holding.ticker
                    ],
                    "quarterly_context": {
                        "quarter_period": f"{quarterly_context.quarter_start} to {quarterly_context.quarter_end}",
                        "portfolio_return": quarterly_context.portfolio_return,
                        "market_benchmarks": quarterly_context.market_context,
                        "stock_performance": {
                            "quarterly_return": quarterly_perf.quarterly_return if quarterly_perf else 0.0,
                            "volatility": quarterly_perf.volatility if quarterly_perf else 0.0,
                            "max_drawdown": quarterly_perf.max_drawdown if quarterly_perf else 0.0,
                            "news_sentiment": quarterly_perf.positive_news_ratio if quarterly_perf else 0.0,
                            "insider_activity": quarterly_perf.net_insider_buying if quarterly_perf else 0.0,
                            "major_events": quarterly_perf.major_events if quarterly_perf else []
                        }
                    },
                    "initial_analysis": initial_results.get(holding.ticker, {})
                }
                
                stock_result = self.debate_system.conduct_debate(
                    holding.ticker,
                    quarterly_context.quarter_end,
                    current_weight=holding.weight,
                    portfolio_context=portfolio_context
                )
                result.add_stock_result(holding.ticker, stock_result)
                
                console.print(f"[green]✅ {holding.ticker} quarterly debate complete[/green]")
                
            except Exception as e:
                console.print(f"[red]❌ Error in quarterly debate for {holding.ticker}: {str(e)}[/red]")
                continue
        
        return result

    def _run_parallel_initial_analysis_with_quarterly_data(self, portfolio: Portfolio, quarterly_context: QuarterlyPortfolioContext) -> Dict[str, Dict]:
        """Run initial analysis for all stocks in parallel with quarterly context"""
        
        def analyze_stock_initial_with_quarterly(holding):
            """Run initial analysis for a single stock with quarterly data"""
            try:
                quarterly_perf = quarterly_context.individual_performance.get(holding.ticker)
                
                # Enhanced context for initial analysis
                context = {
                    "ticker": holding.ticker,
                    "current_weight": holding.weight,
                    "quarterly_performance": {
                        "return": quarterly_perf.quarterly_return if quarterly_perf else 0.0,
                        "volatility": quarterly_perf.volatility if quarterly_perf else 0.0,
                        "news_count": quarterly_perf.news_count if quarterly_perf else 0,
                        "sentiment": quarterly_perf.positive_news_ratio if quarterly_perf else 0.0
                    }
                }
                
                # This would call the initial analysis with quarterly context
                # For now, return the quarterly context as initial analysis
                return holding.ticker, context
                
            except Exception as e:
                console.print(f"[red]Error in initial analysis for {holding.ticker}: {e}[/red]")
                return holding.ticker, {}
        
        console.print("[cyan]🔄 Running parallel initial analysis with quarterly data...[/cyan]")
        
        initial_results = {}
        with ThreadPoolExecutor(max_workers=min(len(portfolio.holdings), 10)) as executor:
            future_to_holding = {
                executor.submit(analyze_stock_initial_with_quarterly, holding): holding 
                for holding in portfolio.holdings
            }
            
            for future in as_completed(future_to_holding):
                ticker, analysis = future.result()
                initial_results[ticker] = analysis
                console.print(f"[green]✓[/green] Initial quarterly analysis complete for {ticker}")
        
        return initial_results
    
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
    
    def generate_trading_actions_json(self, result: PortfolioDebateResult) -> str:
        """Generate JSON output with percentage changes and trading actions for each stock"""
        
        trading_actions = []
        
        for ticker, consensus in result.individual_results.items():
            # Find the holding in the original portfolio
            holding = None
            for h in result.portfolio.holdings:
                if h.ticker == ticker:
                    holding = h
                    break
            
            if not holding:
                continue
                
            current_weight = holding.weight
            
            # Calculate suggested new weight based on consensus
            suggested_weight = self._calculate_suggested_weight(
                current_weight, consensus.signal, consensus.confidence
            )
            
            # Calculate percentage change
            weight_change = suggested_weight - current_weight
            percentage_change = (weight_change / current_weight * 100) if current_weight > 0 else 0
            
            # Determine action
            action = self._determine_action(weight_change, current_weight, suggested_weight)
            
            trading_action = {
                "ticker": ticker,
                "current_weight": round(current_weight * 100, 2),  # Convert to percentage
                "suggested_weight": round(suggested_weight * 100, 2),  # Convert to percentage
                "weight_change": round(weight_change * 100, 2),  # Convert to percentage
                "percentage_change": round(percentage_change, 2),
                "action": action,
                "signal": consensus.signal,
                "confidence": round(consensus.confidence, 1),
                "reasoning": consensus.reasoning[:200] + "..." if len(consensus.reasoning) > 200 else consensus.reasoning
            }
            
            trading_actions.append(trading_action)
        
        # Add cash allocation info
        cash_info = {
            "ticker": "CASH",
            "current_weight": round((1.0 - sum(h.weight for h in result.portfolio.holdings)) * 100, 2),
            "suggested_weight": 10.0,  # Fixed 10% cash allocation
            "weight_change": 0.0,
            "percentage_change": 0.0,
            "action": "hold",
            "signal": "neutral",
            "confidence": 100.0,
            "reasoning": "Fixed cash allocation maintained at 10%"
        }
        
        trading_actions.append(cash_info)
        
        # Create summary
        output = {
            "portfolio_name": result.portfolio.name,
            "analysis_timestamp": result.timestamp.isoformat(),
            "total_holdings": len(result.portfolio.holdings),
            "rebalancing_recommended": any(abs(action["weight_change"]) > 5.0 for action in trading_actions if action["ticker"] != "CASH"),
            "trading_actions": trading_actions,
            "portfolio_summary": {
                "overall_signal": self._calculate_portfolio_signal(result),
                "weighted_confidence": self._calculate_weighted_confidence(result),
                "total_weight_changes": sum(abs(action["weight_change"]) for action in trading_actions if action["ticker"] != "CASH")
            }
        }
        
        return json.dumps(output, indent=2)
    
    def _calculate_suggested_weight(self, current_weight: float, signal: str, confidence: float) -> float:
        """Calculate suggested weight based on consensus signal and confidence"""
        
        # Base adjustment factor based on signal
        if signal == "bullish":
            base_adjustment = 0.3  # Up to 30% increase
        elif signal == "bearish":
            base_adjustment = -0.8  # Up to 80% decrease (minimum 1%)
        else:  # neutral
            base_adjustment = 0.0  # No change
        
        # Scale by confidence (0-100 -> 0-1)
        confidence_factor = confidence / 100.0
        
        # Calculate adjustment
        adjustment = base_adjustment * confidence_factor
        
        # Apply adjustment
        suggested_weight = current_weight * (1 + adjustment)
        
        # Apply constraints
        suggested_weight = max(0.01, min(0.7, suggested_weight))  # Between 1% and 70%
        
        return suggested_weight
    
    def _determine_action(self, weight_change: float, current_weight: float, suggested_weight: float) -> str:
        """Determine trading action based on weight changes"""
        
        # Define thresholds - weight_change is in decimal form (e.g., 0.01 = 1%)
        hold_threshold = 0.01  # 1% threshold for hold vs buy/sell
        
        if abs(weight_change) < hold_threshold:
            return "hold"
        elif weight_change > 0:
            return "buy"
        else:
            return "sell"
    
    def _calculate_portfolio_signal(self, result: PortfolioDebateResult) -> str:
        """Calculate overall portfolio signal"""
        
        bullish_weight = 0
        bearish_weight = 0
        
        for ticker, consensus in result.individual_results.items():
            # Find holding weight
            holding_weight = 0
            for h in result.portfolio.holdings:
                if h.ticker == ticker:
                    holding_weight = h.weight
                    break
            
            signal = consensus.signal
            confidence = consensus.confidence / 100.0
            weighted_signal = holding_weight * confidence
            
            if signal == "bullish":
                bullish_weight += weighted_signal
            elif signal == "bearish":
                bearish_weight += weighted_signal
        
        if bullish_weight > bearish_weight * 1.2:
            return "bullish"
        elif bearish_weight > bullish_weight * 1.2:
            return "bearish"
        else:
            return "neutral"
    
    def _calculate_weighted_confidence(self, result: PortfolioDebateResult) -> float:
        """Calculate portfolio-weighted confidence"""
        
        total_weighted_confidence = 0
        total_weight = 0
        
        for ticker, consensus in result.individual_results.items():
            # Find holding weight
            holding_weight = 0
            for h in result.portfolio.holdings:
                if h.ticker == ticker:
                    holding_weight = h.weight
                    break
            
            confidence = consensus.confidence
            total_weighted_confidence += holding_weight * confidence
            total_weight += holding_weight
        
        return round(total_weighted_confidence / total_weight if total_weight > 0 else 0, 1)
    
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
    
    def _display_quarterly_portfolio_results(self, result: PortfolioDebateResult, quarterly_context: QuarterlyPortfolioContext):
        """Display comprehensive quarterly portfolio analysis results"""
        
        # Quarterly Performance Summary
        quarterly_summary_table = Table(title="📊 Quarterly Performance vs Current Analysis")
        quarterly_summary_table.add_column("Metric", style="cyan", no_wrap=True)
        quarterly_summary_table.add_column("Quarterly Performance", style="blue")
        quarterly_summary_table.add_column("Current Signal", style="bold")
        
        quarterly_summary_table.add_row(
            "Portfolio Return",
            f"{quarterly_context.portfolio_return:.2%}",
            result.overall_signal.upper()
        )
        quarterly_summary_table.add_row(
            "Portfolio Volatility", 
            f"{quarterly_context.portfolio_volatility:.2%}",
            f"{result.weighted_confidence:.1f}% confidence"
        )
        quarterly_summary_table.add_row(
            "Sharpe Ratio",
            f"{quarterly_context.portfolio_sharpe:.2f}",
            f"Score: {result.portfolio_score:.2f}"
        )
        
        console.print(quarterly_summary_table)
        
        # Market Context vs Portfolio
        market_table = Table(title="📈 Market Context")
        market_table.add_column("Benchmark", style="cyan")
        market_table.add_column("Quarterly Return", style="blue")
        market_table.add_column("vs Portfolio", style="bold")
        
        for benchmark, return_val in quarterly_context.market_context.items():
            relative_performance = quarterly_context.portfolio_return - return_val
            performance_indicator = "🟢" if relative_performance > 0 else "🔴"
            market_table.add_row(
                benchmark,
                f"{return_val:.2%}",
                f"{performance_indicator} {relative_performance:+.2%}"
            )
        
        console.print(market_table)
        
        # Individual stock results with quarterly context
        stock_table = Table(title="Individual Stock Analysis with Quarterly Context")
        stock_table.add_column("Ticker", style="cyan", no_wrap=True)
        stock_table.add_column("Q Return", style="blue")
        stock_table.add_column("Q Vol", style="yellow")
        stock_table.add_column("News", style="green")
        stock_table.add_column("Signal", style="bold")
        stock_table.add_column("Confidence", style="magenta")
        
        for holding in result.portfolio.holdings:
            ticker = holding.ticker
            quarterly_perf = quarterly_context.individual_performance.get(ticker)
            
            if ticker in result.individual_results and quarterly_perf:
                stock_result = result.individual_results[ticker]
                
                # Color code the signal
                signal_color = {
                    "bullish": "green",
                    "bearish": "red",
                    "neutral": "yellow"
                }.get(stock_result.signal.lower(), "white")
                
                # News sentiment indicator
                news_indicator = "🟢" if quarterly_perf.positive_news_ratio > 0.6 else "🟡" if quarterly_perf.positive_news_ratio > 0.4 else "🔴"
                
                stock_table.add_row(
                    ticker,
                    f"{quarterly_perf.quarterly_return:.2%}",
                    f"{quarterly_perf.volatility:.1%}",
                    f"{news_indicator} {quarterly_perf.news_count}",
                    f"[{signal_color}]{stock_result.signal.upper()}[/{signal_color}]",
                    f"{stock_result.confidence:.1f}%"
                )
        
        console.print(stock_table)
        
        # Display regular portfolio results as well
        self._display_portfolio_results(result)
    
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
        console.print("\n3. Run quarterly analysis:")
        console.print("   python src/portfolio_debate.py quarterly <portfolio_name>")
        console.print("   python src/portfolio_debate.py quarterly custom AAPL:0.3 TSLA:0.4 NVDA:0.3")
        console.print("\nExamples:")
        console.print("   python src/portfolio_debate.py tech_growth")
        console.print("   python src/portfolio_debate.py quarterly tech_growth")
        console.print("   python src/portfolio_debate.py custom AAPL:0.5 MSFT:0.3 GOOGL:0.2")
        console.print("   python src/portfolio_debate.py quarterly custom AAPL:0.5 MSFT:0.3 GOOGL:0.2")
        return
    
    portfolio_system = PortfolioDebateSystem()
    
    # Check if quarterly analysis is requested
    quarterly_mode = False
    if sys.argv[1].lower() == "quarterly":
        quarterly_mode = True
        if len(sys.argv) < 3:
            console.print("[red]Error: Quarterly mode requires portfolio name[/red]")
            console.print("Example: python src/portfolio_debate.py quarterly tech_growth")
            return
        portfolio_name = sys.argv[2].lower()
        remaining_args = sys.argv[3:]  # For custom portfolios
    else:
        portfolio_name = sys.argv[1].lower()
        remaining_args = sys.argv[2:]  # For custom portfolios
    
    if portfolio_name == "custom":
        # Parse custom portfolio from command line
        if len(remaining_args) < 1:
            console.print("[red]Error: Custom portfolio requires ticker:weight pairs[/red]")
            console.print("Example: python src/portfolio_debate.py custom AAPL:0.4 TSLA:0.6")
            return
        
        try:
            holdings_dict = {}
            for arg in remaining_args:
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
        # Conduct portfolio analysis (quarterly or regular)
        if quarterly_mode:
            console.print(f"[bold cyan]📊 Running QUARTERLY analysis for {portfolio.name}[/bold cyan]")
            result, json_output = portfolio_system.analyze_portfolio_quarterly(portfolio)
            analysis_type = "quarterly"
        else:
            console.print(f"[bold cyan]📈 Running REGULAR analysis for {portfolio.name}[/bold cyan]")
            result, json_output = portfolio_system.analyze_portfolio(portfolio)
            analysis_type = "regular"
        
        console.print(f"\n[bold green]✅ {analysis_type.title()} portfolio analysis complete for {portfolio.name}![/bold green]")
        
        # Optionally save JSON output to file
        json_filename = f"trading_actions_{analysis_type}_{portfolio.name.lower().replace(' ', '_')}.json"
        with open(json_filename, 'w') as f:
            f.write(json_output)
        console.print(f"[dim]💾 JSON output saved to {json_filename}[/dim]")
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Analysis interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"[red]Error during portfolio analysis: {e}[/red]")


if __name__ == "__main__":
    main()
