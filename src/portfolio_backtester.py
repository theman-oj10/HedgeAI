#!/usr/bin/env python3
"""
Portfolio Backtester with Debate System Integration

This script backtests portfolio strategies using the portfolio debate system
to generate trading signals and actions.
"""

import json
import os
import sys
from datetime import datetime, timedelta
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from colorama import Fore, Style, init
from dateutil.relativedelta import relativedelta

from portfolio_config import (
    SAMPLE_PORTFOLIOS,
    Portfolio,
)

# Import the portfolio debate system
from portfolio_debate import PortfolioDebateSystem

# Import price data tools from the original backtester
from tools.api import get_price_data, get_prices

init(autoreset=True)


class PortfolioDebateBacktester:
    """Backtester that uses portfolio debate system for trading decisions"""
    
    def __init__(
        self,
        portfolio: Portfolio,
        start_date: str,
        end_date: str,
        initial_capital: float,
        trading_frequency: str = "QE",
        rebalance_threshold: float = 0.05,  # 5% threshold for rebalancing
        output_dir: str = "backtest_results",
    ):
        """
        Initialize the portfolio debate backtester
        
        :param portfolio: Portfolio object to backtest
        :param start_date: Start date string (YYYY-MM-DD)
        :param end_date: End date string (YYYY-MM-DD)
        :param initial_capital: Starting portfolio cash
        :param trading_frequency: Trading frequency ('B' for daily, 'QE' for quarterly)
        :param rebalance_threshold: Minimum percentage change to trigger rebalancing
        """
        self.portfolio = portfolio
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.trading_frequency = trading_frequency
        self.rebalance_threshold = rebalance_threshold
        self.output_dir = output_dir
        
        # Initialize debate system
        self.debate_system = PortfolioDebateSystem()
        
        # Extract tickers from portfolio
        self.tickers = [holding.ticker for holding in portfolio.holdings]
        
        # Initialize backtesting state
        self.portfolio_values = []
        self.sp500_values = []  # Track S&P 500 for comparison
        self.current_positions = {
            ticker: {
                "shares": 0,
                "cost_basis": 0.0,
                "current_value": 0.0
            }
            for ticker in self.tickers
        }
        self.cash = initial_capital
        self.trades_history = []
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Generate unique session ID for this backtest
        self.session_id = f"{portfolio.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
    def execute_trades_from_json(self, json_output: str, current_prices: Dict[str, float], current_date: str):
        """
        Execute trades based on JSON output from portfolio debate system
        
        :param json_output: JSON string from analyze_portfolio
        :param current_prices: Dictionary of current ticker prices
        :param current_date: Current date as string
        :return: Dictionary of executed trades
        """
        try:
            trading_data = json.loads(json_output)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON output: {e}")
            return {}
        
        executed_trades = {}
        trading_actions = trading_data.get("trading_actions", [])
        
        # Calculate current portfolio value
        total_portfolio_value = self.cash
        for ticker in self.tickers:
            if ticker in current_prices:
                position_value = self.current_positions[ticker]["shares"] * current_prices[ticker]
                self.current_positions[ticker]["current_value"] = position_value
                total_portfolio_value += position_value
        
        # Process each trading action
        for action in trading_actions:
            ticker = action["ticker"]
            
            # Skip cash allocation
            if ticker == "CASH":
                continue
            
            if ticker not in current_prices:
                print(f"Warning: No price data for {ticker}, skipping trade")
                continue
            
            current_price = current_prices[ticker]
            
            # Get weight changes from the JSON
            suggested_weight = action["suggested_weight"] / 100.0
            weight_change = action["weight_change"] / 100.0
            trade_action = action["action"]
            confidence = action["confidence"]
            
            # Only execute if change is significant enough
            if abs(weight_change) < self.rebalance_threshold and trade_action == "hold":
                executed_trades[ticker] = {
                    "action": "hold",
                    "quantity": 0,
                    "reason": f"Change below threshold ({abs(weight_change):.1%} < {self.rebalance_threshold:.1%})"
                }
                continue
            
            # Calculate target position value
            target_value = total_portfolio_value * suggested_weight
            current_position_value = self.current_positions[ticker]["current_value"]
            
            # Calculate value difference
            value_difference = target_value - current_position_value
            
            # Calculate shares to trade
            if abs(value_difference) > 10:  # Minimum $10 trade
                shares_to_trade = int(abs(value_difference) / current_price)
                
                if value_difference > 0:  # Need to buy
                    # Check if we have enough cash
                    cost = shares_to_trade * current_price
                    if cost <= self.cash:
                        # Execute buy
                        self.current_positions[ticker]["shares"] += shares_to_trade
                        self.current_positions[ticker]["cost_basis"] += cost
                        self.cash -= cost
                        
                        executed_trades[ticker] = {
                            "action": "buy",
                            "quantity": shares_to_trade,
                            "price": current_price,
                            "cost": cost,
                            "reason": action["reasoning"][:100]
                        }
                        
                        self.trades_history.append({
                            "date": current_date,
                            "ticker": ticker,
                            "action": "buy",
                            "shares": shares_to_trade,
                            "price": current_price,
                            "value": cost,
                            "signal": action["signal"],
                            "confidence": confidence
                        })
                    else:
                        # Partial buy with available cash
                        max_shares = int(self.cash / current_price)
                        if max_shares > 0:
                            cost = max_shares * current_price
                            self.current_positions[ticker]["shares"] += max_shares
                            self.current_positions[ticker]["cost_basis"] += cost
                            self.cash -= cost
                            
                            executed_trades[ticker] = {
                                "action": "buy",
                                "quantity": max_shares,
                                "price": current_price,
                                "cost": cost,
                                "reason": "Partial buy (limited by cash)"
                            }
                            
                            self.trades_history.append({
                                "date": current_date,
                                "ticker": ticker,
                                "action": "buy",
                                "shares": max_shares,
                                "price": current_price,
                                "value": cost,
                                "signal": action["signal"],
                                "confidence": confidence
                            })
                
                elif value_difference < 0:  # Need to sell
                    # Can only sell what we have
                    shares_to_sell = min(shares_to_trade, self.current_positions[ticker]["shares"])
                    
                    if shares_to_sell > 0:
                        proceeds = shares_to_sell * current_price
                        
                        # Calculate cost basis for sold shares
                        avg_cost_basis = (self.current_positions[ticker]["cost_basis"] / 
                                        self.current_positions[ticker]["shares"] 
                                        if self.current_positions[ticker]["shares"] > 0 else 0)
                        
                        cost_of_sold = shares_to_sell * avg_cost_basis
                        
                        # Update position
                        self.current_positions[ticker]["shares"] -= shares_to_sell
                        self.current_positions[ticker]["cost_basis"] -= cost_of_sold
                        self.cash += proceeds
                        
                        executed_trades[ticker] = {
                            "action": "sell",
                            "quantity": shares_to_sell,
                            "price": current_price,
                            "proceeds": proceeds,
                            "realized_gain": proceeds - cost_of_sold,
                            "reason": action["reasoning"][:100]
                        }
                        
                        self.trades_history.append({
                            "date": current_date,
                            "ticker": ticker,
                            "action": "sell",
                            "shares": shares_to_sell,
                            "price": current_price,
                            "value": proceeds,
                            "signal": action["signal"],
                            "confidence": confidence,
                            "realized_gain": proceeds - cost_of_sold
                        })
            else:
                executed_trades[ticker] = {
                    "action": "hold",
                    "quantity": 0,
                    "reason": f"Trade value too small (${abs(value_difference):.2f})"
                }
        
        return executed_trades
    
    def calculate_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate total portfolio value including cash and positions"""
        total_value = self.cash
        
        for ticker in self.tickers:
            if ticker in current_prices and ticker in self.current_positions:
                position_value = self.current_positions[ticker]["shares"] * current_prices[ticker]
                total_value += position_value
        
        return total_value
    
    def prefetch_data(self):
        """Pre-fetch all data needed for the backtest period"""
        print("\nPre-fetching price data for backtest period...")
        
        # Fetch data for entire period plus buffer
        start_date_dt = datetime.strptime(self.start_date, "%Y-%m-%d") - relativedelta(months=3)
        
        # Fetch portfolio tickers
        for ticker in self.tickers:
            try:
                get_prices(ticker, start_date_dt.strftime("%Y-%m-%d"), self.end_date)
                print(f"✓ Fetched data for {ticker}")
            except Exception as e:
                print(f"✗ Error fetching data for {ticker}: {e}")
        
        # Fetch S&P 500 data (using SPY ETF as proxy)
        try:
            get_prices("SPY", start_date_dt.strftime("%Y-%m-%d"), self.end_date)
            print("✓ Fetched data for SPY (S&P 500)")
        except Exception as e:
            print(f"✗ Error fetching S&P 500 data: {e}")
        
        print("Data pre-fetch complete.\n")
    
    def run_backtest(self):
        """Run the portfolio backtest using debate system signals"""
        
        # Pre-fetch all required data
        self.prefetch_data()
        
        # Generate trading dates based on frequency
        dates = pd.date_range(self.start_date, self.end_date, freq=self.trading_frequency)
        
        print(f"\n{Fore.CYAN}Starting Portfolio Debate Backtest{Style.RESET_ALL}")
        print(f"Portfolio: {self.portfolio.name}")
        print(f"Period: {self.start_date} to {self.end_date}")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Trading Frequency: {'Daily' if self.trading_frequency == 'B' else 'Quarterly'}")
        print(f"Benchmark: S&P 500 (SPY ETF)")
        print("=" * 80)
        
        # Track performance metrics
        performance_metrics = {
            "sharpe_ratio": None,
            "max_drawdown": None,
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "sp500_return": None,
            "portfolio_return": None
        }
        
        # Track initial S&P 500 value
        initial_sp500_price = None
        
        for i, current_date in enumerate(dates):
            current_date_str = current_date.strftime("%Y-%m-%d")
            
            # Skip first date (need historical data)
            if i == 0:
                # Get initial S&P 500 price
                try:
                    sp500_data = get_price_data("SPY", current_date_str, current_date_str)
                    if not sp500_data.empty:
                        initial_sp500_price = sp500_data.iloc[-1]["close"]
                        self.sp500_values.append({
                            "Date": current_date,
                            "SPY Price": initial_sp500_price,
                            "SPY Value": self.initial_capital  # Normalized to initial capital
                        })
                        print(f"Initial S&P 500 (SPY) price: ${initial_sp500_price:.2f}")
                    else:
                        print(f"Warning: No S&P 500 data for initial date {current_date_str}")
                except Exception as e:
                    print(f"Error fetching initial S&P 500 data: {e}")
                
                # Record initial portfolio value
                self.portfolio_values.append({
                    "Date": current_date,
                    "Portfolio Value": self.initial_capital,
                    "Cash": self.initial_capital,
                    "Positions Value": 0.0
                })
                continue
            
            print(f"\n{Fore.YELLOW}Trading Date: {current_date_str}{Style.RESET_ALL}")
            
            # Get current prices
            try:
                current_prices = {}
                previous_date_str = (current_date - timedelta(days=1)).strftime("%Y-%m-%d")
                
                for ticker in self.tickers:
                    price_data = get_price_data(ticker, previous_date_str, current_date_str)
                    if not price_data.empty:
                        current_prices[ticker] = price_data.iloc[-1]["close"]
                    else:
                        print(f"Warning: No price data for {ticker} on {current_date_str}")
                
                if len(current_prices) != len(self.tickers):
                    print(f"Skipping {current_date_str} due to missing price data")
                    continue
                
                # Get S&P 500 price
                current_sp500_price = None
                if initial_sp500_price:
                    try:
                        sp500_data = get_price_data("SPY", previous_date_str, current_date_str)
                        if not sp500_data.empty:
                            current_sp500_price = sp500_data.iloc[-1]["close"]
                            # Calculate S&P 500 equivalent portfolio value
                            sp500_value = self.initial_capital * (current_sp500_price / initial_sp500_price)
                            self.sp500_values.append({
                                "Date": current_date,
                                "SPY Price": current_sp500_price,
                                "SPY Value": sp500_value
                            })
                            print(f"  SPY price: ${current_sp500_price:.2f}, Value: ${sp500_value:,.2f}")
                        else:
                            print(f"  Warning: No S&P 500 data for {current_date_str}")
                    except Exception as e:
                        print(f"  Error fetching S&P 500 data: {e}")
                    
            except Exception as e:
                print(f"Error fetching prices for {current_date_str}: {e}")
                continue
            
            # Run portfolio debate analysis
            print(f"{Fore.CYAN}Running portfolio debate analysis...{Style.RESET_ALL}")
            
            try:
                # Call the portfolio debate system
                result, json_output = self.debate_system.analyze_portfolio(
                    self.portfolio, 
                    end_date=current_date_str,
                    parallel=True  # Use parallel analysis for speed
                )
                
                # Execute trades based on the JSON output
                executed_trades = self.execute_trades_from_json(
                    json_output, 
                    current_prices, 
                    current_date_str
                )
                
                # Update performance metrics
                for ticker, trade in executed_trades.items():
                    if trade["action"] in ["buy", "sell"]:
                        performance_metrics["total_trades"] += 1
                        if trade["action"] == "sell" and "realized_gain" in trade:
                            if trade["realized_gain"] > 0:
                                performance_metrics["winning_trades"] += 1
                            else:
                                performance_metrics["losing_trades"] += 1
                
                # Calculate and record portfolio value
                portfolio_value = self.calculate_portfolio_value(current_prices)
                positions_value = portfolio_value - self.cash
                
                self.portfolio_values.append({
                    "Date": current_date,
                    "Portfolio Value": portfolio_value,
                    "Cash": self.cash,
                    "Positions Value": positions_value
                })
                
                # Display summary for this period
                print(f"\n{Fore.GREEN}Period Summary:{Style.RESET_ALL}")
                print(f"Portfolio Value: ${portfolio_value:,.2f}")
                print(f"Cash: ${self.cash:,.2f}")
                print(f"Positions Value: ${positions_value:,.2f}")
                print(f"Trades Executed: {sum(1 for t in executed_trades.values() if t['action'] != 'hold')}")
                
            except Exception as e:
                print(f"{Fore.RED}Error in portfolio analysis for {current_date_str}: {e}{Style.RESET_ALL}")
                continue
        
        # Calculate final performance metrics
        self._calculate_performance_metrics(performance_metrics, initial_sp500_price)
        
        # Save backtest results to file
        self._save_backtest_results(performance_metrics)
        
        return performance_metrics
    
    def _calculate_performance_metrics(self, metrics: Dict, initial_sp500_price: float = None):
        """Calculate performance metrics from portfolio values"""
        if len(self.portfolio_values) < 2:
            return
        
        values_df = pd.DataFrame(self.portfolio_values).set_index("Date")
        values_df["Returns"] = values_df["Portfolio Value"].pct_change()
        
        # Calculate portfolio return
        initial_value = self.initial_capital
        final_value = values_df["Portfolio Value"].iloc[-1]
        metrics["portfolio_return"] = ((final_value - initial_value) / initial_value) * 100
        
        # Calculate S&P 500 return if available
        if self.sp500_values and len(self.sp500_values) > 1:
            sp500_df = pd.DataFrame(self.sp500_values).set_index("Date")
            initial_sp500_value = sp500_df["SPY Value"].iloc[0]
            final_sp500_value = sp500_df["SPY Value"].iloc[-1]
            metrics["sp500_return"] = ((final_sp500_value - initial_sp500_value) / initial_sp500_value) * 100
        
        # Calculate Sharpe Ratio
        if self.trading_frequency == "B":
            periods_per_year = 252
        else:
            periods_per_year = 4
        
        returns = values_df["Returns"].dropna()
        if len(returns) > 0:
            mean_return = returns.mean()
            std_return = returns.std()
            if std_return > 0:
                metrics["sharpe_ratio"] = np.sqrt(periods_per_year) * (mean_return / std_return)
        
        # Calculate Maximum Drawdown
        rolling_max = values_df["Portfolio Value"].cummax()
        drawdown = (values_df["Portfolio Value"] - rolling_max) / rolling_max
        metrics["max_drawdown"] = drawdown.min() * 100
    
    def _save_backtest_results(self, performance_metrics: Dict):
        """Save backtest results to file for dashboard consumption"""
        
        # Prepare data for saving
        backtest_data = {
            "session_id": self.session_id,
            "portfolio_name": self.portfolio.name,
            "timestamp": datetime.now().isoformat(),
            "config": {
                "start_date": self.start_date,
                "end_date": self.end_date,
                "initial_capital": self.initial_capital,
                "trading_frequency": self.trading_frequency,
                "rebalance_threshold": self.rebalance_threshold
            },
            "performance_metrics": performance_metrics,
            "portfolio_values": self.portfolio_values,
            "sp500_values": self.sp500_values,
            "trades_history": self.trades_history,
            "final_positions": {
                ticker: {
                    "shares": position["shares"],
                    "cost_basis": position["cost_basis"],
                    "current_value": position["current_value"]
                }
                for ticker, position in self.current_positions.items()
            },
            "final_cash": self.cash
        }
        
        # Convert dates to strings for JSON serialization
        for portfolio_val in backtest_data["portfolio_values"]:
            portfolio_val["Date"] = portfolio_val["Date"].isoformat()
        
        for sp500_val in backtest_data["sp500_values"]:
            sp500_val["Date"] = sp500_val["Date"].isoformat()
        
        # Save to individual session file
        session_file = os.path.join(self.output_dir, f"{self.session_id}.json")
        with open(session_file, 'w') as f:
            json.dump(backtest_data, f, indent=2, default=str)
        
        print(f"\n{Fore.GREEN}Backtest results saved to: {session_file}{Style.RESET_ALL}")
        
        # Append summary to master log file
        master_log_file = os.path.join(self.output_dir, "backtest_master_log.jsonl")
        
        summary_data = {
            "session_id": self.session_id,
            "portfolio_name": self.portfolio.name,
            "timestamp": datetime.now().isoformat(),
            "start_date": self.start_date,
            "end_date": self.end_date,
            "initial_capital": self.initial_capital,
            "final_value": self.portfolio_values[-1]["Portfolio Value"] if self.portfolio_values else self.initial_capital,
            "total_return": performance_metrics.get("portfolio_return") or 0,
            "sp500_return": performance_metrics.get("sp500_return") or 0,
            "sharpe_ratio": performance_metrics.get("sharpe_ratio") or 0,
            "max_drawdown": performance_metrics.get("max_drawdown") or 0,
            "total_trades": performance_metrics.get("total_trades") or 0,
            "winning_trades": performance_metrics.get("winning_trades") or 0,
            "losing_trades": performance_metrics.get("losing_trades") or 0
        }
        
        # Append to master log file (one summary per line)
        with open(master_log_file, 'a') as f:
            f.write(json.dumps(summary_data, default=str) + '\n')
        
        print(f"Summary appended to: {master_log_file}")
        
        # Save latest portfolio state for quick access
        latest_state_file = os.path.join(self.output_dir, "latest_portfolio_state.json")
        latest_state = {
            "last_updated": datetime.now().isoformat(),
            "portfolio_name": self.portfolio.name,
            "current_value": self.portfolio_values[-1]["Portfolio Value"] if self.portfolio_values else self.initial_capital,
            "cash": self.cash,
            "positions": self.current_positions,
            "recent_trades": self.trades_history[-10:] if len(self.trades_history) > 10 else self.trades_history,
            "performance": {
                "total_return_pct": performance_metrics.get("portfolio_return") or 0,
                "vs_sp500": (performance_metrics.get("portfolio_return") or 0) - (performance_metrics.get("sp500_return") or 0)
            }
        }
        
        with open(latest_state_file, 'w') as f:
            json.dump(latest_state, f, indent=2, default=str)
        
        print(f"Latest state saved to: {latest_state_file}")
    
    def analyze_performance(self):
        """Generate performance analysis and visualizations"""
        if not self.portfolio_values:
            print("No portfolio data found. Please run the backtest first.")
            return
        
        performance_df = pd.DataFrame(self.portfolio_values).set_index("Date")
        
        # Calculate returns
        initial_value = self.initial_capital
        final_value = performance_df["Portfolio Value"].iloc[-1]
        total_return = ((final_value - initial_value) / initial_value) * 100
        
        print(f"\n{Fore.WHITE}{Style.BRIGHT}PORTFOLIO PERFORMANCE SUMMARY:{Style.RESET_ALL}")
        print(f"Initial Capital: ${initial_value:,.2f}")
        print(f"Final Value: ${final_value:,.2f}")
        print(f"Total Return: {Fore.GREEN if total_return >= 0 else Fore.RED}{total_return:.2f}%{Style.RESET_ALL}")
        
        # Show S&P 500 comparison if available
        if self.sp500_values and len(self.sp500_values) > 1:
            sp500_df = pd.DataFrame(self.sp500_values).set_index("Date")
            sp500_return = ((sp500_df["SPY Value"].iloc[-1] - sp500_df["SPY Value"].iloc[0]) / sp500_df["SPY Value"].iloc[0]) * 100
            print(f"S&P 500 Return: {Fore.GREEN if sp500_return >= 0 else Fore.RED}{sp500_return:.2f}%{Style.RESET_ALL}")
            alpha = total_return - sp500_return
            print(f"Alpha (vs S&P 500): {Fore.GREEN if alpha >= 0 else Fore.RED}{alpha:.2f}%{Style.RESET_ALL}")
        else:
            print(f"S&P 500 comparison not available (data points: {len(self.sp500_values)})")
        
        # Plot portfolio value over time
        plt.figure(figsize=(14, 10))
        
        # Subplot 1: Portfolio Value vs S&P 500
        plt.subplot(3, 1, 1)
        plt.plot(performance_df.index, performance_df["Portfolio Value"], 
                color="blue", linewidth=2, label="Portfolio Value")
        
        # Add S&P 500 if available
        if self.sp500_values and len(self.sp500_values) > 1:
            sp500_df = pd.DataFrame(self.sp500_values).set_index("Date")
            print(f"\nS&P 500 data points: {len(sp500_df)}")
            print(f"Portfolio data points: {len(performance_df)}")
            # Align S&P 500 dates with portfolio dates
            sp500_aligned = sp500_df.reindex(performance_df.index, method='ffill')
            plt.plot(sp500_aligned.index, sp500_aligned["SPY Value"], 
                    color="red", linewidth=2, label="S&P 500 (SPY)", alpha=0.7)
        else:
            print(f"\nNo S&P 500 data for plotting (data points: {len(self.sp500_values)})")
        
        plt.plot(performance_df.index, performance_df["Cash"], 
                color="green", linewidth=1, linestyle="--", label="Cash", alpha=0.5)
        plt.plot(performance_df.index, performance_df["Positions Value"], 
                color="orange", linewidth=1, linestyle="--", label="Positions Value", alpha=0.5)
        plt.title(f"Portfolio Performance vs S&P 500: {self.portfolio.name}")
        plt.ylabel("Value ($)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Relative Performance (Normalized)
        plt.subplot(3, 1, 2)
        # Normalize both to 100 at start
        norm_portfolio = (performance_df["Portfolio Value"] / performance_df["Portfolio Value"].iloc[0]) * 100
        plt.plot(performance_df.index, norm_portfolio, 
                color="blue", linewidth=2, label="Portfolio")
        
        if self.sp500_values and len(self.sp500_values) > 1:
            sp500_df = pd.DataFrame(self.sp500_values).set_index("Date")
            sp500_aligned = sp500_df.reindex(performance_df.index, method='ffill')
            if not sp500_aligned["SPY Value"].isna().all():
                norm_sp500 = (sp500_aligned["SPY Value"] / sp500_aligned["SPY Value"].iloc[0]) * 100
                plt.plot(sp500_aligned.index, norm_sp500, 
                        color="red", linewidth=2, label="S&P 500", alpha=0.7)
            else:
                print("Warning: S&P 500 data could not be aligned with portfolio dates")
        
        plt.title("Normalized Performance (Base = 100)")
        plt.ylabel("Normalized Value")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 3: Period Returns
        plt.subplot(3, 1, 3)
        performance_df["Returns"] = performance_df["Portfolio Value"].pct_change() * 100
        plt.bar(performance_df.index, performance_df["Returns"], 
                color=performance_df["Returns"].apply(lambda x: "green" if x >= 0 else "red"))
        plt.title("Portfolio Period Returns (%)")
        plt.ylabel("Return (%)")
        plt.xlabel("Date")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Display trade history
        if self.trades_history:
            print(f"\n{Fore.CYAN}Trade History:{Style.RESET_ALL}")
            trades_df = pd.DataFrame(self.trades_history)
            print(trades_df.to_string())
        
        return performance_df


def main():
    """Main CLI interface for portfolio backtesting with debate system"""
    
    # Calculate default dates (1 year ago to today)
    today = datetime.now()
    one_year_ago = today - relativedelta(years=1)
    default_start = one_year_ago.strftime("%Y-%m-%d")
    default_end = today.strftime("%Y-%m-%d")
    
    if len(sys.argv) < 2:
        print(f"{Fore.RED}Usage:{Style.RESET_ALL}")
        print("python src/portfolio_backtester.py <portfolio_name> [start_date] [end_date] [initial_capital]")
        print("\nExample:")
        print("python src/portfolio_backtester.py tech_growth")
        print(f"python src/portfolio_backtester.py tech_growth {default_start} {default_end} 100000")
        print("\nAvailable portfolios: tech_growth, value_dividend, balanced_mix, ai_innovation")
        print(f"\nDefault period: {default_start} to {default_end} (1 year)")
        print("Default capital: $100,000")
        print("Default rebalancing: Quarterly")
        return
    
    portfolio_name = sys.argv[1].lower()
    start_date = sys.argv[2] if len(sys.argv) > 2 else default_start
    end_date = sys.argv[3] if len(sys.argv) > 3 else default_end
    initial_capital = float(sys.argv[4]) if len(sys.argv) > 4 else 100000.0
    
    # Get portfolio
    if portfolio_name in SAMPLE_PORTFOLIOS:
        portfolio = SAMPLE_PORTFOLIOS[portfolio_name]
    else:
        print(f"{Fore.RED}Unknown portfolio: {portfolio_name}{Style.RESET_ALL}")
        print(f"Available: {', '.join(SAMPLE_PORTFOLIOS.keys())}")
        return
    
    try:
        # Create and run backtester
        print(f"\n{Fore.CYAN}Backtest Configuration:{Style.RESET_ALL}")
        print(f"Period: {start_date} to {end_date}")
        print(f"Initial Capital: ${initial_capital:,.2f}")
        print("Rebalancing: Quarterly (every 3 months)")
        
        backtester = PortfolioDebateBacktester(
            portfolio=portfolio,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            trading_frequency="QE",  # Quarterly by default for debate system
            output_dir="backtest_results"  # Directory for saving results
        )
        
        # Run backtest
        backtester.run_backtest()
        
        # Analyze and display results
        backtester.analyze_performance()
        
        print(f"\n{Fore.GREEN}✅ Backtest complete!{Style.RESET_ALL}")
        
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Backtest interrupted by user{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}Error during backtest: {e}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()