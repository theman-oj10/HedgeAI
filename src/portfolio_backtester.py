#!/usr/bin/env python3
"""
Portfolio Backtester with Debate System Integration

This script backtests portfolio strategies using the portfolio debate system
to generate trading signals and actions.
"""

import json
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from colorama import Fore, Style, init
from dateutil.relativedelta import relativedelta

from portfolio_config import (
    SAMPLE_PORTFOLIOS,
    Portfolio,
    PortfolioHolding,
    create_custom_portfolio,
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
        
        # Initialize debate system
        self.debate_system = PortfolioDebateSystem()
        
        # Extract tickers from portfolio
        self.tickers = [holding.ticker for holding in portfolio.holdings]
        
        # Initialize backtesting state
        self.portfolio_values = []
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
            current_weight = action["current_weight"] / 100.0  # Convert from percentage
            suggested_weight = action["suggested_weight"] / 100.0
            weight_change = action["weight_change"] / 100.0
            percentage_change = action["percentage_change"]
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
                                "reason": f"Partial buy (limited by cash)"
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
        end_date_dt = datetime.strptime(self.end_date, "%Y-%m-%d")
        start_date_dt = datetime.strptime(self.start_date, "%Y-%m-%d") - relativedelta(months=3)
        
        for ticker in self.tickers:
            try:
                get_prices(ticker, start_date_dt.strftime("%Y-%m-%d"), self.end_date)
                print(f"✓ Fetched data for {ticker}")
            except Exception as e:
                print(f"✗ Error fetching data for {ticker}: {e}")
        
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
        print("=" * 80)
        
        # Track performance metrics
        performance_metrics = {
            "sharpe_ratio": None,
            "max_drawdown": None,
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0
        }
        
        for i, current_date in enumerate(dates):
            current_date_str = current_date.strftime("%Y-%m-%d")
            
            # Skip first date (need historical data)
            if i == 0:
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
        self._calculate_performance_metrics(performance_metrics)
        
        return performance_metrics
    
    def _calculate_performance_metrics(self, metrics: Dict):
        """Calculate performance metrics from portfolio values"""
        if len(self.portfolio_values) < 2:
            return
        
        values_df = pd.DataFrame(self.portfolio_values).set_index("Date")
        values_df["Returns"] = values_df["Portfolio Value"].pct_change()
        
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
        
        # Plot portfolio value over time
        plt.figure(figsize=(14, 8))
        
        # Subplot 1: Portfolio Value
        plt.subplot(2, 1, 1)
        plt.plot(performance_df.index, performance_df["Portfolio Value"], 
                color="blue", linewidth=2, label="Portfolio Value")
        plt.plot(performance_df.index, performance_df["Cash"], 
                color="green", linewidth=1, linestyle="--", label="Cash")
        plt.plot(performance_df.index, performance_df["Positions Value"], 
                color="orange", linewidth=1, linestyle="--", label="Positions Value")
        plt.title(f"Portfolio Performance: {self.portfolio.name}")
        plt.ylabel("Value ($)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Returns
        plt.subplot(2, 1, 2)
        performance_df["Returns"] = performance_df["Portfolio Value"].pct_change() * 100
        plt.bar(performance_df.index, performance_df["Returns"], 
                color=performance_df["Returns"].apply(lambda x: "green" if x >= 0 else "red"))
        plt.title("Period Returns (%)")
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
    
    if len(sys.argv) < 4:
        print(f"{Fore.RED}Usage:{Style.RESET_ALL}")
        print("python src/portfolio_backtester.py <portfolio_name> <start_date> <end_date> [initial_capital]")
        print("\nExample:")
        print("python src/portfolio_backtester.py tech_growth 2024-01-01 2024-12-31 100000")
        print("\nAvailable portfolios: tech_growth, value_dividend, balanced_mix, ai_innovation")
        return
    
    portfolio_name = sys.argv[1].lower()
    start_date = sys.argv[2]
    end_date = sys.argv[3]
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
        backtester = PortfolioDebateBacktester(
            portfolio=portfolio,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            trading_frequency="QE"  # Quarterly by default for debate system
        )
        
        # Run backtest
        performance_metrics = backtester.run_backtest()
        
        # Analyze and display results
        performance_df = backtester.analyze_performance()
        
        print(f"\n{Fore.GREEN}✅ Backtest complete!{Style.RESET_ALL}")
        
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Backtest interrupted by user{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}Error during backtest: {e}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()