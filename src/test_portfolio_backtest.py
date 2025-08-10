#!/usr/bin/env python3
"""
Test Script for Portfolio Backtester with Debate System

This script demonstrates running a 1-year backtest with quarterly rebalancing.
"""

import sys
from datetime import datetime
from dateutil.relativedelta import relativedelta
from portfolio_backtester import PortfolioDebateBacktester
from portfolio_config import SAMPLE_PORTFOLIOS
from colorama import Fore, Style, init

init(autoreset=True)


def run_year_backtest(portfolio_name="tech_growth", initial_capital=100000):
    """
    Run a 1-year backtest with quarterly rebalancing
    """
    print("=" * 80)
    print(f"{Fore.CYAN}PORTFOLIO DEBATE BACKTESTER - 1 YEAR TEST{Style.RESET_ALL}")
    print("=" * 80)
    
    # Calculate dates: 1 year ago to today
    today = datetime.now()
    one_year_ago = today - relativedelta(years=1)
    start_date = one_year_ago.strftime("%Y-%m-%d")
    end_date = today.strftime("%Y-%m-%d")
    
    # Get portfolio
    if portfolio_name not in SAMPLE_PORTFOLIOS:
        print(f"{Fore.RED}Error: Unknown portfolio '{portfolio_name}'{Style.RESET_ALL}")
        print(f"Available portfolios: {', '.join(SAMPLE_PORTFOLIOS.keys())}")
        return
    
    portfolio = SAMPLE_PORTFOLIOS[portfolio_name]
    
    print(f"\n📊 Portfolio: {portfolio.name}")
    print(f"📅 Period: {start_date} to {end_date} (1 year)")
    print(f"💰 Initial Capital: ${initial_capital:,.2f}")
    print("🔄 Rebalancing: Quarterly (every 3 months)")
    
    print("\nHoldings:")
    for holding in portfolio.holdings:
        print(f"  - {holding.ticker}: {holding.weight:.1%}")
    print(f"  - CASH: {portfolio.cash_weight:.1%}")
    
    print("\n" + "-" * 80)
    print(f"{Fore.YELLOW}Starting backtest with quarterly rebalancing...{Style.RESET_ALL}")
    print("-" * 80)
    
    try:
        # Create backtester with quarterly rebalancing
        backtester = PortfolioDebateBacktester(
            portfolio=portfolio,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            trading_frequency="QE",  # Quarterly end rebalancing
            rebalance_threshold=0.05  # 5% threshold for rebalancing
        )
        
        # Run the backtest
        performance_metrics = backtester.run_backtest()
        
        # Analyze and display results
        performance_df = backtester.analyze_performance()
        
        print("\n" + "=" * 80)
        print(f"{Fore.GREEN}📈 BACKTEST COMPLETE - PERFORMANCE METRICS{Style.RESET_ALL}")
        print("=" * 80)
        
        if performance_metrics["sharpe_ratio"] is not None:
            print(f"\nSharpe Ratio: {performance_metrics['sharpe_ratio']:.2f}")
        if performance_metrics["max_drawdown"] is not None:
            print(f"Maximum Drawdown: {performance_metrics['max_drawdown']:.2f}%")
        print(f"Total Trades: {performance_metrics['total_trades']}")
        print(f"Winning Trades: {performance_metrics['winning_trades']}")
        print(f"Losing Trades: {performance_metrics['losing_trades']}")
        
        # Calculate quarterly rebalancing dates for reference
        print(f"\n{Fore.CYAN}Quarterly Rebalancing Dates:{Style.RESET_ALL}")
        current = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        rebalance_count = 0
        while current <= end:
            current = current + relativedelta(months=3)
            if current <= end:
                rebalance_count += 1
                print(f"  Quarter {rebalance_count}: ~{current.strftime('%Y-%m-%d')}")
        
        print(f"\n{Fore.GREEN}✅ Successfully completed 1-year backtest with {rebalance_count} quarterly rebalancings{Style.RESET_ALL}")
        
        return performance_df, performance_metrics
        
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Backtest interrupted by user{Style.RESET_ALL}")
    except Exception as e:
        print(f"\n{Fore.RED}Error during backtest: {e}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()
        return None, None


def main():
    """Main entry point for the test script"""
    
    # Parse command line arguments
    portfolio_name = sys.argv[1] if len(sys.argv) > 1 else "tech_growth"
    initial_capital = float(sys.argv[2]) if len(sys.argv) > 2 else 100000
    
    # Run the 1-year backtest
    run_year_backtest(portfolio_name, initial_capital)


if __name__ == "__main__":
    main()