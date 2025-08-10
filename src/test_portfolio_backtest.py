#!/usr/bin/env python3
"""
Test Script for Portfolio Backtester with Debate System

This script demonstrates how the portfolio backtester integrates with
the portfolio debate system to generate trading signals.
"""

import json
from datetime import datetime
from portfolio_debate import PortfolioDebateSystem
from portfolio_config import SAMPLE_PORTFOLIOS


def demonstrate_portfolio_analysis():
    """
    Demonstrate the portfolio analysis and JSON output generation
    """
    print("=" * 80)
    print("PORTFOLIO DEBATE BACKTESTER - DEMONSTRATION")
    print("=" * 80)
    
    # Select a sample portfolio
    portfolio = SAMPLE_PORTFOLIOS["tech_growth"]
    
    print(f"\n📊 Selected Portfolio: {portfolio.name}")
    print(f"Holdings:")
    for holding in portfolio.holdings:
        print(f"  - {holding.ticker}: {holding.weight:.1%}")
    print(f"  - CASH: {portfolio.cash_weight:.1%}")
    
    # Initialize the debate system
    debate_system = PortfolioDebateSystem()
    
    # Set a test date
    test_date = "2024-12-31"
    
    print(f"\n🎯 Running portfolio analysis for date: {test_date}")
    print("This will generate trading signals based on agent debates...")
    print("-" * 80)
    
    try:
        # Run the portfolio analysis
        result, json_output = debate_system.analyze_portfolio(
            portfolio, 
            end_date=test_date,
            parallel=False  # Set to False for demonstration
        )
        
        # Parse and display the JSON output
        trading_data = json.loads(json_output)
        
        print("\n" + "=" * 80)
        print("📄 TRADING ACTIONS GENERATED FROM DEBATE SYSTEM")
        print("=" * 80)
        
        print(f"\nPortfolio: {trading_data['portfolio_name']}")
        print(f"Timestamp: {trading_data['analysis_timestamp']}")
        print(f"Total Holdings: {trading_data['total_holdings']}")
        print(f"Rebalancing Recommended: {trading_data['rebalancing_recommended']}")
        
        print("\n📊 Individual Stock Actions:")
        print("-" * 80)
        
        for action in trading_data['trading_actions']:
            ticker = action['ticker']
            if ticker == "CASH":
                continue
                
            print(f"\n{ticker}:")
            print(f"  Current Weight: {action['current_weight']:.2f}%")
            print(f"  Suggested Weight: {action['suggested_weight']:.2f}%")
            print(f"  Weight Change: {action['weight_change']:+.2f}%")
            print(f"  Percentage Change: {action['percentage_change']:+.2f}%")
            print(f"  Action: {action['action'].upper()}")
            print(f"  Signal: {action['signal'].upper()}")
            print(f"  Confidence: {action['confidence']:.1f}%")
            print(f"  Reasoning: {action['reasoning'][:100]}...")
        
        print("\n📈 Portfolio Summary:")
        summary = trading_data['portfolio_summary']
        print(f"  Overall Signal: {summary['overall_signal'].upper()}")
        print(f"  Weighted Confidence: {summary['weighted_confidence']:.1f}%")
        print(f"  Total Weight Changes: {summary['total_weight_changes']:.2f}%")
        
        print("\n" + "=" * 80)
        print("💡 HOW THE BACKTESTER USES THIS OUTPUT:")
        print("=" * 80)
        
        print("""
The portfolio_backtester.py script uses this JSON output to:

1. Calculate Trade Quantities:
   - Takes the percentage_change from each trading action
   - Multiplies by current portfolio value to get dollar amount
   - Divides by current stock price to get number of shares

2. Execute Trades:
   - If action is "buy": Purchase calculated shares (if cash available)
   - If action is "sell": Sell calculated shares (if owned)
   - If action is "hold": No trade executed

3. Track Performance:
   - Updates portfolio positions after each trade
   - Records realized gains/losses on sells
   - Calculates new portfolio value

Example Calculation:
   Portfolio Value: $100,000
   AAPL percentage_change: +10%
   Dollar amount to trade: $100,000 * 10% = $10,000
   If AAPL price = $150, shares to buy = $10,000 / $150 = 66 shares
        """)
        
        # Save the JSON output for reference
        output_file = f"demo_trading_actions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            f.write(json_output)
        print(f"\n💾 Full JSON output saved to: {output_file}")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    demonstrate_portfolio_analysis()