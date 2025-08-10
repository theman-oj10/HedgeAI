#!/usr/bin/env python3
"""
Simple test script for JSON trading actions output
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from portfolio_debate import PortfolioDebateSystem
from portfolio_config import create_custom_portfolio

def test_json_output():
    """Test the JSON output functionality with a simple portfolio"""
    
    # Create a simple AAPL portfolio
    portfolio = create_custom_portfolio([("AAPL", 1.0)])
    
    # Create portfolio debate system
    portfolio_system = PortfolioDebateSystem()
    
    try:
        # Run the analysis
        print("Starting portfolio analysis...")
        result, json_output = portfolio_system.analyze_portfolio(portfolio)
        
        print("\n" + "="*50)
        print("JSON OUTPUT:")
        print("="*50)
        print(json_output)
        
        print("\n" + "="*50)
        print("SUCCESS: JSON output generated successfully!")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_json_output()
