"""
Portfolio Configuration Module

Defines portfolio structures and management for investment analysis.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, validator


class PortfolioHolding(BaseModel):
    """Individual holding in a portfolio"""
    ticker: str = Field(..., description="Stock ticker symbol")
    weight: float = Field(..., ge=0.0, le=1.0, description="Portfolio weight (0.0 to 1.0)")
    current_price: Optional[float] = Field(None, description="Current stock price")
    shares: Optional[int] = Field(None, description="Number of shares held")


class Portfolio(BaseModel):
    """Portfolio configuration with holdings and metadata"""
    name: str = Field(..., description="Portfolio name")
    holdings: List[PortfolioHolding] = Field(..., description="List of portfolio holdings")
    cash_weight: float = Field(0.0, ge=0.0, le=1.0, description="Cash allocation weight")
    rebalance_threshold: float = Field(0.05, ge=0.01, le=0.5, description="Rebalancing threshold")
    
    @validator('holdings')
    def validate_holdings(cls, v):
        if not v:
            raise ValueError("Portfolio must have at least one holding")
        return v
    
    @validator('cash_weight')
    def validate_total_weights(cls, v, values):
        if 'holdings' in values:
            total_weight = sum(holding.weight for holding in values['holdings']) + v
            if abs(total_weight - 1.0) > 0.001:  # Allow small floating point errors
                raise ValueError(f"Total portfolio weights must sum to 1.0, got {total_weight}")
        return v
    
    def get_tickers(self) -> List[str]:
        """Get list of all tickers in portfolio"""
        return [holding.ticker for holding in self.holdings]
    
    def get_weight(self, ticker: str) -> float:
        """Get weight for specific ticker"""
        for holding in self.holdings:
            if holding.ticker == ticker:
                return holding.weight
        return 0.0


# Predefined portfolio configurations
SAMPLE_PORTFOLIOS = {
    "tech_growth": Portfolio(
        name="Tech Growth Portfolio",
        holdings=[
            PortfolioHolding(ticker="AAPL", weight=0.02),
            PortfolioHolding(ticker="MSFT", weight=0.02),
            PortfolioHolding(ticker="GOOGL", weight=0.06),
            PortfolioHolding(ticker="TSLA", weight=0.90),
            PortfolioHolding(ticker="NVDA", weight=0.0),
        ],
        cash_weight=0.0
    ),
    
    "value_dividend": Portfolio(
        name="Value & Dividend Portfolio",
        holdings=[
            PortfolioHolding(ticker="BRK.B", weight=0.20),
            PortfolioHolding(ticker="JNJ", weight=0.15),
            PortfolioHolding(ticker="PG", weight=0.15),
            PortfolioHolding(ticker="KO", weight=0.10),
            PortfolioHolding(ticker="JPM", weight=0.15),
            PortfolioHolding(ticker="WMT", weight=0.10),
        ],
        cash_weight=0.15
    ),
    
    "balanced_mix": Portfolio(
        name="Balanced Growth & Value",
        holdings=[
            PortfolioHolding(ticker="AAPL", weight=0.15),
            PortfolioHolding(ticker="MSFT", weight=0.15),
            PortfolioHolding(ticker="BRK.B", weight=0.15),
            PortfolioHolding(ticker="JNJ", weight=0.10),
            PortfolioHolding(ticker="TSLA", weight=0.10),
            PortfolioHolding(ticker="NVDA", weight=0.10),
            PortfolioHolding(ticker="JPM", weight=0.10),
        ],
        cash_weight=0.15
    ),
    
    "ai_innovation": Portfolio(
        name="AI & Innovation Focus",
        holdings=[
            PortfolioHolding(ticker="NVDA", weight=0.25),
            PortfolioHolding(ticker="MSFT", weight=0.20),
            PortfolioHolding(ticker="GOOGL", weight=0.15),
            PortfolioHolding(ticker="TSLA", weight=0.15),
            PortfolioHolding(ticker="AMD", weight=0.10),
        ],
        cash_weight=0.15
    )
}


def create_custom_portfolio(name: str, holdings_dict: Dict[str, float], cash_weight: float = 0.0) -> Portfolio:
    """
    Create a custom portfolio from a dictionary of ticker: weight pairs
    
    Args:
        name: Portfolio name
        holdings_dict: Dictionary of {ticker: weight} pairs
        cash_weight: Cash allocation weight
    
    Returns:
        Portfolio object
    
    Example:
        portfolio = create_custom_portfolio(
            "My Portfolio",
            {"AAPL": 0.3, "TSLA": 0.4, "NVDA": 0.2},
            cash_weight=0.1
        )
    """
    holdings = [
        PortfolioHolding(ticker=ticker, weight=weight)
        for ticker, weight in holdings_dict.items()
    ]
    
    return Portfolio(
        name=name,
        holdings=holdings,
        cash_weight=cash_weight
    )


def load_portfolio_from_file(filepath: str) -> Portfolio:
    """Load portfolio configuration from JSON file"""
    import json
    with open(filepath, 'r') as f:
        data = json.load(f)
    return Portfolio(**data)


def save_portfolio_to_file(portfolio: Portfolio, filepath: str):
    """Save portfolio configuration to JSON file"""
    import json
    with open(filepath, 'w') as f:
        json.dump(portfolio.model_dump(), f, indent=2)
