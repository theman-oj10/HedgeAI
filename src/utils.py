"""
Simplified utilities for investment agents
"""
import json
import os
from typing import Any, Dict, List, Optional

import httpx
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

# Load environment variables
load_dotenv()


def get_llm_client(provider: str = None, model: str = None):
    """Get LLM client based on provider"""
    provider = provider or os.getenv("DEFAULT_LLM_PROVIDER", "openai")
    model = model or os.getenv("DEFAULT_MODEL", "gpt-4o-mini")
    
    if provider == "openai":
        return ChatOpenAI(
            model=model,
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.1
        )
    elif provider == "anthropic":
        return ChatAnthropic(
            model=model,
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            temperature=0.1
        )
    elif provider == "cerebras":
        
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


def call_llm(prompt, pydantic_model: BaseModel = None, **kwargs):
    """Simplified LLM calling function with demo mode fallback"""
    # Check if we're in demo mode (no real API keys)
    openai_key = os.getenv("OPENAI_API_KEY", "")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
    
    if (not openai_key or openai_key.startswith("your_")) and (not anthropic_key or anthropic_key.startswith("your_")):
        return _demo_mode_response(pydantic_model, **kwargs)
    
    try:
        llm = get_llm_client()
        
        if pydantic_model:
            structured_llm = llm.with_structured_output(pydantic_model)
            return structured_llm.invoke(prompt)
        else:
            return llm.invoke(prompt)
    except Exception as e:
        print(f"Error calling LLM: {e}")
        # Return default instance
        if pydantic_model:
            return pydantic_model(
                signal="neutral", 
                confidence=0.0, 
                reasoning="Error in analysis, defaulting to neutral"
            )
        return None


def _demo_mode_response(pydantic_model: BaseModel = None, **kwargs):
    """Generate demo responses when API keys are not available"""
    if pydantic_model and hasattr(pydantic_model, '__name__'):
        model_name = pydantic_model.__name__
        
        if "WarrenBuffett" in model_name:
            return pydantic_model(
                signal="bullish",
                confidence=78.5,
                reasoning="This appears to be a solid business with consistent earnings and a reasonable valuation. The company shows the kind of predictable cash flows I look for, though I'd want to understand the competitive moat better. The debt levels are manageable and management appears to be allocating capital sensibly. While not a slam dunk, it meets many of my criteria for a long-term holding. I'd be comfortable owning this at current prices."
            )
        elif "CathieWood" in model_name:
            return pydantic_model(
                signal="neutral",
                confidence=65.2,
                reasoning="While this company operates in an interesting space, I'm not seeing the exponential growth trajectory and disruptive innovation that typically drives our investment thesis. The R&D spending is moderate but not at the levels we'd expect for a truly transformative technology company. The market opportunity is significant, but the company's current approach seems more incremental than revolutionary. We'd need to see stronger evidence of platform effects and network advantages before taking a position."
            )
    
    return pydantic_model(
        signal="neutral",
        confidence=50.0,
        reasoning="Demo mode analysis - please configure API keys for full functionality"
    ) if pydantic_model else None


def get_financial_metrics(ticker: str, end_date: str, period: str = "ttm", limit: int = 5, api_key: str = None) -> List[Dict]:
    """
    Simplified financial metrics fetcher
    In a real implementation, this would call the Financial Datasets API
    For now, returns mock data for demonstration
    """
    # Mock data structure similar to what the real API would return
    mock_metrics = []
    for i in range(limit):
        mock_metrics.append({
            "ticker": ticker,
            "period": period,
            "date": end_date,
            "return_on_equity": 15.2 + i * 0.5,
            "debt_to_equity": 0.3 - i * 0.02,
            "current_ratio": 2.1 + i * 0.1,
            "gross_margin": 0.45 + i * 0.01,
            "operating_margin": 0.25 + i * 0.005,
            "net_margin": 0.15 + i * 0.003,
            "revenue_growth": 0.12 - i * 0.01,
            "earnings_growth": 0.18 - i * 0.015
        })
    return mock_metrics


def get_market_cap(ticker: str, end_date: str, api_key: str = None) -> float:
    """
    Get market capitalization
    Returns mock data for demonstration
    """
    # Mock market cap data
    mock_market_caps = {
        "AAPL": 3000000000000,  # $3T
        "MSFT": 2800000000000,  # $2.8T
        "GOOGL": 1800000000000, # $1.8T
        "TSLA": 800000000000,   # $800B
        "NVDA": 2200000000000,  # $2.2T
    }
    return mock_market_caps.get(ticker, 100000000000)  # Default $100B


def search_line_items(ticker: str, line_items: List[str], end_date: str, period: str = "ttm", limit: int = 5, api_key: str = None) -> List[Dict]:
    """
    Search for specific financial line items
    Returns mock data for demonstration
    """
    mock_line_items = []
    for i in range(limit):
        item_data = {
            "ticker": ticker,
            "period": period,
            "date": end_date,
        }
        
        # Add mock values for requested line items
        for item in line_items:
            if item == "revenue":
                item_data[item] = 100000000000 + i * 5000000000  # Growing revenue
            elif item == "net_income":
                item_data[item] = 15000000000 + i * 1000000000   # Growing earnings
            elif item == "free_cash_flow":
                item_data[item] = 20000000000 + i * 1500000000   # Growing FCF
            elif item == "capital_expenditure":
                item_data[item] = 8000000000 + i * 500000000     # Growing capex
            elif item == "research_and_development":
                item_data[item] = 12000000000 + i * 800000000    # Growing R&D
            elif item == "outstanding_shares":
                item_data[item] = 16000000000 - i * 100000000    # Share buybacks
            elif item == "shareholders_equity":
                item_data[item] = 80000000000 + i * 5000000000   # Growing equity
            elif item == "total_assets":
                item_data[item] = 200000000000 + i * 10000000000 # Growing assets
            elif item == "total_liabilities":
                item_data[item] = 120000000000 + i * 6000000000  # Growing liabilities
            else:
                item_data[item] = 1000000000 + i * 100000000     # Default value
        
        mock_line_items.append(item_data)
    
    return mock_line_items


class AnalysisResult:
    """Simple container for analysis results"""
    def __init__(self, score: float, details: str, **kwargs):
        self.score = score
        self.details = details
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def to_dict(self):
        return {
            "score": self.score,
            "details": self.details,
            **{k: v for k, v in self.__dict__.items() if k not in ["score", "details"]}
        }


def print_analysis_result(ticker: str, agent_name: str, result: Dict[str, Any]):
    """Pretty print analysis results"""
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    
    console = Console()
    
    signal_color = {
        "bullish": "green",
        "bearish": "red", 
        "neutral": "yellow"
    }.get(result.get("signal", "neutral"), "white")
    
    content = Text()
    content.append(f"Signal: ", style="bold")
    content.append(f"{result.get('signal', 'N/A').upper()}", style=f"bold {signal_color}")
    content.append(f"\nConfidence: {result.get('confidence', 0):.1f}%\n\n", style="bold")
    content.append(f"Reasoning:\n{result.get('reasoning', 'No reasoning provided')}")
    
    console.print(Panel(
        content,
        title=f"[bold blue]{agent_name}[/bold blue] Analysis for [bold cyan]{ticker}[/bold cyan]",
        border_style="blue"
    ))
