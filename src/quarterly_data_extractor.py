#!/usr/bin/env python3
"""
Quarterly Data Extractor

Extracts and analyzes historical data for the past quarter to inform portfolio debates.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
# For now, we'll create mock data functions since the main src API is not directly accessible
# In a production environment, you would properly integrate with the main src API

def get_prices(ticker: str, start_date: str, end_date: str, api_key: str = None):
    """Mock function - replace with actual API call"""
    # This would normally call the main src API
    # For demo purposes, return mock data
    from dataclasses import dataclass
    
    @dataclass
    class MockPrice:
        open: float
        close: float
        high: float
        low: float
        volume: int
        time: str
    
    # Generate some mock price data for demonstration
    import random
    from datetime import datetime, timedelta
    
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    prices = []
    current_date = start
    base_price = 100.0 + random.uniform(-20, 20)  # Random starting price
    
    while current_date <= end:
        # Simulate price movement
        change = random.uniform(-0.05, 0.05)  # ±5% daily change
        base_price *= (1 + change)
        
        prices.append(MockPrice(
            open=base_price * random.uniform(0.99, 1.01),
            close=base_price,
            high=base_price * random.uniform(1.0, 1.03),
            low=base_price * random.uniform(0.97, 1.0),
            volume=random.randint(1000000, 10000000),
            time=current_date.strftime("%Y-%m-%d")
        ))
        
        current_date += timedelta(days=1)
    
    return prices

def get_financial_metrics(ticker: str, end_date: str, period: str = "ttm", limit: int = 10, api_key: str = None):
    """Mock function - replace with actual API call"""
    return []  # Return empty for demo

def get_company_news(ticker: str, end_date: str, start_date: str = None, limit: int = 1000, api_key: str = None):
    """Mock function - replace with actual API call"""
    return []  # Return empty for demo

def get_insider_trades(ticker: str, end_date: str, start_date: str = None, limit: int = 1000, api_key: str = None):
    """Mock function - replace with actual API call"""
    return []  # Return empty for demo

def get_market_cap(ticker: str, end_date: str, api_key: str = None):
    """Mock function - replace with actual API call"""
    return None

# Mock data models
class Price:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

class FinancialMetrics:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

class CompanyNews:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

class InsiderTrade:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


@dataclass
class QuarterlyPerformance:
    """Quarterly performance metrics for a stock"""
    ticker: str
    quarter_start: str
    quarter_end: str
    
    # Price performance
    start_price: float
    end_price: float
    quarterly_return: float
    volatility: float
    max_drawdown: float
    
    # Trading metrics
    avg_volume: float
    volume_change: float
    
    # Financial metrics comparison
    financial_metrics_start: Optional[FinancialMetrics] = None
    financial_metrics_end: Optional[FinancialMetrics] = None
    
    # News and sentiment
    news_count: int = 0
    positive_news_ratio: float = 0.0
    major_events: List[str] = None
    
    # Insider activity
    insider_trades_count: int = 0
    net_insider_buying: float = 0.0
    
    def __post_init__(self):
        if self.major_events is None:
            self.major_events = []


@dataclass
class QuarterlyPortfolioContext:
    """Complete quarterly context for portfolio debate"""
    quarter_start: str
    quarter_end: str
    individual_performance: Dict[str, QuarterlyPerformance]
    market_context: Dict[str, float]  # SPY, QQQ benchmarks
    sector_performance: Dict[str, float]
    
    # Portfolio-level metrics
    portfolio_return: float = 0.0
    portfolio_volatility: float = 0.0
    portfolio_sharpe: float = 0.0


class QuarterlyDataExtractor:
    """Extracts and analyzes quarterly data for portfolio debates"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
    
    def get_quarter_dates(self, reference_date: str = None) -> Tuple[str, str]:
        """
        Get the start and end dates of the most recent completed quarter
        
        Args:
            reference_date: Reference date (YYYY-MM-DD), defaults to today
            
        Returns:
            Tuple of (quarter_start, quarter_end) in YYYY-MM-DD format
        """
        if reference_date:
            ref_date = datetime.strptime(reference_date, "%Y-%m-%d")
        else:
            ref_date = datetime.now()
        
        # Determine the most recent completed quarter
        current_month = ref_date.month
        current_year = ref_date.year
        
        if current_month <= 3:  # Q1 just ended or in progress
            quarter_start = datetime(current_year - 1, 10, 1)  # Previous Q4
            quarter_end = datetime(current_year - 1, 12, 31)
        elif current_month <= 6:  # Q2 just ended or in progress
            quarter_start = datetime(current_year, 1, 1)  # Q1
            quarter_end = datetime(current_year, 3, 31)
        elif current_month <= 9:  # Q3 just ended or in progress
            quarter_start = datetime(current_year, 4, 1)  # Q2
            quarter_end = datetime(current_year, 6, 30)
        else:  # Q4 just ended or in progress
            quarter_start = datetime(current_year, 7, 1)  # Q3
            quarter_end = datetime(current_year, 9, 30)
        
        return quarter_start.strftime("%Y-%m-%d"), quarter_end.strftime("%Y-%m-%d")
    
    def calculate_performance_metrics(self, prices: List[Price]) -> Dict[str, float]:
        """Calculate performance metrics from price data"""
        if not prices or len(prices) < 2:
            return {
                "return": 0.0,
                "volatility": 0.0,
                "max_drawdown": 0.0,
                "avg_volume": 0.0
            }
        
        # Convert to DataFrame for easier calculation
        df = pd.DataFrame([{
            'close': p.close,
            'volume': p.volume,
            'date': p.time
        } for p in prices])
        
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # Calculate returns
        start_price = df['close'].iloc[0]
        end_price = df['close'].iloc[-1]
        quarterly_return = (end_price - start_price) / start_price
        
        # Calculate daily returns for volatility
        df['daily_return'] = df['close'].pct_change()
        volatility = df['daily_return'].std() * np.sqrt(252)  # Annualized
        
        # Calculate max drawdown
        df['cumulative'] = (1 + df['daily_return']).cumprod()
        df['running_max'] = df['cumulative'].expanding().max()
        df['drawdown'] = (df['cumulative'] - df['running_max']) / df['running_max']
        max_drawdown = df['drawdown'].min()
        
        # Average volume
        avg_volume = df['volume'].mean()
        
        return {
            "return": quarterly_return,
            "volatility": volatility,
            "max_drawdown": abs(max_drawdown),
            "avg_volume": avg_volume
        }
    
    def analyze_news_sentiment(self, news: List[CompanyNews]) -> Dict[str, any]:
        """Analyze news sentiment and extract major events"""
        if not news:
            return {
                "count": 0,
                "positive_ratio": 0.0,
                "major_events": []
            }
        
        # Count sentiment
        positive_count = sum(1 for n in news if n.sentiment and n.sentiment.lower() in ['positive', 'bullish'])
        total_with_sentiment = sum(1 for n in news if n.sentiment)
        
        positive_ratio = positive_count / total_with_sentiment if total_with_sentiment > 0 else 0.0
        
        # Extract major events (simplified - look for keywords)
        major_event_keywords = [
            'earnings', 'acquisition', 'merger', 'lawsuit', 'fda approval',
            'partnership', 'breakthrough', 'recall', 'investigation', 'ceo'
        ]
        
        major_events = []
        for article in news[:10]:  # Top 10 most recent
            title_lower = article.title.lower()
            if any(keyword in title_lower for keyword in major_event_keywords):
                major_events.append(f"{article.date[:10]}: {article.title}")
        
        return {
            "count": len(news),
            "positive_ratio": positive_ratio,
            "major_events": major_events[:5]  # Top 5 events
        }
    
    def analyze_insider_activity(self, trades: List[InsiderTrade]) -> Dict[str, float]:
        """Analyze insider trading activity"""
        if not trades:
            return {
                "count": 0,
                "net_buying": 0.0
            }
        
        net_buying = 0.0
        for trade in trades:
            if trade.transaction_value:
                # Positive for purchases, negative for sales
                if trade.transaction_shares and trade.transaction_shares > 0:
                    net_buying += trade.transaction_value
                else:
                    net_buying -= abs(trade.transaction_value or 0)
        
        return {
            "count": len(trades),
            "net_buying": net_buying
        }
    
    def extract_quarterly_performance(self, ticker: str, quarter_start: str, quarter_end: str) -> QuarterlyPerformance:
        """Extract comprehensive quarterly performance for a single ticker"""
        
        # Get price data
        try:
            prices = get_prices(ticker, quarter_start, quarter_end, self.api_key)
            performance_metrics = self.calculate_performance_metrics(prices)
        except Exception as e:
            print(f"Warning: Could not fetch price data for {ticker}: {e}")
            performance_metrics = {
                "return": 0.0,
                "volatility": 0.0,
                "max_drawdown": 0.0,
                "avg_volume": 0.0
            }
            prices = []
        
        # Get financial metrics at start and end of quarter
        financial_start = None
        financial_end = None
        try:
            financial_start_data = get_financial_metrics(ticker, quarter_start, api_key=self.api_key)
            financial_start = financial_start_data[0] if financial_start_data else None
            
            financial_end_data = get_financial_metrics(ticker, quarter_end, api_key=self.api_key)
            financial_end = financial_end_data[0] if financial_end_data else None
        except Exception as e:
            print(f"Warning: Could not fetch financial metrics for {ticker}: {e}")
        
        # Get news data
        news_analysis = {"count": 0, "positive_ratio": 0.0, "major_events": []}
        try:
            news = get_company_news(ticker, quarter_end, quarter_start, api_key=self.api_key)
            news_analysis = self.analyze_news_sentiment(news)
        except Exception as e:
            print(f"Warning: Could not fetch news for {ticker}: {e}")
        
        # Get insider trades
        insider_analysis = {"count": 0, "net_buying": 0.0}
        try:
            insider_trades = get_insider_trades(ticker, quarter_end, quarter_start, api_key=self.api_key)
            insider_analysis = self.analyze_insider_activity(insider_trades)
        except Exception as e:
            print(f"Warning: Could not fetch insider trades for {ticker}: {e}")
        
        # Calculate volume change (compare to previous quarter)
        volume_change = 0.0
        if prices and len(prices) > 10:
            recent_volume = np.mean([p.volume for p in prices[-10:]])
            early_volume = np.mean([p.volume for p in prices[:10]])
            if early_volume > 0:
                volume_change = (recent_volume - early_volume) / early_volume
        
        return QuarterlyPerformance(
            ticker=ticker,
            quarter_start=quarter_start,
            quarter_end=quarter_end,
            start_price=prices[0].close if prices else 0.0,
            end_price=prices[-1].close if prices else 0.0,
            quarterly_return=performance_metrics["return"],
            volatility=performance_metrics["volatility"],
            max_drawdown=performance_metrics["max_drawdown"],
            avg_volume=performance_metrics["avg_volume"],
            volume_change=volume_change,
            financial_metrics_start=financial_start,
            financial_metrics_end=financial_end,
            news_count=news_analysis["count"],
            positive_news_ratio=news_analysis["positive_ratio"],
            major_events=news_analysis["major_events"],
            insider_trades_count=insider_analysis["count"],
            net_insider_buying=insider_analysis["net_buying"]
        )
    
    def extract_market_context(self, quarter_start: str, quarter_end: str) -> Dict[str, float]:
        """Extract market benchmark performance for context"""
        benchmarks = ["SPY", "QQQ", "IWM"]  # S&P 500, NASDAQ, Russell 2000
        market_context = {}
        
        for benchmark in benchmarks:
            try:
                prices = get_prices(benchmark, quarter_start, quarter_end, self.api_key)
                if prices and len(prices) >= 2:
                    start_price = prices[0].close
                    end_price = prices[-1].close
                    benchmark_return = (end_price - start_price) / start_price
                    market_context[benchmark] = benchmark_return
                else:
                    market_context[benchmark] = 0.0
            except Exception as e:
                print(f"Warning: Could not fetch benchmark data for {benchmark}: {e}")
                market_context[benchmark] = 0.0
        
        return market_context
    
    def extract_quarterly_context(self, tickers: List[str], reference_date: str = None) -> QuarterlyPortfolioContext:
        """
        Extract comprehensive quarterly context for all portfolio holdings
        
        Args:
            tickers: List of ticker symbols
            reference_date: Reference date for quarter calculation
            
        Returns:
            QuarterlyPortfolioContext with all quarterly data
        """
        quarter_start, quarter_end = self.get_quarter_dates(reference_date)
        
        print(f"Extracting quarterly data for period: {quarter_start} to {quarter_end}")
        
        # Extract individual stock performance
        individual_performance = {}
        for ticker in tickers:
            print(f"Processing {ticker}...")
            performance = self.extract_quarterly_performance(ticker, quarter_start, quarter_end)
            individual_performance[ticker] = performance
        
        # Extract market context
        print("Extracting market benchmarks...")
        market_context = self.extract_market_context(quarter_start, quarter_end)
        
        # Calculate portfolio-level metrics (simplified equal weighting)
        portfolio_return = np.mean([perf.quarterly_return for perf in individual_performance.values()])
        portfolio_volatility = np.mean([perf.volatility for perf in individual_performance.values()])
        
        # Simple Sharpe ratio approximation (assuming 2% risk-free rate)
        portfolio_sharpe = (portfolio_return - 0.02) / portfolio_volatility if portfolio_volatility > 0 else 0.0
        
        return QuarterlyPortfolioContext(
            quarter_start=quarter_start,
            quarter_end=quarter_end,
            individual_performance=individual_performance,
            market_context=market_context,
            sector_performance={},  # Could be enhanced with sector data
            portfolio_return=portfolio_return,
            portfolio_volatility=portfolio_volatility,
            portfolio_sharpe=portfolio_sharpe
        )
    
    def format_quarterly_summary(self, context: QuarterlyPortfolioContext) -> str:
        """Format quarterly context into a readable summary for debate agents"""
        
        summary = f"""
QUARTERLY PERFORMANCE SUMMARY ({context.quarter_start} to {context.quarter_end})

PORTFOLIO OVERVIEW:
• Portfolio Return: {context.portfolio_return:.2%}
• Portfolio Volatility: {context.portfolio_volatility:.2%}
• Portfolio Sharpe Ratio: {context.portfolio_sharpe:.2f}

MARKET BENCHMARKS:
"""
        
        for benchmark, return_val in context.market_context.items():
            summary += f"• {benchmark}: {return_val:.2%}\n"
        
        summary += "\nINDIVIDUAL STOCK PERFORMANCE:\n"
        
        # Sort by quarterly return
        sorted_performance = sorted(
            context.individual_performance.items(),
            key=lambda x: x[1].quarterly_return,
            reverse=True
        )
        
        for ticker, perf in sorted_performance:
            summary += f"""
{ticker}:
  • Return: {perf.quarterly_return:.2%}
  • Volatility: {perf.volatility:.2%}
  • Max Drawdown: {perf.max_drawdown:.2%}
  • News Articles: {perf.news_count} (Positive: {perf.positive_news_ratio:.1%})
  • Insider Net Buying: ${perf.net_insider_buying:,.0f}
"""
            
            if perf.major_events:
                summary += "  • Major Events:\n"
                for event in perf.major_events:
                    summary += f"    - {event}\n"
        
        return summary


if __name__ == "__main__":
    # Example usage
    extractor = QuarterlyDataExtractor()
    
    # Test with sample tickers
    test_tickers = ["AAPL", "TSLA", "NVDA"]
    context = extractor.extract_quarterly_context(test_tickers)
    
    print(extractor.format_quarterly_summary(context))
