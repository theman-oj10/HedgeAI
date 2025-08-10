"""
Simplified Warren Buffett Investment Agent
"""
from pydantic import BaseModel
from typing_extensions import Literal
from langchain_core.prompts import ChatPromptTemplate
import json
import sys
import os
# Add parent directory to path for utils import (portable approach)
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from utils import get_financial_metrics, get_market_cap, search_line_items, call_llm, AnalysisResult
except ImportError:
    # Fallback for when running as module
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils import get_financial_metrics, get_market_cap, search_line_items, call_llm, AnalysisResult


class WarrenBuffettSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str
    suggested_weight: float = None  # Suggested portfolio weight (0.0 to 1.0)
    weight_reasoning: str = None    # Reasoning for the weight suggestion


class WarrenBuffettAgent:
    """Simplified Warren Buffett investment agent"""
    
    def __init__(self):
        self.agent_name = "Warren Buffett"
    
    def analyze_stock(self, ticker: str, end_date: str = "2024-12-31", current_weight: float = None, portfolio_context: dict = None) -> WarrenBuffettSignal:
        """Analyze a stock using Buffett's investment principles"""
        
        # Fetch financial data
        metrics = get_financial_metrics(ticker, end_date, period="ttm", limit=5)
        financial_line_items = search_line_items(
            ticker,
            [
                "capital_expenditure",
                "depreciation_and_amortization", 
                "net_income",
                "outstanding_shares",
                "total_assets",
                "total_liabilities",
                "shareholders_equity",
                "dividends_and_other_cash_distributions",
                "issuance_or_purchase_of_equity_shares",
                "gross_profit",
                "revenue",
                "free_cash_flow",
                "research_and_development"
            ],
            end_date,
            period="ttm",
            limit=5
        )
        market_cap = get_market_cap(ticker, end_date)
        
        # Perform Buffett-style analysis
        fundamental_analysis = self._analyze_fundamentals(metrics)
        consistency_analysis = self._analyze_consistency(financial_line_items)
        moat_analysis = self._analyze_moat(metrics)
        management_analysis = self._analyze_management_quality(financial_line_items)
        intrinsic_value_analysis = self._calculate_intrinsic_value(financial_line_items, market_cap)
        
        # Compile analysis data
        analysis_data = {
            "ticker": ticker,
            "market_cap": market_cap,
            "fundamental_analysis": fundamental_analysis.to_dict(),
            "consistency_analysis": consistency_analysis.to_dict(),
            "moat_analysis": moat_analysis.to_dict(),
            "management_analysis": management_analysis.to_dict(),
            "intrinsic_value_analysis": intrinsic_value_analysis.to_dict(),
            "recent_metrics": metrics[:2] if metrics else [],
            "recent_financials": financial_line_items[:2] if financial_line_items else [],
            "current_weight": current_weight,
            "portfolio_context": portfolio_context
        }
        
        # Generate final investment decision using LLM
        return self._generate_buffett_decision(ticker, analysis_data, current_weight, portfolio_context)
    
    def _analyze_fundamentals(self, metrics) -> AnalysisResult:
        """Analyze fundamental metrics like ROE, debt levels, margins"""
        if not metrics:
            return AnalysisResult(0, "No metrics available")
        
        latest = metrics[0]
        score = 0
        details = []
        
        # ROE analysis (Buffett likes ROE > 15%)
        roe = latest.get("return_on_equity", 0)
        if roe > 20:
            score += 3
            details.append(f"Excellent ROE of {roe:.1f}%")
        elif roe > 15:
            score += 2
            details.append(f"Good ROE of {roe:.1f}%")
        elif roe > 10:
            score += 1
            details.append(f"Moderate ROE of {roe:.1f}%")
        else:
            details.append(f"Low ROE of {roe:.1f}%")
        
        # Debt analysis (Buffett prefers low debt)
        debt_to_equity = latest.get("debt_to_equity", 1.0)
        if debt_to_equity < 0.3:
            score += 2
            details.append(f"Low debt-to-equity of {debt_to_equity:.2f}")
        elif debt_to_equity < 0.6:
            score += 1
            details.append(f"Moderate debt-to-equity of {debt_to_equity:.2f}")
        else:
            details.append(f"High debt-to-equity of {debt_to_equity:.2f}")
        
        # Margin analysis
        gross_margin = latest.get("gross_margin", 0)
        if gross_margin > 0.4:
            score += 2
            details.append(f"Strong gross margin of {gross_margin:.1%}")
        elif gross_margin > 0.25:
            score += 1
            details.append(f"Decent gross margin of {gross_margin:.1%}")
        
        return AnalysisResult(score, "; ".join(details), max_score=7)
    
    def _analyze_consistency(self, financial_line_items) -> AnalysisResult:
        """Analyze earnings and revenue consistency"""
        if len(financial_line_items) < 3:
            return AnalysisResult(0, "Insufficient data for consistency analysis")
        
        revenues = [item.get("revenue", 0) for item in financial_line_items]
        net_incomes = [item.get("net_income", 0) for item in financial_line_items]
        
        score = 0
        details = []
        
        # Revenue consistency
        revenue_growth_rates = []
        for i in range(1, len(revenues)):
            if revenues[i] > 0:
                growth = (revenues[i-1] - revenues[i]) / revenues[i]
                revenue_growth_rates.append(growth)
        
        if revenue_growth_rates and all(g >= 0 for g in revenue_growth_rates):
            score += 2
            details.append("Consistent revenue growth")
        elif revenue_growth_rates and sum(g >= 0 for g in revenue_growth_rates) >= len(revenue_growth_rates) * 0.7:
            score += 1
            details.append("Mostly consistent revenue")
        
        # Earnings consistency
        positive_earnings = sum(1 for income in net_incomes if income > 0)
        if positive_earnings == len(net_incomes):
            score += 2
            details.append("Consistent positive earnings")
        elif positive_earnings >= len(net_incomes) * 0.8:
            score += 1
            details.append("Mostly positive earnings")
        
        return AnalysisResult(score, "; ".join(details), max_score=4)
    
    def _analyze_moat(self, metrics) -> AnalysisResult:
        """Analyze competitive moat indicators"""
        if not metrics:
            return AnalysisResult(0, "No metrics for moat analysis")
        
        score = 0
        details = []
        
        # High and consistent ROE suggests moat
        roes = [m.get("return_on_equity", 0) for m in metrics[:3]]
        avg_roe = sum(roes) / len(roes) if roes else 0
        
        if avg_roe > 20:
            score += 3
            details.append(f"Excellent average ROE of {avg_roe:.1f}% suggests strong moat")
        elif avg_roe > 15:
            score += 2
            details.append(f"Good average ROE of {avg_roe:.1f}% suggests decent moat")
        
        # Margin stability
        margins = [m.get("net_margin", 0) for m in metrics[:3]]
        if margins and max(margins) - min(margins) < 0.05:  # Stable margins
            score += 2
            details.append("Stable profit margins indicate pricing power")
        
        return AnalysisResult(score, "; ".join(details), max_score=5)
    
    def _analyze_management_quality(self, financial_line_items) -> AnalysisResult:
        """Analyze management quality through capital allocation"""
        if len(financial_line_items) < 2:
            return AnalysisResult(0, "Insufficient data for management analysis")
        
        score = 0
        details = []
        
        # Share buyback analysis
        share_changes = []
        for i in range(1, len(financial_line_items)):
            current_shares = financial_line_items[i-1].get("outstanding_shares", 0)
            previous_shares = financial_line_items[i].get("outstanding_shares", 0)
            if previous_shares > 0:
                change = (current_shares - previous_shares) / previous_shares
                share_changes.append(change)
        
        if share_changes and sum(change < 0 for change in share_changes) >= len(share_changes) * 0.7:
            score += 2
            details.append("Management returning cash through buybacks")
        
        # Dividend consistency
        dividends = [item.get("dividends_and_other_cash_distributions", 0) for item in financial_line_items]
        if dividends and all(d > 0 for d in dividends):
            score += 1
            details.append("Consistent dividend payments")
        
        return AnalysisResult(score, "; ".join(details), max_score=3)
    
    def _calculate_intrinsic_value(self, financial_line_items, market_cap) -> AnalysisResult:
        """Simple intrinsic value calculation"""
        if not financial_line_items:
            return AnalysisResult(0, "No financial data for valuation")
        
        latest = financial_line_items[0]
        fcf = latest.get("free_cash_flow", 0)
        
        if fcf <= 0:
            return AnalysisResult(0, "Negative or zero free cash flow")
        
        # Simple DCF with 10% discount rate and 3% growth
        discount_rate = 0.10
        growth_rate = 0.03
        
        # 10-year DCF
        intrinsic_value = 0
        for year in range(1, 11):
            future_fcf = fcf * ((1 + growth_rate) ** year)
            present_value = future_fcf / ((1 + discount_rate) ** year)
            intrinsic_value += present_value
        
        # Terminal value
        terminal_fcf = fcf * ((1 + growth_rate) ** 10)
        terminal_value = terminal_fcf / (discount_rate - growth_rate)
        intrinsic_value += terminal_value / ((1 + discount_rate) ** 10)
        
        margin_of_safety = (intrinsic_value - market_cap) / market_cap if market_cap > 0 else 0
        
        score = 0
        if margin_of_safety > 0.3:  # 30% margin of safety
            score = 3
            details = f"Significantly undervalued with {margin_of_safety:.1%} margin of safety"
        elif margin_of_safety > 0.1:  # 10% margin of safety
            score = 2
            details = f"Moderately undervalued with {margin_of_safety:.1%} margin of safety"
        elif margin_of_safety > -0.1:  # Fair value
            score = 1
            details = f"Fairly valued with {margin_of_safety:.1%} margin of safety"
        else:
            details = f"Overvalued with {margin_of_safety:.1%} margin of safety"
        
        return AnalysisResult(
            score, 
            details, 
            intrinsic_value=intrinsic_value,
            margin_of_safety=margin_of_safety,
            max_score=3
        )
    
    def _generate_buffett_decision(self, ticker: str, analysis_data: dict, current_weight: float = None, portfolio_context: dict = None) -> WarrenBuffettSignal:
        """Generate final investment decision using LLM with Buffett's voice"""
        
        template = ChatPromptTemplate.from_messages([
            ("system", """You are Warren Buffett, the legendary value investor. Analyze this investment opportunity using your proven principles:

MY CORE INVESTMENT PHILOSOPHY:
- Buy wonderful businesses at fair prices, not fair businesses at wonderful prices
- Look for companies with durable competitive advantages (moats)
- Invest only within my circle of competence
- Focus on long-term intrinsic value, not short-term price movements
- Prefer businesses with consistent earnings and strong management
- Seek companies with pricing power and low capital requirements

INDUSTRIES I TYPICALLY AVOID:
- Technology companies I don't understand
- Biotechnology and pharmaceuticals (too unpredictable)
- Airlines and commodities (poor economics)
- Cryptocurrency and fintech speculation
- Complex derivatives or financial instruments

MY INVESTMENT CRITERIA HIERARCHY:
1. Circle of Competence - If I don't understand it, I don't invest
2. Business Quality - Does it have a moat? Will it thrive in 20 years?
3. Management - Do they act in shareholders' interests?
4. Financial Strength - Consistent earnings, low debt, strong returns?
5. Valuation - Am I paying a reasonable price?

MY LANGUAGE & STYLE:
- Use folksy wisdom and simple analogies
- Reference past investments when relevant (Coca-Cola, Apple, GEICO, See's Candies)
- Be candid about what I don't understand
- Show patience - most opportunities don't meet my criteria
- Express genuine enthusiasm for truly exceptional businesses

CONFIDENCE LEVELS:
- 90-100%: Exceptional business within my circle, attractive price
- 70-89%: Good business with decent moat, fair valuation
- 50-69%: Mixed signals, need more information or better price
- 30-49%: Outside my expertise or concerning fundamentals
- 10-29%: Poor business or significantly overvalued"""),
            
            ("human", """Analyze this investment opportunity for {ticker}:

COMPREHENSIVE ANALYSIS DATA:
{analysis_data}

PORTFOLIO CONTEXT:
Current Weight: {current_weight}% of portfolio
Portfolio Information: {portfolio_info}

Please provide your investment decision in exactly this JSON format:
{{
  "signal": "bullish" | "bearish" | "neutral",
  "confidence": float between 0 and 100,
  "reasoning": "string with your detailed Warren Buffett-style analysis",
  "suggested_weight": float between 0.0 and 1.0 (suggested portfolio allocation),
  "weight_reasoning": "string explaining your portfolio weight recommendation"
}}

In your reasoning, be specific about:
1. Whether this falls within your circle of competence (CRITICAL FIRST STEP)
2. Your assessment of the business's competitive moat
3. Management quality and capital allocation
4. Financial health and consistency
5. Valuation relative to intrinsic value
6. Long-term prospects and any red flags

In your weight_reasoning, consider:
1. Current portfolio allocation vs your suggested allocation
2. Position sizing based on conviction level and risk
3. Diversification principles (avoid over-concentration)
4. How this fits with other portfolio holdings
5. Whether to increase, decrease, or maintain current weight

Remember my principles:
- "Diversification is protection against ignorance" - but concentration in great businesses is fine
- Never put more than 20-25% in any single position unless extraordinary conviction
- Size positions based on opportunity quality and downside protection
- "It's far better to buy a wonderful company at a fair price than a fair company at a wonderful price"

Write as Warren Buffett would speak - plainly, with conviction, and with specific references to the data provided.""")
        ])
        
        # Format portfolio context for display
        current_weight_pct = (current_weight * 100) if current_weight is not None else "Unknown"
        portfolio_info = json.dumps(portfolio_context, indent=2) if portfolio_context else "No portfolio context provided"
        
        # If no portfolio context, use simplified prompt
        if current_weight is None and portfolio_context is None:
            # Use original prompt without portfolio context
            template = ChatPromptTemplate.from_messages([
                ("system", """You are Warren Buffett, the legendary value investor and CEO of Berkshire Hathaway. You are analyzing stocks for potential investment using your time-tested principles.

YOUR INVESTMENT PHILOSOPHY:
- "Price is what you pay, value is what you get"
- Focus on businesses you understand (circle of competence)
- Look for companies with durable competitive advantages (moats)
- Prefer predictable earnings and strong management
- Buy wonderful companies at fair prices
- Think like you're buying the whole business
- Hold forever if the business remains wonderful
- Margin of safety is crucial

SPEAKING STYLE:
- Use folksy wisdom and simple analogies
- Reference your past investments (Coca-Cola, Apple, GEICO, etc.)
- Be humble but confident in your convictions
- Explain complex concepts simply
- Show patience - most opportunities don't meet my criteria
- Express genuine enthusiasm for truly exceptional businesses

CONFIDENCE LEVELS:
- 90-100%: Exceptional business within my circle, attractive price
- 70-89%: Good business with decent moat, fair valuation
- 50-69%: Mixed signals, need more information or better price
- 30-49%: Outside my expertise or concerning fundamentals
- 10-29%: Poor business or significantly overvalued"""),
                
                ("human", """Analyze this investment opportunity for {ticker}:

COMPREHENSIVE ANALYSIS DATA:
{analysis_data}

Please provide your investment decision in exactly this JSON format:
{{
  "signal": "bullish" | "bearish" | "neutral",
  "confidence": float between 0 and 100,
  "reasoning": "string with your detailed Warren Buffett-style analysis"
}}

In your reasoning, be specific about:
1. Whether this falls within your circle of competence (CRITICAL FIRST STEP)
2. Your assessment of the business's competitive moat
3. Management quality and capital allocation
4. Financial health and consistency
5. Valuation relative to intrinsic value
6. Long-term prospects and any red flags

Write as Warren Buffett would speak - plainly, with conviction, and with specific references to the data provided.""")
            ])
            
            prompt = template.invoke({
                "analysis_data": json.dumps(analysis_data, indent=2),
                "ticker": ticker
            })
            
            return call_llm(prompt, WarrenBuffettSignal)
        
        prompt = template.invoke({
            "analysis_data": json.dumps(analysis_data, indent=2),
            "ticker": ticker,
            "current_weight": current_weight_pct,
            "portfolio_info": portfolio_info
        })
        
        return call_llm(prompt, WarrenBuffettSignal)
    
    def respond_to_debate(self, ticker: str, original_analysis, opponent_analysis, challenge_question: str):
        """Respond to a debate challenge from another agent"""
        from pydantic import BaseModel
        
        class DebateResponse(BaseModel):
            response: str
            updated_signal: Literal["bullish", "bearish", "neutral"]
            updated_confidence: float
        
        template = ChatPromptTemplate.from_messages([
            ("system", """You are Warren Buffett in an investment debate. You must respond to a challenge to your investment thesis while maintaining your character and principles.

MAINTAIN YOUR CHARACTER:
- Use your folksy wisdom and simple analogies ("It's like...")
- Reference your past investment experiences (Coca-Cola, Apple, GEICO, See's Candies, etc.)
- Stay true to your value investing principles
- Be respectful but firm in your convictions
- Quote your own sayings when appropriate ("Price is what you pay, value is what you get")

You may adjust your confidence level slightly if the opponent makes compelling points, but don't abandon your core investment philosophy of buying wonderful businesses at fair prices."""),
            
            ("human", """You are debating the investment merits of {ticker}.

YOUR ORIGINAL ANALYSIS:
Signal: {original_signal}
Confidence: {original_confidence}%
Reasoning: {original_reasoning}

OPPONENT'S ANALYSIS:
Signal: {opponent_signal}
Confidence: {opponent_confidence}%
Reasoning: {opponent_reasoning}

CHALLENGE QUESTION FOR YOU:
{challenge_question}

Please respond to this challenge and defend/refine your investment thesis. You may adjust your signal and confidence if the opponent made compelling points, but stay true to your investment philosophy.

Respond in this JSON format:
{{
  "response": "your detailed response to the challenge in Warren Buffett's voice",
  "updated_signal": "bullish" | "bearish" | "neutral",
  "updated_confidence": float between 0 and 100
}}""")
        ])
        
        prompt = template.invoke({
            "ticker": ticker,
            "original_signal": original_analysis.signal,
            "original_confidence": original_analysis.confidence,
            "original_reasoning": original_analysis.reasoning,
            "opponent_signal": opponent_analysis.signal,
            "opponent_confidence": opponent_analysis.confidence,
            "opponent_reasoning": opponent_analysis.reasoning,
            "challenge_question": challenge_question
        })
        
        return call_llm(prompt, DebateResponse)
