"""
Simplified Cathie Wood Investment Agent
"""
from pydantic import BaseModel
from typing_extensions import Literal
from langchain_core.prompts import ChatPromptTemplate
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import get_financial_metrics, get_market_cap, search_line_items, call_llm, AnalysisResult


class CathieWoodSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str


class CathieWoodAgent:
    """Simplified Cathie Wood investment agent focused on disruptive innovation"""
    
    def __init__(self):
        self.agent_name = "Cathie Wood"
    
    def analyze_stock(self, ticker: str, end_date: str = "2024-12-31") -> CathieWoodSignal:
        """Analyze a stock using Cathie Wood's innovation-focused investment principles"""
        
        # Fetch financial data
        metrics = get_financial_metrics(ticker, end_date, period="annual", limit=5)
        financial_line_items = search_line_items(
            ticker,
            [
                "revenue",
                "gross_margin",
                "operating_margin",
                "debt_to_equity",
                "free_cash_flow",
                "total_assets",
                "total_liabilities",
                "dividends_and_other_cash_distributions",
                "outstanding_shares",
                "research_and_development",
                "capital_expenditure",
                "operating_expense",
            ],
            end_date,
            period="annual",
            limit=5
        )
        market_cap = get_market_cap(ticker, end_date)
        
        # Perform Cathie Wood-style analysis
        disruptive_analysis = self._analyze_disruptive_potential(metrics, financial_line_items)
        innovation_analysis = self._analyze_innovation_growth(metrics, financial_line_items)
        valuation_analysis = self._analyze_growth_valuation(financial_line_items, market_cap)
        tam_analysis = self._analyze_market_opportunity(ticker, financial_line_items)
        
        # Compile analysis data
        analysis_data = {
            "ticker": ticker,
            "market_cap": market_cap,
            "disruptive_analysis": disruptive_analysis.to_dict(),
            "innovation_analysis": innovation_analysis.to_dict(),
            "valuation_analysis": valuation_analysis.to_dict(),
            "tam_analysis": tam_analysis.to_dict(),
            "recent_metrics": metrics[:2] if metrics else [],
            "recent_financials": financial_line_items[:2] if financial_line_items else []
        }
        
        # Generate final investment decision using LLM
        return self._generate_cathie_wood_decision(ticker, analysis_data)
    
    def _analyze_disruptive_potential(self, metrics, financial_line_items) -> AnalysisResult:
        """Analyze whether the company has disruptive products, technology, or business model"""
        if not metrics or not financial_line_items:
            return AnalysisResult(0, "Insufficient data for disruption analysis")
        
        score = 0
        details = []
        
        # Revenue growth acceleration (indicates market adoption)
        revenues = [item.get("revenue", 0) for item in financial_line_items[:3]]
        if len(revenues) >= 3:
            growth_rates = []
            for i in range(1, len(revenues)):
                if revenues[i] > 0:
                    growth = (revenues[i-1] - revenues[i]) / revenues[i]
                    growth_rates.append(growth)
            
            if growth_rates:
                avg_growth = sum(growth_rates) / len(growth_rates)
                if avg_growth > 0.3:  # 30%+ growth
                    score += 3
                    details.append(f"Exceptional revenue growth of {avg_growth:.1%}")
                elif avg_growth > 0.15:  # 15%+ growth
                    score += 2
                    details.append(f"Strong revenue growth of {avg_growth:.1%}")
                elif avg_growth > 0.05:  # 5%+ growth
                    score += 1
                    details.append(f"Moderate revenue growth of {avg_growth:.1%}")
        
        # Gross margin expansion (indicates pricing power/efficiency)
        gross_margins = [m.get("gross_margin", 0) for m in metrics[:3]]
        if len(gross_margins) >= 2:
            margin_trend = gross_margins[0] - gross_margins[-1]
            if margin_trend > 0.05:  # 5% improvement
                score += 2
                details.append("Expanding gross margins show operational leverage")
            elif margin_trend > 0:
                score += 1
                details.append("Improving gross margins")
        
        return AnalysisResult(score, "; ".join(details), max_score=5)
    
    def _analyze_innovation_growth(self, metrics, financial_line_items) -> AnalysisResult:
        """Evaluate the company's commitment to innovation and potential for exponential growth"""
        if not financial_line_items:
            return AnalysisResult(0, "No financial data for innovation analysis")
        
        score = 0
        details = []
        
        # R&D investment analysis
        latest = financial_line_items[0]
        revenue = latest.get("revenue", 0)
        rd_expense = latest.get("research_and_development", 0)
        
        if revenue > 0 and rd_expense > 0:
            rd_ratio = rd_expense / revenue
            if rd_ratio > 0.15:  # 15%+ of revenue on R&D
                score += 3
                details.append(f"Heavy R&D investment at {rd_ratio:.1%} of revenue")
            elif rd_ratio > 0.08:  # 8%+ of revenue on R&D
                score += 2
                details.append(f"Significant R&D investment at {rd_ratio:.1%} of revenue")
            elif rd_ratio > 0.03:  # 3%+ of revenue on R&D
                score += 1
                details.append(f"Moderate R&D investment at {rd_ratio:.1%} of revenue")
        
        # Operating leverage analysis
        operating_margins = [m.get("operating_margin", 0) for m in metrics[:3]]
        if len(operating_margins) >= 2:
            margin_improvement = operating_margins[0] - operating_margins[-1]
            if margin_improvement > 0.05:  # 5% improvement
                score += 2
                details.append("Strong operating leverage with improving margins")
            elif margin_improvement > 0:
                score += 1
                details.append("Positive operating leverage")
        
        return AnalysisResult(score, "; ".join(details), max_score=5)
    
    def _analyze_growth_valuation(self, financial_line_items, market_cap) -> AnalysisResult:
        """Cathie Wood's growth-focused valuation approach"""
        if not financial_line_items:
            return AnalysisResult(0, "No financial data for valuation")
        
        latest = financial_line_items[0]
        revenue = latest.get("revenue", 0)
        
        if revenue <= 0:
            return AnalysisResult(0, "No revenue for valuation analysis")
        
        # Price-to-Sales analysis for growth companies
        ps_ratio = market_cap / revenue if revenue > 0 else float('inf')
        
        # Revenue growth rate
        revenues = [item.get("revenue", 0) for item in financial_line_items[:3]]
        growth_rate = 0
        if len(revenues) >= 2 and revenues[1] > 0:
            growth_rate = (revenues[0] - revenues[1]) / revenues[1]
        
        # PEG-like ratio for growth companies (PS ratio / growth rate)
        score = 0
        details = []
        
        if growth_rate > 0.2:  # 20%+ growth
            if ps_ratio < 10:
                score = 3
                details.append(f"Attractive valuation: P/S {ps_ratio:.1f} for {growth_rate:.1%} growth")
            elif ps_ratio < 20:
                score = 2
                details.append(f"Reasonable valuation: P/S {ps_ratio:.1f} for {growth_rate:.1%} growth")
            else:
                score = 1
                details.append(f"High valuation: P/S {ps_ratio:.1f} for {growth_rate:.1%} growth")
        elif growth_rate > 0.1:  # 10%+ growth
            if ps_ratio < 5:
                score = 2
                details.append(f"Good valuation for moderate growth company")
            elif ps_ratio < 15:
                score = 1
                details.append(f"Fair valuation for moderate growth")
        else:
            details.append(f"Low growth rate of {growth_rate:.1%} concerning for growth strategy")
        
        return AnalysisResult(score, "; ".join(details), ps_ratio=ps_ratio, growth_rate=growth_rate, max_score=3)
    
    def _analyze_market_opportunity(self, ticker: str, financial_line_items) -> AnalysisResult:
        """Analyze Total Addressable Market (TAM) and market penetration potential"""
        # Simplified TAM analysis based on industry/ticker
        tam_estimates = {
            "TSLA": {"tam": 10000000000000, "description": "Global automotive and energy storage market"},  # $10T
            "NVDA": {"tam": 1000000000000, "description": "AI/GPU computing market"},  # $1T
            "ROKU": {"tam": 500000000000, "description": "Global streaming/CTV advertising market"},  # $500B
            "SQ": {"tam": 200000000000, "description": "Global payments and fintech market"},  # $200B
            "TDOC": {"tam": 300000000000, "description": "Global healthcare market"},  # $300B
        }
        
        tam_info = tam_estimates.get(ticker, {"tam": 100000000000, "description": "Estimated addressable market"})
        
        if not financial_line_items:
            return AnalysisResult(1, f"Large TAM in {tam_info['description']}")
        
        latest_revenue = financial_line_items[0].get("revenue", 0)
        market_penetration = latest_revenue / tam_info["tam"] if tam_info["tam"] > 0 else 0
        
        score = 0
        details = []
        
        if market_penetration < 0.01:  # Less than 1% penetration
            score = 3
            details.append(f"Massive growth runway with {market_penetration:.3%} market penetration in {tam_info['description']}")
        elif market_penetration < 0.05:  # Less than 5% penetration
            score = 2
            details.append(f"Significant growth potential with {market_penetration:.2%} market penetration")
        elif market_penetration < 0.1:  # Less than 10% penetration
            score = 1
            details.append(f"Moderate growth runway with {market_penetration:.2%} market penetration")
        else:
            details.append(f"High market penetration of {market_penetration:.2%} may limit growth")
        
        return AnalysisResult(
            score, 
            "; ".join(details), 
            tam=tam_info["tam"],
            market_penetration=market_penetration,
            max_score=3
        )
    
    def _generate_cathie_wood_decision(self, ticker: str, analysis_data: dict) -> CathieWoodSignal:
        """Generate final investment decision using LLM with Cathie Wood's voice"""
        
        template = ChatPromptTemplate.from_messages([
            ("system", """You are Cathie Wood, founder and CEO of ARK Invest, known for investing in disruptive innovation. Analyze this investment opportunity using your proven methodology:

MY CORE INVESTMENT PHILOSOPHY:
- Invest in companies leveraging disruptive innovation with exponential growth potential
- Focus on technologies that can transform industries and create new markets
- Seek companies with large Total Addressable Markets (TAM) and low market penetration
- Prioritize long-term exponential growth over short-term profitability
- Look for management teams with bold vision and commitment to innovation
- Accept higher volatility in pursuit of transformational returns

MY KEY INNOVATION PLATFORMS:
- Artificial Intelligence and Machine Learning
- Robotics and Automation
- Genomics and Precision Medicine
- Fintech and Digital Wallets
- Electric Vehicles and Autonomous Technology
- Space Exploration and Satellite Technology
- Blockchain and Cryptocurrency
- Cloud Computing and Cybersecurity

WHAT I LOOK FOR:
1. Disruptive Technology - Is this creating new markets or transforming existing ones?
2. Exponential Growth Potential - Can this scale rapidly with network effects?
3. Large TAM - Is the addressable market measured in hundreds of billions or trillions?
4. Innovation Investment - Is management investing heavily in R&D and future capabilities?
5. Competitive Moats - Does the technology create sustainable advantages?
6. Visionary Leadership - Does management think in decades, not quarters?

MY INVESTMENT APPROACH:
- 5+ year investment horizon focused on transformational change
- Willing to endure volatility for exponential upside potential
- Price-to-Sales ratios matter less than growth trajectory and market opportunity
- Look for companies that could be 10x larger in 10 years
- Focus on platform companies that can expand into adjacent markets

MY LANGUAGE & STYLE:
- Enthusiastic about transformational technologies
- Reference specific innovation trends and adoption curves
- Discuss long-term vision and exponential possibilities
- Use data to support bold predictions
- Express conviction about disruptive change

CONFIDENCE LEVELS:
- 90-100%: Revolutionary technology, massive TAM, early adoption phase
- 70-89%: Strong innovation platform, good growth metrics, clear disruption
- 50-69%: Interesting technology but unclear market adoption or competition
- 30-49%: Limited innovation or small addressable market
- 10-29%: Incremental improvement rather than true disruption"""),
            
            ("human", """Based on the following analysis, create a Cathie Wood-style investment signal for {ticker}:

COMPREHENSIVE ANALYSIS DATA:
{analysis_data}

Please provide your investment decision in exactly this JSON format:
{{
  "signal": "bullish" | "bearish" | "neutral",
  "confidence": float between 0 and 100,
  "reasoning": "string with your detailed Cathie Wood-style analysis"
}}

In your reasoning, be specific about:
1. The specific disruptive technologies/innovations the company is leveraging
2. Growth metrics that indicate exponential potential (revenue acceleration, expanding TAM)
3. Long-term vision and transformative potential over 5+ year horizons
4. How the company might disrupt traditional industries or create new markets
5. R&D investment and innovation pipeline that could drive future growth
6. Market opportunity and current penetration levels

Use my optimistic, future-focused, and conviction-driven voice with specific references to the data provided.""")
        ])
        
        prompt = template.invoke({
            "analysis_data": json.dumps(analysis_data, indent=2),
            "ticker": ticker
        })
        
        return call_llm(prompt, CathieWoodSignal)
    
    def respond_to_debate(self, ticker: str, original_analysis, opponent_analysis, challenge_question: str):
        """Respond to a debate challenge from another agent"""
        from pydantic import BaseModel
        
        class DebateResponse(BaseModel):
            response: str
            updated_signal: Literal["bullish", "bearish", "neutral"]
            updated_confidence: float
        
        template = ChatPromptTemplate.from_messages([
            ("system", """You are Cathie Wood in an investment debate. You must respond to a challenge to your investment thesis while maintaining your character and principles.

MAINTAIN YOUR CHARACTER:
- Express enthusiasm for transformational technologies
- Reference innovation trends and exponential growth potential
- Stay true to your growth investing principles
- Be respectful but passionate about disruptive innovation
- Discuss platform effects, network advantages, and TAM expansion
- Reference specific innovation platforms (AI, robotics, genomics, fintech, etc.)

You may adjust your confidence level slightly if the opponent makes compelling points, but don't abandon your focus on exponential growth and transformative innovation."""),
            
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
  "response": "your detailed response to the challenge in Cathie Wood's voice",
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
