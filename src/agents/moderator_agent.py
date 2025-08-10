"""
Moderator Agent for facilitating debates between investment agents
"""
from pydantic import BaseModel
from typing_extensions import Literal
from langchain_core.prompts import ChatPromptTemplate
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import call_llm


class DebatePoint(BaseModel):
    key_disagreement: str
    buffett_position: str
    wood_position: str
    question_for_buffett: str
    question_for_wood: str


class ConsensusDecision(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str
    buffett_weight: float  # 0-1, how much Buffett's view influenced the decision
    wood_weight: float     # 0-1, how much Wood's view influenced the decision


class ModeratorAgent:
    """Moderator agent that facilitates debates and synthesizes consensus"""
    
    def __init__(self):
        self.agent_name = "Investment Moderator"
    
    def identify_disagreements(self, ticker: str, buffett_analysis: dict, wood_analysis: dict) -> DebatePoint:
        """Identify key disagreements between the two agents"""
        
        template = ChatPromptTemplate.from_messages([
            ("system", """You are an experienced investment moderator facilitating a debate between Warren Buffett and Cathie Wood. 

Your role is to:
1. Identify the KEY DISAGREEMENT between their investment analyses
2. Understand each agent's core position and reasoning
3. Formulate specific questions that will help them debate the disagreement
4. Focus on the most important philosophical or analytical differences

You should look for disagreements in:
- Valuation approaches (intrinsic value vs growth potential)
- Time horizons (long-term stability vs exponential growth)
- Risk assessment (margin of safety vs innovation risk)
- Market analysis (competitive moats vs disruptive potential)
- Business model preferences (proven vs transformative)

Be specific and actionable in your questions to promote meaningful debate."""),
            
            ("human", """Analyze these two investment perspectives for {ticker} and identify the key disagreement:

WARREN BUFFETT'S ANALYSIS:
Signal: {buffett_signal}
Confidence: {buffett_confidence}%
Reasoning: {buffett_reasoning}

CATHIE WOOD'S ANALYSIS:
Signal: {wood_signal}
Confidence: {wood_confidence}%
Reasoning: {wood_reasoning}

Please identify the core disagreement and formulate debate questions in this JSON format:
{{
  "key_disagreement": "string describing the main point of disagreement",
  "buffett_position": "summary of Buffett's core argument",
  "wood_position": "summary of Wood's core argument", 
  "question_for_buffett": "specific question challenging Buffett's view",
  "question_for_wood": "specific question challenging Wood's view"
}}""")
        ])
        
        prompt = template.invoke({
            "ticker": ticker,
            "buffett_signal": buffett_analysis["signal"],
            "buffett_confidence": buffett_analysis["confidence"],
            "buffett_reasoning": buffett_analysis["reasoning"],
            "wood_signal": wood_analysis["signal"],
            "wood_confidence": wood_analysis["confidence"],
            "wood_reasoning": wood_analysis["reasoning"]
        })
        
        return call_llm(prompt, DebatePoint)
    
    def synthesize_consensus(self, ticker: str, debate_history: list) -> ConsensusDecision:
        """Synthesize final consensus after the debate"""
        
        template = ChatPromptTemplate.from_messages([
            ("system", """You are an expert investment moderator tasked with synthesizing a consensus between Warren Buffett and Cathie Wood after their debate.

Your role is to:
1. Weigh the strength of each agent's arguments
2. Consider how their different time horizons and philosophies can be reconciled
3. Create a balanced investment decision that incorporates both perspectives
4. Assign weights showing how much each agent's view influenced the final decision
5. Provide clear reasoning for the consensus

SYNTHESIS PRINCIPLES:
- Value Buffett's focus on downside protection and business fundamentals
- Value Wood's focus on upside potential and innovation
- Consider both short-term risks and long-term opportunities
- Look for areas where both agents actually agree (often overlooked)
- Create a decision that neither agent would strongly object to

CONFIDENCE LEVELS:
- 90-100%: Both agents strongly agree or compelling synthesis of their views
- 70-89%: One agent strongly convinced the other, or clear compromise
- 50-69%: Balanced but uncertain, significant disagreement remains
- 30-49%: Weak synthesis, agents remain far apart
- 10-29%: No meaningful consensus possible

WEIGHTING GUIDELINES:
- If Buffett's valuation concerns are compelling: Higher buffett_weight
- If Wood's growth thesis is compelling: Higher wood_weight
- If both have valid points: Balanced weights (0.5 each)
- Weights should sum to 1.0"""),
            
            ("human", """Based on this debate between Warren Buffett and Cathie Wood about {ticker}, synthesize a consensus investment decision:

DEBATE HISTORY:
{debate_history}

Please provide the consensus in this JSON format:
{{
  "signal": "bullish" | "bearish" | "neutral",
  "confidence": float between 0 and 100,
  "reasoning": "detailed explanation of the consensus and how both perspectives were considered",
  "buffett_weight": float between 0 and 1,
  "wood_weight": float between 0 and 1
}}

In your reasoning, explain:
1. Which arguments were most compelling from each agent
2. How you reconciled their different time horizons and risk tolerances
3. What the final investment thesis represents
4. Why this consensus makes sense for both value and growth investors""")
        ])
        
        prompt = template.invoke({
            "ticker": ticker,
            "debate_history": json.dumps(debate_history, indent=2)
        })
        
        return call_llm(prompt, ConsensusDecision)
    
    def generate_challenge_question(self, ticker: str, challenger_analysis, opponent_analysis, direction: str):
        """Generate a challenge question for the debate"""
        
        from pydantic import BaseModel
        
        class ChallengeQuestion(BaseModel):
            question: str
        
        if direction == "buffett_to_wood":
            challenger_name = "Warren Buffett"
            opponent_name = "Cathie Wood"
            challenger = challenger_analysis
            opponent = opponent_analysis
        else:  # wood_to_buffett
            challenger_name = "Cathie Wood"
            opponent_name = "Warren Buffett"
            challenger = challenger_analysis
            opponent = opponent_analysis
        
        template = ChatPromptTemplate.from_messages([
            ("system", f"""You are facilitating a debate between Warren Buffett and Cathie Wood. 
            
Generate a challenging question that {challenger_name} would ask {opponent_name} about their investment thesis for {{ticker}}.

The question should:
1. Challenge a key weakness in the opponent's analysis
2. Highlight where {challenger_name}'s philosophy differs
3. Be specific and actionable
4. Force the opponent to defend their position
5. Stay true to {challenger_name}'s investment style

Make it a thoughtful, professional challenge that could lead to meaningful debate."""),
            
            ("human", f"""{challenger_name}'s Analysis:
Signal: {{challenger_signal}}
Confidence: {{challenger_confidence}}%
Reasoning: {{challenger_reasoning}}

{opponent_name}'s Analysis:
Signal: {{opponent_signal}}
Confidence: {{opponent_confidence}}%
Reasoning: {{opponent_reasoning}}

Generate a challenge question that {challenger_name} would ask {opponent_name} about {{ticker}}.""")
        ])
        
        prompt = template.invoke({
            "ticker": ticker,
            "challenger_signal": challenger.signal,
            "challenger_confidence": challenger.confidence,
            "challenger_reasoning": challenger.reasoning,
            "opponent_signal": opponent.signal,
            "opponent_confidence": opponent.confidence,
            "opponent_reasoning": opponent.reasoning
        })
        
        return call_llm(prompt, ChallengeQuestion)
    
    def make_final_decision(self, ticker: str, buffett_initial, wood_initial, buffett_response, wood_response, buffett_challenge, wood_challenge) -> ConsensusDecision:
        """Make final consensus decision after the full debate"""
        
        # Prepare debate history for synthesis
        debate_history = [
            {
                "phase": "initial_presentations",
                "buffett_analysis": {
                    "signal": buffett_initial.signal,
                    "confidence": buffett_initial.confidence,
                    "reasoning": buffett_initial.reasoning
                },
                "wood_analysis": {
                    "signal": wood_initial.signal,
                    "confidence": wood_initial.confidence,
                    "reasoning": wood_initial.reasoning
                }
            },
            {
                "phase": "cross_examination",
                "buffett_challenge": buffett_challenge.question,
                "wood_response": {
                    "response": wood_response.response,
                    "updated_signal": wood_response.updated_signal,
                    "updated_confidence": wood_response.updated_confidence
                },
                "wood_challenge": wood_challenge.question,
                "buffett_response": {
                    "response": buffett_response.response,
                    "updated_signal": buffett_response.updated_signal,
                    "updated_confidence": buffett_response.updated_confidence
                }
            }
        ]
        
        return self.synthesize_consensus(ticker, debate_history)
