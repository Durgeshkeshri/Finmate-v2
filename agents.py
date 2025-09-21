"""
Multi-Agent Financial Advisory System with Agent-to-Agent Communication
Built for Build & Grow AI Hackathon

This file contains the core agent system with proper agent-to-agent communication,
MongoDB integration, and MCP (Model Context Protocol) support.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import google.generativeai as genai
from pymongo import MongoClient
from pymongo.collection import Collection
import os
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or "AIzaSyAIiwdBBGPpiG1PgztbQI34XIQxHPhcpDc"
MONGODB_URL = "mongodb+srv://durgeshkeshri7:Durgesh1027@cluster0.ezhdt7p.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

genai.configure(api_key=GEMINI_API_KEY)

@dataclass
class FinancialProfile:
    """Enhanced financial profile data structure"""
    age: int
    income: float
    fixed_expenses: float
    variable_expenses: float
    current_savings: float
    debt: float
    risk_tolerance: str
    financial_goals: List[str]
    goal_amounts: List[float]
    goal_timelines: List[int]
    employment_status: str = "Full-time"
    family_size: int = 1
    location: str = "US"
    user_id: str = None
    created_at: datetime = None

@dataclass
class AgentMessage:
    """Message structure for agent-to-agent communication"""
    sender: str
    receiver: str
    message_type: str
    content: Dict[str, Any]
    timestamp: datetime = None
    session_id: str = None

class DatabaseManager:
    """MongoDB database manager with MCP support"""
    
    def __init__(self, mongodb_url: str):
        self.client = MongoClient(mongodb_url)
        self.db = self.client.financial_advisor
        self.profiles: Collection = self.db.profiles
        self.analyses: Collection = self.db.analyses
        self.agent_messages: Collection = self.db.agent_messages
        self.sessions: Collection = self.db.sessions
        
        # Create indexes for better performance
        self.profiles.create_index("user_id")
        self.analyses.create_index([("user_id", 1), ("created_at", -1)])
        self.agent_messages.create_index([("session_id", 1), ("timestamp", -1)])
    
    def save_profile(self, profile: FinancialProfile) -> str:
        """Save financial profile to database"""
        profile.created_at = datetime.utcnow()
        if not profile.user_id:
            profile.user_id = f"user_{datetime.utcnow().timestamp()}"
        
        profile_dict = asdict(profile)
        result = self.profiles.insert_one(profile_dict)
        logger.info(f"Profile saved with ID: {result.inserted_id}")
        return str(result.inserted_id)
    
    def save_analysis(self, user_id: str, analysis: Dict[str, Any]) -> str:
        """Save analysis results to database"""
        analysis_doc = {
            "user_id": user_id,
            "analysis": analysis,
            "created_at": datetime.utcnow()
        }
        result = self.analyses.insert_one(analysis_doc)
        return str(result.inserted_id)
    
    def save_agent_message(self, message: AgentMessage) -> str:
        """Save agent-to-agent communication message"""
        message.timestamp = datetime.utcnow()
        message_dict = asdict(message)
        result = self.agent_messages.insert_one(message_dict)
        return str(result.inserted_id)
    
    def get_user_history(self, user_id: str, limit: int = 10) -> List[Dict]:
        """Get user's analysis history"""
        return list(self.analyses.find(
            {"user_id": user_id}
        ).sort("created_at", -1).limit(limit))

class BaseAgent:
    """Enhanced base class for all agents with communication capabilities"""
    
    def __init__(self, name: str, agent_id: str, db_manager: DatabaseManager):
        self.name = name
        self.agent_id = agent_id
        self.db_manager = db_manager
        self.model = genai.GenerativeModel('gemini-pro')
        self.message_queue = asyncio.Queue()
        self.is_running = False
        self.session_id = None
        
        # Agent's knowledge base and capabilities
        self.capabilities = []
        self.knowledge_base = {}
        
    async def start(self):
        """Start the agent's message processing loop"""
        self.is_running = True
        logger.info(f"Agent {self.name} started")
        
    async def stop(self):
        """Stop the agent"""
        self.is_running = False
        logger.info(f"Agent {self.name} stopped")
    
    async def send_message(self, receiver_id: str, message_type: str, content: Dict[str, Any]):
        """Send message to another agent"""
        message = AgentMessage(
            sender=self.agent_id,
            receiver=receiver_id,
            message_type=message_type,
            content=content,
            session_id=self.session_id
        )
        
        # Save message to database for audit trail
        self.db_manager.save_agent_message(message)
        
        logger.info(f"{self.name} -> {receiver_id}: {message_type}")
        return message
    
    async def receive_message(self, message: AgentMessage):
        """Receive and queue message from another agent"""
        await self.message_queue.put(message)
        logger.info(f"{self.name} received message from {message.sender}: {message.message_type}")
    
    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process incoming message - to be overridden by specific agents"""
        raise NotImplementedError("Agents must implement process_message")
    
    def generate_ai_response(self, prompt: str) -> str:
        """Generate AI response using Gemini"""
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"AI response error in {self.name}: {e}")
            return f"I apologize, but I'm experiencing technical difficulties. {self.name} will provide basic guidance."

class DataValidationAgent(BaseAgent):
    """Specialized agent for data validation and processing"""
    
    def __init__(self, db_manager: DatabaseManager):
        super().__init__("Data Validation Agent", "data_validator", db_manager)
        self.capabilities = [
            "financial_data_validation",
            "data_structuring", 
            "risk_assessment",
            "compliance_checking"
        ]
        
        self.validation_rules = {
            "max_expense_ratio": 0.95,
            "min_emergency_months": 0.5,
            "max_debt_to_income": 10.0,
            "reasonable_age_range": (16, 100)
        }
    
    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process validation requests"""
        if message.message_type == "validate_profile":
            profile_data = message.content["profile"]
            profile = FinancialProfile(**profile_data)
            
            validation_result = await self.validate_financial_profile(profile)
            
            # Send validation results to Financial Analyst
            response = await self.send_message(
                "financial_analyst",
                "validation_complete",
                {
                    "validation_result": validation_result,
                    "profile": profile_data,
                    "request_id": message.content.get("request_id")
                }
            )
            return response
            
        elif message.message_type == "request_user_history":
            user_id = message.content["user_id"]
            history = self.db_manager.get_user_history(user_id)
            
            response = await self.send_message(
                message.sender,
                "user_history_response",
                {
                    "user_id": user_id,
                    "history": history,
                    "request_id": message.content.get("request_id")
                }
            )
            return response
    
    async def validate_financial_profile(self, profile: FinancialProfile) -> Dict[str, Any]:
        """Comprehensive financial data validation with AI insights"""
        
        validation_result = {
            "is_valid": True,
            "warnings": [],
            "suggestions": [],
            "structured_data": {},
            "risk_factors": [],
            "ai_insights": ""
        }
        
        # Basic validations
        if profile.age < 16 or profile.age > 100:
            validation_result["warnings"].append("Please verify your age is correct")
        
        if profile.income <= 0:
            validation_result["is_valid"] = False
            validation_result["warnings"].append("Income must be positive")
            return validation_result
        
        # Calculate key financial metrics
        total_expenses = profile.fixed_expenses + profile.variable_expenses
        expense_ratio = total_expenses / profile.income
        available_savings = max(0, profile.income - total_expenses)
        savings_rate = available_savings / profile.income if profile.income > 0 else 0
        debt_to_income_annual = profile.debt / (profile.income * 12) if profile.income > 0 else 0
        
        # Advanced expense analysis
        if expense_ratio > 1.0:
            validation_result["risk_factors"].append("Critical: Expenses exceed income")
            validation_result["warnings"].append("🚨 URGENT: Your expenses exceed your income. Immediate action required.")
        elif expense_ratio > 0.85:
            validation_result["warnings"].append("⚠️ High expense ratio - budget optimization recommended")
        
        # Structure the data
        validation_result["structured_data"] = {
            "monthly_income": profile.income,
            "total_expenses": total_expenses,
            "expense_ratio": expense_ratio,
            "savings_rate": savings_rate,
            "available_for_savings": available_savings,
            "debt_to_income": debt_to_income_annual,
            "net_worth_estimate": profile.current_savings - profile.debt,
            "financial_runway_months": profile.current_savings / total_expenses if total_expenses > 0 else float('inf')
        }
        
        # Generate AI insights
        ai_prompt = f"""
        As a financial data validation expert, analyze this profile:
        
        Age: {profile.age}, Income: ${profile.income:,.0f}/month
        Expenses: ${total_expenses:,.0f}/month ({expense_ratio:.1%} of income)
        Savings Rate: {savings_rate:.1%}
        Current Savings: ${profile.current_savings:,.0f}
        Debt: ${profile.debt:,.0f}
        Risk Tolerance: {profile.risk_tolerance}
        
        Provide 2-3 key insights about this financial profile's health and immediate priorities.
        Focus on data quality, completeness, and any red flags. Be concise and actionable.
        """
        
        validation_result["ai_insights"] = self.generate_ai_response(ai_prompt)
        
        # Save validated profile to database
        self.db_manager.save_profile(profile)
        
        logger.info(f"Validation complete for user profile - Valid: {validation_result['is_valid']}")
        return validation_result

class FinancialAnalysisAgent(BaseAgent):
    """Specialized agent for financial analysis and strategy generation"""
    
    def __init__(self, db_manager: DatabaseManager):
        super().__init__("Financial Analysis Agent", "financial_analyst", db_manager)
        self.capabilities = [
            "investment_strategy",
            "budget_analysis",
            "goal_planning",
            "risk_assessment",
            "retirement_projection"
        ]
        
        self.financial_models = AdvancedFinancialModels()
        self.analysis_cache = {}
    
    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process financial analysis requests"""
        if message.message_type == "validation_complete":
            validation_result = message.content["validation_result"]
            profile_data = message.content["profile"]
            request_id = message.content.get("request_id")
            
            if validation_result["is_valid"]:
                # Perform comprehensive analysis
                analysis = await self.perform_comprehensive_analysis(
                    FinancialProfile(**profile_data),
                    validation_result
                )
                
                # Save analysis to database
                user_id = profile_data.get("user_id")
                if user_id:
                    self.db_manager.save_analysis(user_id, analysis)
                
                # Send results back to coordinator or UI
                response = await self.send_message(
                    "coordinator",
                    "analysis_complete",
                    {
                        "analysis": analysis,
                        "validation_result": validation_result,
                        "request_id": request_id
                    }
                )
                return response
            else:
                # Send validation errors back
                response = await self.send_message(
                    "coordinator",
                    "analysis_failed",
                    {
                        "errors": validation_result["warnings"],
                        "request_id": request_id
                    }
                )
                return response
        
        elif message.message_type == "request_strategy_update":
            # Handle requests for strategy updates based on market conditions
            profile_data = message.content["profile"]
            analysis = await self.update_investment_strategy(FinancialProfile(**profile_data))
            
            response = await self.send_message(
                message.sender,
                "strategy_update_complete",
                {"updated_analysis": analysis}
            )
            return response
    
    async def perform_comprehensive_analysis(self, profile: FinancialProfile, validation_result: Dict) -> Dict[str, Any]:
        """Perform comprehensive financial analysis"""
        
        logger.info("Starting comprehensive financial analysis...")
        
        structured_data = validation_result["structured_data"]
        
        # Apply core financial models
        budget_analysis = self.apply_50_30_20_rule(profile.income)
        emergency_fund = self.calculate_emergency_fund(
            structured_data["total_expenses"], 
            profile.risk_tolerance,
            profile.employment_status
        )
        investment_strategy = await self.generate_investment_strategy(profile)
        health_analysis = self.analyze_financial_health(profile, structured_data)
        goal_analysis = self.analyze_financial_goals(profile)
        
        # Generate AI-powered insights
        ai_insights = await self.generate_ai_insights(profile, structured_data)
        
        analysis_result = {
            "profile_summary": {
                "age": profile.age,
                "monthly_income": profile.income,
                "savings_rate": structured_data["savings_rate"],
                "risk_tolerance": profile.risk_tolerance
            },
            "budget_analysis": budget_analysis,
            "emergency_fund": emergency_fund,
            "investment_strategy": investment_strategy,
            "health_analysis": health_analysis,
            "goal_analysis": goal_analysis,
            "ai_insights": ai_insights,
            "structured_data": structured_data,
            "analysis_timestamp": datetime.utcnow().isoformat()
        }
        
        logger.info("Comprehensive analysis completed")
        return analysis_result
    
    def apply_50_30_20_rule(self, income: float) -> Dict[str, float]:
        """Apply 50/30/20 budgeting rule"""
        return {
            "needs": income * 0.50,
            "wants": income * 0.30,
            "savings": income * 0.20
        }
    
    def calculate_emergency_fund(self, monthly_expenses: float, risk_tolerance: str, employment_status: str = "Full-time") -> Dict[str, Any]:
        """Calculate recommended emergency fund"""
        base_multipliers = {
            "Conservative": 6,
            "Moderate": 4,
            "Aggressive": 3
        }
        
        employment_adjustments = {
            "Full-time": 0,
            "Contract": 1,
            "Freelance": 2,
            "Self-employed": 2,
            "Part-time": 1
        }
        
        base_months = base_multipliers.get(risk_tolerance, 4)
        adjustment = employment_adjustments.get(employment_status, 0)
        recommended_months = base_months + adjustment
        
        return {
            "recommended_months": recommended_months,
            "target_amount": monthly_expenses * recommended_months,
            "reasoning": f"Based on {risk_tolerance.lower()} risk tolerance and {employment_status.lower()} employment"
        }
    
    async def generate_investment_strategy(self, profile: FinancialProfile) -> Dict[str, Any]:
        """Generate detailed investment strategy with AI recommendations"""
        
        monthly_available = max(0, profile.income - profile.fixed_expenses - profile.variable_expenses)
        
        # Age and risk-based allocation
        risk_adjustments = {"Conservative": -0.2, "Moderate": 0.0, "Aggressive": 0.2}
        risk_adj = risk_adjustments.get(profile.risk_tolerance, 0.0)
        stock_percentage = max(20, min(90, (100 - profile.age) + (risk_adj * 100)))
        bond_percentage = max(10, 100 - stock_percentage - 10)
        
        # Generate AI-powered strategy recommendations
        ai_prompt = f"""
        Create detailed investment recommendations for:
        - Age: {profile.age}
        - Risk Tolerance: {profile.risk_tolerance}
        - Monthly Investment Capacity: ${monthly_available:,.0f}
        - Goals: {', '.join(profile.financial_goals)}
        
        Provide specific fund recommendations, account prioritization, and implementation timeline.
        Focus on low-cost index funds and tax-advantaged accounts.
        """
        
        ai_recommendations = self.generate_ai_response(ai_prompt)
        
        return {
            "asset_allocation": {
                "stocks": int(stock_percentage),
                "bonds": int(bond_percentage),
                "cash": 10
            },
            "monthly_investment": monthly_available * 0.8,
            "ai_recommendations": ai_recommendations,
            "risk_level": profile.risk_tolerance,
            "implementation_priority": [
                "Emergency fund completion",
                "401(k) employer match",
                "Roth IRA maximization",
                "Taxable account investing"
            ]
        }
    
    def analyze_financial_health(self, profile: FinancialProfile, structured_data: Dict) -> Dict[str, Any]:
        """Analyze overall financial health with scoring"""
        
        score = 0
        factors = []
        recommendations = []
        
        # Savings rate analysis (25 points)
        savings_rate = structured_data["savings_rate"]
        if savings_rate >= 0.20:
            score += 25
            factors.append("✅ Excellent savings rate (20%+)")
        elif savings_rate >= 0.15:
            score += 20
            factors.append("✅ Good savings rate (15%+)")
        elif savings_rate >= 0.10:
            score += 15
            factors.append("⚠️ Moderate savings rate (10%+)")
        else:
            score += 5
            factors.append("❌ Low savings rate (<10%)")
            recommendations.append("Priority: Increase savings rate to 15%+")
        
        # Additional health factors...
        debt_to_income = structured_data["debt_to_income"]
        if debt_to_income <= 0.1:
            score += 25
            factors.append("✅ Low debt burden")
        elif debt_to_income <= 0.3:
            score += 20
            factors.append("✅ Manageable debt levels")
        else:
            score += 5
            factors.append("❌ High debt burden")
            recommendations.append("Focus on debt reduction")
        
        # Calculate grade
        grades = {90: "A+", 80: "A", 70: "B", 60: "C"}
        grade = next((g for s, g in grades.items() if score >= s), "D")
        
        return {
            "score": min(100, score),
            "grade": grade,
            "factors": factors,
            "recommendations": recommendations
        }
    
    def analyze_financial_goals(self, profile: FinancialProfile) -> Dict[str, Any]:
        """Analyze financial goals and provide recommendations"""
        
        goal_analysis = []
        for i, goal in enumerate(profile.financial_goals):
            if i < len(profile.goal_amounts) and i < len(profile.goal_timelines):
                amount = profile.goal_amounts[i]
                timeline = profile.goal_timelines[i]
                monthly_needed = amount / (timeline * 12)
                feasibility = "High" if monthly_needed < profile.income * 0.1 else "Medium" if monthly_needed < profile.income * 0.2 else "Low"
                
                goal_analysis.append({
                    "goal": goal,
                    "target_amount": amount,
                    "timeline_years": timeline,
                    "monthly_required": monthly_needed,
                    "feasibility": feasibility
                })
        
        return {
            "goals": goal_analysis,
            "total_monthly_required": sum(g["monthly_required"] for g in goal_analysis),
            "recommendations": "Prioritize high-feasibility goals and consider extending timelines for others"
        }
    
    async def generate_ai_insights(self, profile: FinancialProfile, structured_data: Dict) -> str:
        """Generate AI-powered financial insights"""
        
        ai_prompt = f"""
        Provide personalized financial insights for this profile:
        
        Demographics: {profile.age} years old, {profile.employment_status}, family of {profile.family_size}
        Income: ${profile.income:,.0f}/month
        Savings Rate: {structured_data['savings_rate']:.1%}
        Net Worth: ${structured_data['net_worth_estimate']:,.0f}
        Risk Tolerance: {profile.risk_tolerance}
        Goals: {', '.join(profile.financial_goals)}
        
        Provide 3-4 specific, actionable insights focusing on:
        1. Biggest opportunities for improvement
        2. Strengths to build upon
        3. Risk mitigation strategies
        4. Next steps for goal achievement
        
        Be encouraging but realistic. Limit to 300 words.
        """
        
        return self.generate_ai_response(ai_prompt)

class AgentCoordinator:
    """Central coordinator for agent-to-agent communication"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.agents = {}
        self.active_sessions = {}
        self.message_history = []
        
    def register_agent(self, agent: BaseAgent):
        """Register an agent with the coordinator"""
        self.agents[agent.agent_id] = agent
        logger.info(f"Registered agent: {agent.name}")
    
    async def start_analysis_session(self, profile_data: Dict, session_id: str) -> str:
        """Start a new financial analysis session"""
        
        request_id = f"req_{datetime.utcnow().timestamp()}"
        self.active_sessions[session_id] = {
            "request_id": request_id,
            "status": "started",
            "start_time": datetime.utcnow()
        }
        
        # Set session ID for all agents
        for agent in self.agents.values():
            agent.session_id = session_id
        
        # Send profile to Data Validation Agent
        data_agent = self.agents.get("data_validator")
        if data_agent:
            message = AgentMessage(
                sender="coordinator",
                receiver="data_validator",
                message_type="validate_profile",
                content={
                    "profile": profile_data,
                    "request_id": request_id,
                    "session_id": session_id
                },
                session_id=session_id
            )
            
            await data_agent.receive_message(message)
            await data_agent.process_message(message)
        
        return request_id
    
    async def get_session_result(self, session_id: str) -> Optional[Dict]:
        """Get the final result for a session"""
        if session_id not in self.active_sessions:
            return None
        
        # In a real implementation, you'd wait for the analysis_complete message
        # For now, return the session status
        return self.active_sessions[session_id]

class AdvancedFinancialModels:
    """Advanced financial calculations and projections"""
    
    @staticmethod
    def retirement_projection(age: int, retirement_age: int, current_savings: float, 
                            monthly_contribution: float, expected_return: float = 0.07) -> Dict:
        """Calculate retirement savings projection"""
        years_to_retirement = max(1, retirement_age - age)
        
        # Future value calculations
        fv_current = current_savings * (1 + expected_return) ** years_to_retirement
        
        if expected_return > 0:
            monthly_rate = expected_return / 12
            months = years_to_retirement * 12
            fv_contributions = monthly_contribution * (((1 + monthly_rate) ** months - 1) / monthly_rate)
        else:
            fv_contributions = monthly_contribution * years_to_retirement * 12
        
        total_at_retirement = fv_current + fv_contributions
        annual_income = total_at_retirement * 0.04  # 4% rule
        
        return {
            "total_at_retirement": total_at_retirement,
            "annual_income": annual_income,
            "monthly_income": annual_income / 12,
            "years_to_retirement": years_to_retirement
        }

# MCP Integration - Model Context Protocol Support
class MCPHandler:
    """Handle Model Context Protocol for external integrations"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.tools = {
            "get_market_data": self.get_market_data,
            "calculate_tax_impact": self.calculate_tax_impact,
            "get_investment_recommendations": self.get_investment_recommendations
        }
    
    async def get_market_data(self, symbol: str) -> Dict:
        """Get current market data (placeholder for real integration)"""
        return {
            "symbol": symbol,
            "price": 100.0,  # Mock data
            "change": 0.05,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def calculate_tax_impact(self, income: float, state: str = "US") -> Dict:
        """Calculate tax impact for financial planning"""
        # Simplified tax calculation
        federal_rate = 0.22 if income > 5000 else 0.12
        state_rate = 0.05  # Average state tax
        
        return {
            "federal_tax": income * 12 * federal_rate,
            "state_tax": income * 12 * state_rate,
            "effective_rate": federal_rate + state_rate,
            "after_tax_income": income * (1 - federal_rate - state_rate)
        }
    
    async def get_investment_recommendations(self, risk_profile: str, amount: float) -> List[Dict]:
        """Get investment recommendations based on risk profile"""
        recommendations = {
            "Conservative": [
                {"fund": "Vanguard Total Bond Market", "allocation": 0.6},
                {"fund": "Vanguard Total Stock Market", "allocation": 0.3},
                {"fund": "Cash/Money Market", "allocation": 0.1}
            ],
            "Moderate": [
                {"fund": "Vanguard Total Stock Market", "allocation": 0.6},
                {"fund": "Vanguard Total International Stock", "allocation": 0.2},
                {"fund": "Vanguard Total Bond Market", "allocation": 0.2}
            ],
            "Aggressive": [
                {"fund": "Vanguard Total Stock Market", "allocation": 0.7},
                {"fund": "Vanguard Total International Stock", "allocation": 0.2},
                {"fund": "Vanguard Emerging Markets", "allocation": 0.1}
            ]
        }
        
        return recommendations.get(risk_profile, recommendations["Moderate"])

# Factory function to create the complete agent system
async def create_agent_system() -> tuple[AgentCoordinator, DatabaseManager]:
    """Factory function to create and initialize the complete agent system"""
    
    # Initialize database manager
    db_manager = DatabaseManager(MONGODB_URL)
    
    # Create coordinator
    coordinator = AgentCoordinator(db_manager)
    
    # Create and register agents
    data_agent = DataValidationAgent(db_manager)
    financial_agent = FinancialAnalysisAgent(db_manager)
    
    coordinator.register_agent(data_agent)
    coordinator.register_agent(financial_agent)
    
    # Start agents
    await data_agent.start()
    await financial_agent.start()
    
    logger.info("Agent system initialized successfully")
    return coordinator, db_manager

if __name__ == "__main__":
    # Example usage for testing
    async def test_system():
        coordinator, db_manager = await create_agent_system()
        
        # Test profile
        test_profile = {
            "age": 30,
            "income": 5000,
            "fixed_expenses": 2500,
            "variable_expenses": 1500,
            "current_savings": 15000,
            "debt": 8000,
            "risk_tolerance": "Moderate",
            "financial_goals": ["Emergency Fund", "Retirement"],
            "goal_amounts": [30000, 500000],
            "goal_timelines": [2, 30],
            "employment_status": "Full-time",
            "family_size": 1
        }
        
        # Start analysis
        session_id = "test_session_123"
        request_id = await coordinator.start_analysis_session(test_profile, session_id)
        print(f"Started analysis with request ID: {request_id}")
        
        # In a real application, you'd wait for the analysis to complete
        # and retrieve the results
    
    # Run test
    # asyncio.run(test_system())