"""
MCP (Model Context Protocol) Integration for Data Fetching and Analysis
Handles user data retrieval, historical analysis, and context preparation
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
from pymongo.collection import Collection

logger = logging.getLogger(__name__)

@dataclass
class FinancialDataPoint:
    """Single financial data entry with date"""
    date: datetime
    income: float
    fixed_expenses: float
    variable_expenses: float
    current_savings: float
    debt: float
    risk_tolerance: str
    user_notes: str = ""

@dataclass
class AnalysisTrend:
    """Analysis of trends over time"""
    metric_name: str
    current_value: float
    previous_value: float
    change_amount: float
    change_percentage: float
    trend_direction: str
    significance: str

class MCPDataProcessor:
    """Process and analyze user financial data using MCP patterns"""
    
    def __init__(self, user_data_collection: Collection):
        self.user_data = user_data_collection
        self.analysis_cache = {}
    
    def fetch_user_context(self, user_id: str) -> Dict[str, Any]:
        """Fetch comprehensive user context for AI chat"""
        
        # Get all user data
        user_history = list(self.user_data.find(
            {"user_id": user_id}
        ).sort("data_date", -1))
        
        if not user_history:
            return {
                "has_data": False,
                "message": "No financial data found. Please complete your financial profile first."
            }
        
        latest_data = user_history[0]["financial_data"]
        
        # Calculate basic metrics
        context = {
            "has_data": True,
            "user_id": user_id,
            "data_points_count": len(user_history),
            "latest_update": user_history[0]["data_date"].strftime("%Y-%m-%d"),
            "current_profile": {
                "age": latest_data.get("age", 0),
                "monthly_income": latest_data.get("income", 0),
                "total_expenses": latest_data.get("fixed_expenses", 0) + latest_data.get("variable_expenses", 0),
                "savings_rate": self._calculate_savings_rate(latest_data),
                "debt_to_income": self._calculate_debt_to_income(latest_data),
                "risk_tolerance": latest_data.get("risk_tolerance", "Unknown"),
                "financial_goals": latest_data.get("financial_goals", [])
            }
        }
        
        # Add trend analysis if multiple data points
        if len(user_history) > 1:
            context["trends"] = self._analyze_trends(user_history)
            context["insights"] = self._generate_insights(user_history)
        
        return context
    
    def fetch_historical_comparison(self, user_id: str, days_back: int = 90) -> Dict[str, Any]:
        """Fetch and compare data over specified period"""
        
        cutoff_date = datetime.utcnow() - timedelta(days=days_back)
        
        recent_data = list(self.user_data.find({
            "user_id": user_id,
            "data_date": {"$gte": cutoff_date}
        }).sort("data_date", -1))
        
        if len(recent_data) < 2:
            return {
                "has_comparison": False,
                "message": f"Need at least 2 data points in the last {days_back} days for comparison"
            }
        
        latest = recent_data[0]["financial_data"]
        oldest_in_period = recent_data[-1]["financial_data"]
        
        comparison = {
            "has_comparison": True,
            "period_days": days_back,
            "data_points": len(recent_data),
            "comparison_metrics": {
                "income_change": latest.get("income", 0) - oldest_in_period.get("income", 0),
                "expense_change": (latest.get("fixed_expenses", 0) + latest.get("variable_expenses", 0)) - 
                                (oldest_in_period.get("fixed_expenses", 0) + oldest_in_period.get("variable_expenses", 0)),
                "savings_change": latest.get("current_savings", 0) - oldest_in_period.get("current_savings", 0),
                "debt_change": latest.get("debt", 0) - oldest_in_period.get("debt", 0)
            },
            "improvement_areas": self._identify_improvements(recent_data),
            "concerning_trends": self._identify_concerns(recent_data)
        }
        
        return comparison
    
    def prepare_ai_context_string(self, user_id: str) -> str:
        """Prepare formatted context string for AI chat"""
        
        context = self.fetch_user_context(user_id)
        
        if not context["has_data"]:
            return "User has no financial data on record. Encourage them to complete their financial profile first."
        
        profile = context["current_profile"]
        
        context_string = f"""
USER FINANCIAL CONTEXT:
========================

Basic Profile:
- Age: {profile['age']} years old
- Monthly Income: ${profile['monthly_income']:,.2f}
- Monthly Expenses: ${profile['total_expenses']:,.2f}
- Savings Rate: {profile['savings_rate']:.1%}
- Debt-to-Income Ratio: {profile['debt_to_income']:.1%}
- Risk Tolerance: {profile['risk_tolerance']}
- Financial Goals: {', '.join(profile['financial_goals']) if profile['financial_goals'] else 'None specified'}

Data History:
- Total data entries: {context['data_points_count']}
- Last updated: {context['latest_update']}
"""
        
        if "trends" in context:
            context_string += "\nRecent Trends:\n"
            for trend in context["trends"][:5]:  # Top 5 trends
                direction_emoji = "📈" if trend.trend_direction == "improving" else "📉" if trend.trend_direction == "declining" else "➡️"
                context_string += f"- {trend.metric_name}: {direction_emoji} {trend.change_percentage:+.1%} ({trend.significance})\n"
        
        if "insights" in context:
            context_string += f"\nKey Insights:\n"
            for insight in context["insights"][:3]:  # Top 3 insights
                context_string += f"- {insight}\n"
        
        context_string += "\nInstructions: Use this context to provide personalized financial advice. Be specific and reference their actual numbers when relevant."
        
        return context_string
    
    def analyze_data_changes(self, user_id: str, current_data: Dict, previous_data_date: datetime = None) -> Dict[str, Any]:
        """Analyze changes between current and previous data"""
        
        if previous_data_date is None:
            previous_data_date = datetime.utcnow() - timedelta(days=30)
        
        # Get previous data
        previous_entry = self.user_data.find_one({
            "user_id": user_id,
            "data_date": {"$lte": previous_data_date}
        }, sort=[("data_date", -1)])
        
        if not previous_entry:
            return {
                "has_previous": False,
                "message": "No previous data found for comparison"
            }
        
        previous_data = previous_entry["financial_data"]
        
        changes = {
            "has_previous": True,
            "comparison_date": previous_entry["data_date"],
            "days_between": (datetime.utcnow() - previous_entry["data_date"]).days,
            "changes": []
        }
        
        # Compare key metrics
        comparisons = [
            ("Monthly Income", current_data.get("income", 0), previous_data.get("income", 0)),
            ("Fixed Expenses", current_data.get("fixed_expenses", 0), previous_data.get("fixed_expenses", 0)),
            ("Variable Expenses", current_data.get("variable_expenses", 0), previous_data.get("variable_expenses", 0)),
            ("Current Savings", current_data.get("current_savings", 0), previous_data.get("current_savings", 0)),
            ("Total Debt", current_data.get("debt", 0), previous_data.get("debt", 0))
        ]
        
        for metric_name, current_val, previous_val in comparisons:
            if previous_val != 0:
                change_pct = ((current_val - previous_val) / abs(previous_val)) * 100
                change_amount = current_val - previous_val
                
                changes["changes"].append({
                    "metric": metric_name,
                    "current": current_val,
                    "previous": previous_val,
                    "change_amount": change_amount,
                    "change_percentage": change_pct,
                    "is_improvement": self._is_improvement(metric_name, change_amount)
                })
        
        return changes
    
    def get_goal_progress_analysis(self, user_id: str) -> Dict[str, Any]:
        """Analyze progress toward financial goals over time"""
        
        user_history = list(self.user_data.find(
            {"user_id": user_id}
        ).sort("data_date", -1).limit(12))  # Last 12 entries
        
        if not user_history:
            return {"has_data": False}
        
        goal_analysis = {
            "has_data": True,
            "analysis_period": f"{len(user_history)} data points",
            "goal_progress": []
        }
        
        latest = user_history[0]["financial_data"]
        goals = latest.get("financial_goals", [])
        goal_amounts = latest.get("goal_amounts", [])
        
        if not goals:
            return goal_analysis
        
        # Analyze savings growth for goal progress
        savings_history = [entry["financial_data"].get("current_savings", 0) for entry in user_history]
        
        if len(savings_history) > 1:
            monthly_savings_trend = np.mean(np.diff(savings_history[::-1]))  # Reverse to get chronological
            
            for i, goal in enumerate(goals):
                if i < len(goal_amounts):
                    target_amount = goal_amounts[i]
                    current_savings = latest.get("current_savings", 0)
                    
                    progress_pct = (current_savings / target_amount) * 100 if target_amount > 0 else 0
                    remaining = max(0, target_amount - current_savings)
                    
                    months_to_goal = remaining / monthly_savings_trend if monthly_savings_trend > 0 else float('inf')
                    
                    goal_analysis["goal_progress"].append({
                        "goal": goal,
                        "target_amount": target_amount,
                        "current_amount": current_savings,
                        "progress_percentage": progress_pct,
                        "remaining_amount": remaining,
                        "estimated_months_to_completion": months_to_goal,
                        "monthly_savings_trend": monthly_savings_trend
                    })
        
        return goal_analysis
    
    def _calculate_savings_rate(self, financial_data: Dict) -> float:
        """Calculate savings rate from financial data"""
        income = financial_data.get("income", 0)
        fixed = financial_data.get("fixed_expenses", 0)
        variable = financial_data.get("variable_expenses", 0)
        
        if income == 0:
            return 0.0
        
        available = income - fixed - variable
        return max(0, available / income)
    
    def _calculate_debt_to_income(self, financial_data: Dict) -> float:
        """Calculate debt-to-income ratio"""
        annual_income = financial_data.get("income", 0) * 12
        debt = financial_data.get("debt", 0)
        
        if annual_income == 0:
            return 0.0
        
        return debt / annual_income
    
    def _analyze_trends(self, user_history: List[Dict]) -> List[AnalysisTrend]:
        """Analyze trends in user financial data"""
        trends = []
        
        if len(user_history) < 2:
            return trends
        
        latest = user_history[0]["financial_data"]
        previous = user_history[1]["financial_data"]
        
        metrics = [
            ("Savings Rate", self._calculate_savings_rate(latest), self._calculate_savings_rate(previous)),
            ("Monthly Income", latest.get("income", 0), previous.get("income", 0)),
            ("Total Expenses", 
             latest.get("fixed_expenses", 0) + latest.get("variable_expenses", 0),
             previous.get("fixed_expenses", 0) + previous.get("variable_expenses", 0)),
            ("Current Savings", latest.get("current_savings", 0), previous.get("current_savings", 0)),
            ("Total Debt", latest.get("debt", 0), previous.get("debt", 0))
        ]
        
        for metric_name, current, prev in metrics:
            if prev != 0:
                change_amount = current - prev
                change_pct = (change_amount / abs(prev)) * 100
                
                if abs(change_pct) < 1:
                    trend_direction = "stable"
                    significance = "minimal"
                elif change_pct > 0:
                    if metric_name in ["Savings Rate", "Monthly Income", "Current Savings"]:
                        trend_direction = "improving"
                    else:
                        trend_direction = "increasing"
                    significance = "significant" if abs(change_pct) > 10 else "moderate"
                else:
                    if metric_name in ["Total Expenses", "Total Debt"]:
                        trend_direction = "improving"
                    else:
                        trend_direction = "declining"
                    significance = "significant" if abs(change_pct) > 10 else "moderate"
                
                trends.append(AnalysisTrend(
                    metric_name=metric_name,
                    current_value=current,
                    previous_value=prev,
                    change_amount=change_amount,
                    change_percentage=change_pct,
                    trend_direction=trend_direction,
                    significance=significance
                ))
        
        return sorted(trends, key=lambda x: abs(x.change_percentage), reverse=True)
    
    def _generate_insights(self, user_history: List[Dict]) -> List[str]:
        """Generate insights from user data trends"""
        insights = []
        
        if len(user_history) < 2:
            return insights
        
        trends = self._analyze_trends(user_history)
        
        for trend in trends[:3]:  # Top 3 trends
            if trend.significance in ["significant", "moderate"]:
                if trend.trend_direction == "improving":
                    insights.append(f"Great progress on {trend.metric_name} - {trend.change_percentage:+.1%} improvement")
                elif trend.trend_direction == "declining" and "Savings" in trend.metric_name:
                    insights.append(f"Consider reviewing {trend.metric_name} - down {abs(trend.change_percentage):.1%}")
                elif trend.trend_direction == "increasing" and "Expenses" in trend.metric_name:
                    insights.append(f"Monitor {trend.metric_name} - increased by {trend.change_percentage:.1%}")
        
        return insights
    
    def _identify_improvements(self, recent_data: List[Dict]) -> List[str]:
        """Identify improvement areas from recent data"""
        improvements = []
        
        if len(recent_data) < 2:
            return improvements
        
        latest = recent_data[0]["financial_data"]
        
        # Check savings rate
        savings_rate = self._calculate_savings_rate(latest)
        if savings_rate < 0.10:
            improvements.append("Increase savings rate to at least 10%")
        
        # Check debt-to-income
        debt_to_income = self._calculate_debt_to_income(latest)
        if debt_to_income > 0.30:
            improvements.append("Focus on debt reduction - current ratio is high")
        
        # Check expense ratios
        income = latest.get("income", 0)
        if income > 0:
            expense_ratio = (latest.get("fixed_expenses", 0) + latest.get("variable_expenses", 0)) / income
            if expense_ratio > 0.90:
                improvements.append("Review budget - expenses are consuming most of income")
        
        return improvements
    
    def _identify_concerns(self, recent_data: List[Dict]) -> List[str]:
        """Identify concerning trends in recent data"""
        concerns = []
        
        if len(recent_data) < 3:
            return concerns
        
        # Check for declining savings
        savings_trend = [entry["financial_data"].get("current_savings", 0) for entry in recent_data[:3]]
        if len(set(savings_trend)) > 1 and all(savings_trend[i] <= savings_trend[i+1] for i in range(len(savings_trend)-1)):
            concerns.append("Savings appear to be declining over recent periods")
        
        # Check for increasing debt
        debt_trend = [entry["financial_data"].get("debt", 0) for entry in recent_data[:3]]
        if len(set(debt_trend)) > 1 and all(debt_trend[i] >= debt_trend[i+1] for i in range(len(debt_trend)-1)):
            concerns.append("Debt levels are increasing - consider debt management strategy")
        
        return concerns
    
    def _is_improvement(self, metric_name: str, change_amount: float) -> bool:
        """Determine if a change represents an improvement"""
        positive_metrics = ["Monthly Income", "Current Savings"]
        negative_metrics = ["Fixed Expenses", "Variable Expenses", "Total Debt"]
        
        if metric_name in positive_metrics:
            return change_amount > 0
        elif metric_name in negative_metrics:
            return change_amount < 0
        else:
            return False