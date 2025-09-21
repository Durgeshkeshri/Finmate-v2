"""
Visualization Engine for AI Financial Advisor
Handles all data visualization, chart creation, and UI components

This file contains the visualization logic separated from the agent system
for better maintainability and performance.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Optional
import streamlit as st
from datetime import datetime, timedelta
import numpy as np


class FinancialChartGenerator:
    """Advanced chart generation for financial data"""
    
    def __init__(self):
        self.color_palette = {
            'primary': '#667eea',
            'secondary': '#764ba2', 
            'success': '#4CAF50',
            'warning': '#FF9800',
            'danger': '#f44336',
            'info': '#2196F3',
            'light': '#f8f9fa',
            'dark': '#343a40'
        }
        
        self.chart_configs = {
            'font_family': 'Arial, sans-serif',
            'title_size': 16,
            'axis_size': 12,
            'legend_size': 11
        }
    
    def create_budget_comparison_chart(self, profile_data: Dict, analysis_data: Dict) -> go.Figure:
        """Create enhanced budget comparison chart with 50/30/20 rule"""
        
        rule_allocation = analysis_data.get("budget_analysis", {})
        income = profile_data["income"]
        fixed = profile_data["fixed_expenses"]
        variable = profile_data["variable_expenses"]
        available = max(0, income - fixed - variable)
        
        # Calculate percentages
        fixed_pct = (fixed / income) * 100
        variable_pct = (variable / income) * 100
        available_pct = (available / income) * 100
        
        categories = ["Needs (Fixed)", "Wants (Variable)", "Savings/Debt Payment"]
        recommended = [
            rule_allocation.get("needs", income * 0.5),
            rule_allocation.get("wants", income * 0.3),
            rule_allocation.get("savings", income * 0.2)
        ]
        actual = [fixed, variable, available]
        
        fig = go.Figure()
        
        # Add recommended bars
        fig.add_trace(go.Bar(
            name='50/30/20 Recommended',
            x=categories,
            y=recommended,
            marker_color=self.color_palette['info'],
            text=[f"${x:,.0f}<br>({(x/income)*100:.0f}%)" for x in recommended],
            textposition='auto',
            opacity=0.7
        ))
        
        # Add actual bars
        fig.add_trace(go.Bar(
            name='Your Current',
            x=categories,
            y=actual,
            marker_color=[
                self.color_palette['danger'] if fixed > rule_allocation.get("needs", income * 0.5) else self.color_palette['success'],
                self.color_palette['warning'] if variable > rule_allocation.get("wants", income * 0.3) else self.color_palette['success'],
                self.color_palette['success'] if available >= rule_allocation.get("savings", income * 0.2) else self.color_palette['danger']
            ],
            text=[f"${x:,.0f}<br>({(x/income)*100:.0f}%)" for x in actual],
            textposition='auto'
        ))
        
        fig.update_layout(
            title="Budget Analysis: 50/30/20 Rule vs Your Spending",
            barmode='group',
            yaxis_title="Amount ($)",
            xaxis_title="Category",
            font=dict(family=self.chart_configs['font_family']),
            showlegend=True,
            height=400,
            margin=dict(t=50, b=50, l=50, r=50)
        )
        
        return fig
    
    def create_financial_health_gauge(self, health_analysis: Dict) -> go.Figure:
        """Create financial health score gauge"""
        
        score = health_analysis.get("score", 0)
        grade = health_analysis.get("grade", "F")
        
        # Determine color based on score
        if score >= 80:
            color = self.color_palette['success']
        elif score >= 60:
            color = self.color_palette['warning']
        else:
            color = self.color_palette['danger']
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=score,
            number={
            'font': {'size': 36},  # control size of number so it doesn’t crop
                },
            title={'text': f"Financial Health Score<br><span style='font-size:0.8em;color:gray'>Grade: {grade}</span>"},
            delta={'reference': 80, 'position': "top"},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': color, 'thickness': 0.3},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                
                'steps': [
                    {'range': [0, 60], 'color': '#ffebee'},
                    {'range': [60, 80], 'color': '#fff3e0'},
                    {'range': [80, 100], 'color': '#e8f5e8'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(
            height=400,
            font=dict(family=self.chart_configs['font_family'], size=12)
        )
        
        return fig
    
    def create_investment_allocation_pie(self, investment_strategy: Dict, profile_data: Dict) -> go.Figure:
        """Create investment allocation pie chart"""
        
        allocation = investment_strategy.get("asset_allocation", {})
        age = profile_data.get("age", 30)
        risk_tolerance = profile_data.get("risk_tolerance", "Moderate")
        
        labels = list(allocation.keys())
        values = list(allocation.values())
        colors = [self.color_palette['danger'], self.color_palette['info'], self.color_palette['success']]
        
        fig = go.Figure(data=[go.Pie(
            labels=[label.title() for label in labels],
            values=values,
            hole=.4,
            marker=dict(colors=colors, line=dict(color='#FFFFFF', width=2)),
            textinfo='label+percent',
            textposition='outside',
            textfont=dict(size=12)
        )])
        
        fig.update_layout(
            title=f"Recommended Investment Allocation<br><sub>Age {age} • {risk_tolerance} Risk Tolerance</sub>",
            annotations=[
                dict(
                    text=f'Age {age}<br>{risk_tolerance}', 
                    x=0.5, y=0.5, 
                    font_size=14, 
                    showarrow=False,
                    font=dict(color=self.color_palette['dark'])
                )
            ],
            font=dict(family=self.chart_configs['font_family']),
            showlegend=True,
            height=400
        )
        
        return fig
    
    def create_retirement_projection_chart(self, profile_data: Dict, investment_strategy: Dict) -> Optional[go.Figure]:
        """Create retirement savings projection timeline"""
        
        age = profile_data.get("age", 30)
        if age >= 65:
            return None
        
        current_savings = profile_data.get("current_savings", 0)
        monthly_investment = investment_strategy.get("monthly_investment", 0)
        
        # Calculate projection timeline
        retirement_age = 65
        years_range = list(range(age, retirement_age + 1, 2))  # Every 2 years
        
        portfolio_values = []
        contributions_total = []
        growth_total = []
        
        annual_return = 0.07  # 7% average return
        
        for target_age in years_range:
            years_invested = target_age - age
            
            if years_invested == 0:
                portfolio_values.append(current_savings)
                contributions_total.append(0)
                growth_total.append(0)
            else:
                # Future value of current savings
                fv_current = current_savings * (1 + annual_return) ** years_invested
                
                # Future value of monthly contributions
                months = years_invested * 12
                monthly_rate = annual_return / 12
                if monthly_rate > 0:
                    fv_contributions = monthly_investment * (((1 + monthly_rate) ** months - 1) / monthly_rate)
                else:
                    fv_contributions = monthly_investment * months
                
                total_value = fv_current + fv_contributions
                total_contributed = monthly_investment * months
                growth = total_value - current_savings - total_contributed
                
                portfolio_values.append(total_value)
                contributions_total.append(total_contributed)
                growth_total.append(growth)
        
        fig = go.Figure()
        
        # Add stacked area chart showing contributions vs growth
        fig.add_trace(go.Scatter(
            x=years_range,
            y=[current_savings] * len(years_range),
            mode='lines',
            name='Initial Savings',
            fill='tonexty',
            fillcolor='rgba(102, 126, 234, 0.3)',
            line=dict(color=self.color_palette['primary'], width=0)
        ))
        
        fig.add_trace(go.Scatter(
            x=years_range,
            y=[current_savings + contrib for contrib in contributions_total],
            mode='lines',
            name='Total Contributions',
            fill='tonexty',
            fillcolor='rgba(118, 75, 162, 0.3)',
            line=dict(color=self.color_palette['secondary'], width=0)
        ))
        
        fig.add_trace(go.Scatter(
            x=years_range,
            y=portfolio_values,
            mode='lines+markers',
            name='Total Portfolio Value',
            fill='tonexty',
            fillcolor='rgba(76, 175, 80, 0.3)',
            line=dict(color=self.color_palette['success'], width=3),
            marker=dict(size=6, color=self.color_palette['success'])
        ))
        
        # Add annotations for key milestones
        if portfolio_values:
            final_value = portfolio_values[-1]
            fig.add_annotation(
                x=retirement_age,
                y=final_value,
                text=f"${final_value/1000:.0f}K<br>at {retirement_age}",
                showarrow=True,
                arrowhead=2,
                arrowcolor=self.color_palette['success'],
                bgcolor="white",
                bordercolor=self.color_palette['success']
            )
        
        fig.update_layout(
            title=f"Retirement Savings Projection (${monthly_investment:,.0f}/month)",
            xaxis_title="Age",
            yaxis_title="Portfolio Value ($)",
            font=dict(family=self.chart_configs['font_family']),
            hovermode='x unified',
            height=400,
            showlegend=True
        )
        
        return fig
    
    def create_emergency_fund_progress(self, profile_data: Dict, emergency_fund_data: Dict) -> go.Figure:
        """Create emergency fund progress chart"""
        
        current_savings = profile_data.get("current_savings", 0)
        target_amount = emergency_fund_data.get("target_amount", 0)
        recommended_months = emergency_fund_data.get("recommended_months", 6)
        
        progress_pct = min(100, (current_savings / target_amount) * 100) if target_amount > 0 else 100
        remaining = max(0, target_amount - current_savings)
        
        fig = go.Figure()
        
        # Current progress
        fig.add_trace(go.Bar(
            x=['Emergency Fund Progress'],
            y=[current_savings],
            name='Current Savings',
            marker_color=self.color_palette['success'],
            text=f'${current_savings:,.0f}',
            textposition='auto'
        ))
        
        # Remaining needed
        if remaining > 0:
            fig.add_trace(go.Bar(
                x=['Emergency Fund Progress'],
                y=[remaining],
                name='Remaining Needed',
                marker_color=self.color_palette['warning'],
                text=f'${remaining:,.0f}',
                textposition='auto'
            ))
        
        fig.update_layout(
            title=f"Emergency Fund Progress: {progress_pct:.1f}% Complete<br><sub>Target: {recommended_months} months of expenses</sub>",
            barmode='stack',
            yaxis_title="Amount ($)",
            font=dict(family=self.chart_configs['font_family']),
            showlegend=True,
            height=400
        )
        
        return fig
    
    def create_cash_flow_waterfall(self, profile_data: Dict) -> go.Figure:
        """Create monthly cash flow waterfall chart"""
        
        income = profile_data["income"]
        fixed = -profile_data["fixed_expenses"]
        variable = -profile_data["variable_expenses"]
        remaining = income + fixed + variable
        
        fig = go.Figure(go.Waterfall(
            name="Monthly Cash Flow",
            orientation="v",
            measure=["absolute", "relative", "relative", "total"],
            x=["Monthly Income", "Fixed Expenses", "Variable Expenses", "Available for Savings"],
            textposition="outside",
            text=[f"${income:,.0f}", f"${abs(fixed):,.0f}", f"${abs(variable):,.0f}", f"${remaining:,.0f}"],
            y=[income, fixed, variable, remaining],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            increasing={"marker": {"color": self.color_palette['success']}},
            decreasing={"marker": {"color": self.color_palette['danger']}},
            totals={"marker": {"color": self.color_palette['info']}}
        ))
        
        fig.update_layout(
            title="Monthly Cash Flow Analysis",
            yaxis_title="Amount ($)",
            font=dict(family=self.chart_configs['font_family']),
            height=400
        )
        
        return fig
    
    def create_goal_timeline_chart(self, goals_data: List[Dict]) -> Optional[go.Figure]:
        """Create financial goals timeline scatter chart"""
        
        if not goals_data:
            return None
        
        fig = go.Figure()
        
        # Extract data for scatter plot
        timelines = [goal["timeline_years"] for goal in goals_data]
        amounts = [goal["target_amount"] for goal in goals_data]
        monthly_required = [goal["monthly_required"] for goal in goals_data]
        goal_names = [goal["goal"] for goal in goals_data]
        feasibility = [goal["feasibility"] for goal in goals_data]
        
        # Color mapping for feasibility
        color_map = {"High": self.color_palette['success'], 
                    "Medium": self.color_palette['warning'], 
                    "Low": self.color_palette['danger']}
        colors = [color_map.get(f, self.color_palette['info']) for f in feasibility]
        
        fig.add_trace(go.Scatter(
            x=timelines,
            y=amounts,
            mode='markers+text',
            marker=dict(
                size=[mr/50 for mr in monthly_required],  # Size based on monthly required
                color=colors,
                opacity=0.7,
                line=dict(width=2, color='white')
            ),
            text=goal_names,
            textposition="top center",
            hovertemplate='<b>%{text}</b><br>Target: $%{y:,.0f}<br>Timeline: %{x} years<br>Monthly: $%{customdata:,.0f}<extra></extra>',
            customdata=monthly_required
        ))
        
        fig.update_layout(
            title="Financial Goals Timeline vs Amount",
            xaxis_title="Years to Goal",
            yaxis_title="Target Amount ($)",
            font=dict(family=self.chart_configs['font_family']),
            height=400,
            showlegend=False
        )
        
        return fig
    
    def create_comprehensive_dashboard_charts(self, profile_data: Dict, analysis_data: Dict) -> Dict[str, go.Figure]:
        """Create all dashboard charts at once for better performance"""
        
        charts = {}
        
        try:
            # Budget comparison
            charts["budget_comparison"] = self.create_budget_comparison_chart(profile_data, analysis_data)
            
            # Financial health
            health_data = analysis_data.get("health_analysis", {})
            charts["health_score"] = self.create_financial_health_gauge(health_data)
            
            # Investment allocation
            investment_data = analysis_data.get("investment_strategy", {})
            charts["investment_allocation"] = self.create_investment_allocation_pie(investment_data, profile_data)
            
            # Retirement projection
            retirement_chart = self.create_retirement_projection_chart(profile_data, investment_data)
            if retirement_chart:
                charts["retirement_projection"] = retirement_chart
            
            # Emergency fund
            ef_data = analysis_data.get("emergency_fund", {})
            charts["emergency_fund"] = self.create_emergency_fund_progress(profile_data, ef_data)
            
            # Cash flow
            charts["cash_flow"] = self.create_cash_flow_waterfall(profile_data)
            
            # Goals timeline
            goals_data = analysis_data.get("goal_analysis", {}).get("goals", [])
            goals_chart = self.create_goal_timeline_chart(goals_data)
            if goals_chart:
                charts["goals_timeline"] = goals_chart
                
        except Exception as e:
            st.error(f"Error creating charts: {e}")
            
        return charts


class UIComponentManager:
    """Manage UI components and styling"""
    
    def __init__(self):
        self.apply_custom_styling()
    
    def apply_custom_styling(self):
        """Apply custom CSS styling to Streamlit"""
        
        st.markdown("""
        <style>
            /* Main container styling */
            .main > div {
                padding-top: 2rem;
            }
            
            /* Metric box styling with proper contrast */
            .stMetric {
                background-color: white;
                border: 1px solid #d1d5db;
                padding: 15px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                text-align: center;
            }
            
            /* Ensure all metric text is visible */
            .stMetric label, .stMetric [data-testid="stMetricValue"], 
            .stMetric [data-testid="stMetricDelta"] {
                color: #1f2937 !important;
                font-weight: 500;
            }
            
            /* Custom metric box */
            .custom-metric {
                background: white;
                padding: 15px;
                border-radius: 8px;
                border: 1px solid #d1d5db;
                text-align: center;
                margin: 5px 0;
            }
            
            .metric-title {
                color: #6b7280;
                font-size: 14px;
                font-weight: 500;
                margin-bottom: 5px;
            }
            
            .metric-value {
                color: #1f2937;
                font-size: 24px;
                font-weight: 600;
                margin-bottom: 5px;
            }
            
            .metric-subtitle {
                color: #6b7280;
                font-size: 12px;
            }
            
            /* Header styling */
            .gradient-header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 40px;
                border-radius: 15px;
                color: white;
                text-align: center;
                margin: 20px 0;
            }
            
            /* Dashboard sections */
            .dashboard-section {
                background: #f8f9fa;
                padding: 20px;
                border-radius: 10px;
                border: 1px solid #e9ecef;
                margin: 15px 0;
            }
            
            .dashboard-section h4 {
                color: #1f2937 !important;
                margin-bottom: 15px;
            }
            
            /* Quick stats container */
            .quick-stats-container {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 30px;
                border-radius: 15px;
                color: white;
                margin: 20px 0;
            }
            
            .stat-item {
                background: rgba(255, 255, 255, 0.1);
                padding: 20px;
                border-radius: 10px;
                text-align: center;
                margin: 10px 0;
                border: 1px solid rgba(255, 255, 255, 0.2);
            }
            
            .stat-value {
                font-size: 28px;
                font-weight: 600;
                margin-bottom: 5px;
                color: white;
            }
            
            .stat-label {
                font-size: 14px;
                opacity: 0.9;
                color: white;
            }
            
            /* Success/warning boxes */
            .success-box {
                background-color: #d4edda;
                border: 1px solid #c3e6cb;
                padding: 15px;
                border-radius: 8px;
                margin: 10px 0;
                color: #155724;
            }
            
            .warning-box {
                background-color: #fff3cd;
                border: 1px solid #ffeaa7;
                padding: 15px;
                border-radius: 8px;
                margin: 10px 0;
                color: #856404;
            }
            
            .info-box {
                background-color: #e3f2fd;
                border: 1px solid #bbdefb;
                padding: 15px;
                border-radius: 8px;
                margin: 10px 0;
                color: #0d47a1;
            }
            
            /* Step boxes for landing page */
            .step-box {
                text-align: center;
                padding: 25px;
                background: linear-gradient(145deg, #f0f2f5, #ffffff);
                border: 2px solid #e9ecef;
                border-radius: 12px;
                margin: 15px 0;
                color: #1f2937;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                transition: transform 0.2s ease;
            }
            
            .step-box:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 12px rgba(0,0,0,0.15);
            }
            
            .step-box h3 {
                font-size: 2em;
                margin-bottom: 10px;
                color: #667eea !important;
            }
            
            .step-box h4 {
                color: #1f2937 !important;
                margin-bottom: 10px;
                font-weight: 600;
            }
            
            .step-box p {
                color: #6b7280 !important;
                margin: 0;
                line-height: 1.4;
            }
            
            /* Call to action styling */
            .gradient-cta {
                background: linear-gradient(45deg, #4CAF50, #2196F3);
                padding: 30px;
                border-radius: 15px;
                text-align: center;
                color: white;
                margin: 20px 0;
            }
            
            /* Loading animation */
            .loading-animation {
                text-align: center;
                padding: 20px;
            }
            
            .loading-spinner {
                border: 4px solid #f3f3f3;
                border-top: 4px solid #3498db;
                border-radius: 50%;
                width: 50px;
                height: 50px;
                animation: spin 2s linear infinite;
                margin: 0 auto 20px auto;
            }
            
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            
            /* Responsive adjustments */
            @media (max-width: 768px) {
                .gradient-header {
                    padding: 20px;
                }
                
                .dashboard-section {
                    padding: 15px;
                }
                
                .metric-value {
                    font-size: 20px;
                }
            }
        </style>
        """, unsafe_allow_html=True)
    
    def create_custom_metric(self, title: str, value: str, subtitle: str = "", delta_color: str = "normal") -> str:
        """Create a custom metric HTML component"""
        
        color_class = ""
        if "positive" in delta_color or "success" in delta_color:
            color_class = "color: #059669;"
        elif "negative" in delta_color or "danger" in delta_color:
            color_class = "color: #dc2626;"
        
        return f"""
        <div class="custom-metric">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-subtitle" style="{color_class}">{subtitle}</div>
        </div>
        """
    
    def display_key_metrics_dashboard(self, profile_data: Dict, validation_data: Dict, analysis_data: Dict):
        """Display enhanced key metrics dashboard"""
        
        st.markdown('<div class="dashboard-section">', unsafe_allow_html=True)
        st.markdown("### 📊 Key Financial Metrics")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        # Savings Rate
        with col1:
            savings_rate = validation_data["structured_data"]["savings_rate"]
            delta_text = f"{savings_rate - 0.20:.1%} vs 20% target"
            delta_color = "success" if savings_rate >= 0.20 else "danger"
            
            st.markdown(self.create_custom_metric(
                "Savings Rate",
                f"{savings_rate:.1%}",
                delta_text
            ), unsafe_allow_html=True)
        
        # Financial Health Score
        with col2:
            health_score = analysis_data["health_analysis"]["score"]
            grade = analysis_data["health_analysis"]["grade"]
            
            st.markdown(self.create_custom_metric(
                "Financial Health",
                f"{health_score}/100",
                f"Grade: {grade}"
            ), unsafe_allow_html=True)
        
        # Monthly Surplus
        with col3:
            available = validation_data["structured_data"]["available_for_savings"]
            
            st.markdown(self.create_custom_metric(
                "Monthly Surplus",
                f"${available:,.0f}",
                f"${available * 12:,.0f}/year"
            ), unsafe_allow_html=True)
        
        # Debt-to-Income
        with col4:
            debt_ratio = validation_data["structured_data"]["debt_to_income"]
            status = "Excellent" if debt_ratio <= 0.1 else "Good" if debt_ratio <= 0.3 else "High"
            status_color = "success" if debt_ratio <= 0.3 else "danger"
            
            st.markdown(self.create_custom_metric(
                "Debt-to-Income",
                f"{debt_ratio:.1%}",
                status
            ), unsafe_allow_html=True)
        
        # Net Worth
        with col5:
            net_worth = validation_data["structured_data"]["net_worth_estimate"]
            
            st.markdown(self.create_custom_metric(
                "Est. Net Worth",
                f"${net_worth:,.0f}",
                "Assets minus debts"
            ), unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def create_quick_stats_section(self, profile_data: Dict, analysis_data: Dict, validation_data: Dict):
        """Create comprehensive quick stats section"""
        
        st.markdown('<div class="quick-stats-container">', unsafe_allow_html=True)
        st.markdown('<h2 style="color: white; text-align: center; margin-bottom: 25px;">📊 Your Financial Quick Stats</h2>', unsafe_allow_html=True)
        
        # First row of stats
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_expenses = profile_data["fixed_expenses"] + profile_data["variable_expenses"]
            emergency_months = profile_data["current_savings"] / total_expenses if total_expenses > 0 else 0
            st.markdown(f"""
            <div class="stat-item">
                <div class="stat-value">{emergency_months:.1f}</div>
                <div class="stat-label">Emergency Fund<br>(months covered)</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            age = profile_data["age"]
            if age < 65:
                retirement_years = 65 - age
                st.markdown(f"""
                <div class="stat-item">
                    <div class="stat-value">{retirement_years}</div>
                    <div class="stat-label">Years to<br>Retirement</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="stat-item">
                    <div class="stat-value">Retired</div>
                    <div class="stat-label">Retirement<br>Status</div>
                </div>
                """, unsafe_allow_html=True)
        
        with col3:
            investment_capacity = analysis_data.get("investment_strategy", {}).get("monthly_investment", 0)
            st.markdown(f"""
            <div class="stat-item">
                <div class="stat-value">${investment_capacity:,.0f}</div>
                <div class="stat-label">Monthly Investment<br>Capacity</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            expense_ratio = validation_data["structured_data"]["expense_ratio"]
            st.markdown(f"""
            <div class="stat-item">
                <div class="stat-value">{expense_ratio:.1%}</div>
                <div class="stat-label">Expense to<br>Income Ratio</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Second row of insights
        st.markdown('<h3 style="color: white; text-align: center; margin: 30px 0 20px 0;">💡 Key Insights</h3>', unsafe_allow_html=True)
        
        insights_col1, insights_col2 = st.columns(2)
        
        with insights_col1:
            st.markdown("""
            <div style="background: rgba(255, 255, 255, 0.1); padding: 20px; border-radius: 10px; border: 1px solid rgba(255, 255, 255, 0.2);">
                <h4 style="color: white; margin-bottom: 15px;">💰 Financial Strengths</h4>
            """, unsafe_allow_html=True)
            
            strengths = []
            if validation_data["structured_data"]["savings_rate"] >= 0.20:
                strengths.append("• Excellent 20%+ savings rate")
            if validation_data["structured_data"]["debt_to_income"] <= 0.3:
                strengths.append("• Manageable debt levels")
            if analysis_data["health_analysis"]["score"] >= 80:
                strengths.append("• Strong overall financial health")
            
            if not strengths:
                strengths = ["• Building financial foundation", "• Taking steps to improve"]
            
            for strength in strengths[:3]:
                st.markdown(f'<div style="color: white; margin: 5px 0;">{strength}</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with insights_col2:
            st.markdown("""
            <div style="background: rgba(255, 255, 255, 0.1); padding: 20px; border-radius: 10px; border: 1px solid rgba(255, 255, 255, 0.2);">
                <h4 style="color: white; margin-bottom: 15px;">🎯 Areas for Growth</h4>
            """, unsafe_allow_html=True)
            
            recommendations = analysis_data["health_analysis"]["recommendations"][:3]
            if not recommendations:
                recommendations = ["• Continue current practices", "• Review goals quarterly"]
            
            for rec in recommendations:
                clean_rec = rec.replace("• ", "")
                st.markdown(f'<div style="color: white; margin: 5px 0;">• {clean_rec}</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def create_loading_animation(self, message: str = "AI Agents are analyzing your financial data..."):
        """Create loading animation for agent processing"""
        
        return f"""
        <div class="loading-animation">
            <div class="loading-spinner"></div>
            <h3>{message}</h3>
            <p>Our specialized agents are working together to provide you with comprehensive insights...</p>
        </div>
        """
    
    def display_agent_communication_status(self, messages: List[Dict]):
        """Display real-time agent communication status"""
        
        st.markdown("### 🤖 Agent Communication Status")
        
        with st.expander("View Agent Messages", expanded=False):
            for msg in messages[-10:]:  # Show last 10 messages
                timestamp = msg.get("timestamp", "Unknown")
                sender = msg.get("sender", "Unknown")
                receiver = msg.get("receiver", "Unknown")
                msg_type = msg.get("message_type", "Unknown")
                
                st.markdown(f"""
                **{timestamp}**: `{sender}` → `{receiver}` 
                
                Message Type: `{msg_type}`
                
                ---
                """)
    
    def create_educational_content_section(self):
        """Create educational content section with interactive elements"""
        
        st.header("📚 Financial Education Center")
        
        educational_tabs = st.tabs(["Budgeting 101", "Emergency Funds", "Investment Basics", "Retirement Planning"])
        
        with educational_tabs[0]:
            st.markdown("""
            ### 💰 Budgeting Fundamentals
            
            **The 50/30/20 Rule Explained:**
            - **50% for Needs**: Essential expenses like rent, utilities, groceries, insurance
            - **30% for Wants**: Entertainment, dining out, hobbies, non-essential purchases  
            - **20% for Savings**: Emergency fund, retirement, investments, extra debt payments
            
            **Getting Started:**
            1. **Track Everything**: Use apps like Mint, YNAB, or simple spreadsheets for 2-4 weeks
            2. **Categorize**: Separate needs from wants (be honest!)
            3. **Adjust Gradually**: Don't make drastic cuts immediately
            4. **Automate**: Set up automatic transfers for savings
            
            **Pro Tips:**
            - Review and adjust monthly
            - Use the envelope method for discretionary spending
            - Start with small wins to build momentum
            """)
        
        with educational_tabs[1]:
            st.markdown("""
            ### 🚨 Emergency Fund Essentials
            
            **Why You Need One:**
            - Job loss or income reduction
            - Medical emergencies and healthcare costs
            - Major home or car repairs
            - Unexpected family emergencies
            
            **How Much to Save:**
            - **Minimum**: $1,000 starter emergency fund
            - **Standard**: 3-6 months of essential expenses
            - **Higher Risk Jobs**: 6-12 months (freelancers, contractors)
            - **Stable Employment**: 3-4 months may suffice
            
            **Where to Keep It:**
            - High-yield savings account (current rates: 4-5%)
            - Money market accounts
            - Short-term CDs (for portion of fund)
            - **NOT in**: Checking accounts, investment accounts, or cash at home
            
            **Building Your Fund:**
            - Start small: $25-50 per paycheck
            - Use windfalls: tax refunds, bonuses, gifts
            - Automate transfers on payday
            - Sell unused items for quick cash
            """)
        
        with educational_tabs[2]:
            st.markdown("""
            ### 📈 Investment Fundamentals
            
            **Core Investment Principles:**
            - **Start Early**: Time is your biggest advantage due to compound interest
            - **Diversification**: Don't put all eggs in one basket
            - **Low Costs**: Every 1% in fees costs you ~$30,000 over 30 years
            - **Stay Consistent**: Regular investing beats timing the market
            
            **Investment Account Priority:**
            1. **401(k) Match**: Free money from your employer (typically 3-6%)
            2. **High-Yield Savings**: For emergency fund
            3. **Roth IRA**: $6,500/year limit ($7,500 if 50+), tax-free growth
            4. **Max 401(k)**: $22,500/year limit ($30,000 if 50+)
            5. **Taxable Account**: For goals beyond retirement
            
            **Simple Investment Options:**
            - **Target-Date Funds**: Automatically adjusts as you age
            - **Total Stock Market Index**: Broad US market exposure
            - **Total International**: Global diversification
            - **Bond Index Funds**: Stability and income
            
            **Common Mistakes to Avoid:**
            - Trying to time the market
            - Picking individual stocks without research
            - Paying high fees for actively managed funds
            - Panic selling during market downturns
            """)
        
        with educational_tabs[3]:
            st.markdown("""
            ### 🏖️ Retirement Planning Basics
            
            **The Magic of Compound Interest:**
            - Starting at 25: $200/month → $525,000 by 65
            - Starting at 35: $200/month → $245,000 by 65
            - **Lesson**: Start as early as possible!
            
            **Retirement Savings Targets:**
            - **Age 30**: 1x annual salary saved
            - **Age 40**: 3x annual salary saved
            - **Age 50**: 6x annual salary saved
            - **Age 60**: 8x annual salary saved
            - **Age 67**: 10x annual salary saved
            
            **Key Strategies:**
            - **Increase Contributions**: Raise by 1% each year
            - **Employer Match**: Always contribute enough to get full match
            - **Roth vs Traditional**: Roth better if you expect higher tax rates later
            - **Rebalance**: Review and adjust portfolio annually
            
            **The 4% Rule:**
            - You can safely withdraw 4% of your retirement savings annually
            - Example: $1M saved → $40K/year in retirement income
            - Adjust for inflation and market conditions
            
            **Don't Forget:**
            - Social Security will provide some income
            - Healthcare costs increase in retirement
            - Consider long-term care insurance
            - Plan for 20-30 years of retirement
            """)
    
    def display_sample_analysis_preview(self):
        """Display sample analysis for landing page"""
        
        st.markdown("### 👀 Sample Analysis Preview")
        
        sample_col1, sample_col2 = st.columns(2)
        
        with sample_col1:
            # Sample budget chart
            sample_fig = go.Figure()
            sample_fig.add_trace(go.Bar(
                name='Recommended',
                x=['Needs', 'Wants', 'Savings'],
                y=[2500, 1500, 1000],
                marker_color='lightblue'
            ))
            sample_fig.add_trace(go.Bar(
                name='Your Current',
                x=['Needs', 'Wants', 'Savings'], 
                y=[2800, 1200, 1000],
                marker_color='darkblue'
            ))
            sample_fig.update_layout(
                title="Sample: Budget Analysis vs 50/30/20 Rule",
                barmode='group',
                height=350
            )
            st.plotly_chart(sample_fig, use_container_width=True)
        
        with sample_col2:
            # Sample health score
            sample_health = go.Figure(go.Indicator(
                mode="gauge+number",
                value=78,
                title={'text': "Sample: Financial Health Score"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "orange"},
                    'steps': [
                        {'range': [0, 60], 'color': "lightgray"},
                        {'range': [60, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "lightgreen"}
                    ]
                }
            ))
            sample_health.update_layout(height=350)
            st.plotly_chart(sample_health, use_container_width=True)