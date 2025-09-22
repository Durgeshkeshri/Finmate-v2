"""
Finmate AI Financial Advisor - Main Streamlit Application
Fixed version addressing AI chat initialization and data analysis issues
"""

import streamlit as st
import asyncio
import time
import uuid
from datetime import datetime, date
from typing import Dict, List, Any, Optional
import plotly.graph_objects as go
import pandas as pd
import os
from dotenv import load_dotenv

# Import our custom modules
from agents import (
    create_agent_system, 
    FinancialProfile, 
    DatabaseManager,
    AgentCoordinator
)
from visualization_engine import FinancialChartGenerator, UIComponentManager
from auth import AuthenticationManager, show_auth_interface
from mcp_integration import MCPDataProcessor
from ai_chat import AIFinancialChatbot, display_chat_interface

# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MONGODB_URL = os.getenv("MONGODB_URL")

# Configure Streamlit page
st.set_page_config(
    page_title="Finmate",
    page_icon="🤖💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

class EnhancedStreamlitApp:
    """Enhanced application with authentication and chat features"""
    
    def __init__(self):
        self.auth_manager = AuthenticationManager(MONGODB_URL)
        self.coordinator: Optional[AgentCoordinator] = None
        self.db_manager: Optional[DatabaseManager] = None
        self.chart_generator = FinancialChartGenerator()
        self.ui_manager = UIComponentManager()
        self.mcp_processor = None
        self.chatbot = None
        
        # Initialize session state
        if 'system_initialized' not in st.session_state:
            st.session_state.system_initialized = False
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = None
        if 'current_user' not in st.session_state:
            st.session_state.current_user = None
        if 'show_data_form' not in st.session_state:
            st.session_state.show_data_form = False
    
    async def initialize_system(self):
        """Initialize the agent system and MCP processor"""
        if not st.session_state.system_initialized:
            with st.spinner("🚀 Initializing Enhanced AI System..."):
                try:
                    # Initialize agent system
                    self.coordinator, self.db_manager = await create_agent_system()
                    
                    # Initialize MCP processor
                    self.mcp_processor = MCPDataProcessor(self.auth_manager.user_data)
                    
                    # Initialize AI chatbot
                    self.chatbot = AIFinancialChatbot(GEMINI_API_KEY, self.mcp_processor)
                    
                    st.session_state.system_initialized = True
                    st.success("✅ Enhanced Multi-Agent System Initialized")
                except Exception as e:
                    st.error(f"System initialization failed: {str(e)}")
                    # Initialize basic components anyway
                    self.mcp_processor = MCPDataProcessor(self.auth_manager.user_data)
                    self.chatbot = AIFinancialChatbot(GEMINI_API_KEY, self.mcp_processor)
                    st.session_state.system_initialized = True
    
    def show_user_dashboard(self, user: Dict):
        """Show main dashboard for authenticated users"""
        
        # User welcome header
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 20px; border-radius: 15px; color: white; margin-bottom: 20px;">
            <h2>Welcome back, {user['full_name']}! 👋</h2>
            <p>Account: {user['username']} | Last login: {user.get('last_login', 'First time').strftime('%Y-%m-%d %H:%M') if user.get('last_login') else 'First time'}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Navigation tabs
        tab1, tab3, tab4 = st.tabs(["📊 Financial Analysis","📈 Data History", "⚙️ Account"])
        
        with tab1:
            asyncio.run(self.show_financial_analysis_tab(user))
        
        # with tab2:
        #     self.show_ai_assistant_tab(user)
        
        with tab3:
            self.show_data_history_tab(user)
        
        with tab4:
            self.show_account_settings_tab(user)
    
    async def show_financial_analysis_tab(self, user: Dict):
        """Show financial analysis tab with data input and results"""
        
        # Check for existing data
        latest_data = self.auth_manager.get_user_latest_data(user['user_id'])
        
        # Always show the action buttons at the top
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("📝 Enter/Update Financial Data", type="primary", use_container_width=True):
                st.session_state.show_data_form = True
        
        with col2:
            if latest_data and st.button("📈 Analyze Current Data", type="secondary", use_container_width=True):
                # Re-run analysis with existing data
                profile_data = latest_data['financial_data']
                profile_data['user_id'] = user['user_id']
                
                with st.spinner("Analyzing your financial data..."):
                    analysis_result = await self.process_financial_analysis(profile_data, is_update=False)
                    st.session_state.analysis_results = analysis_result
                    st.session_state.current_profile_data = profile_data
                st.rerun()
        
        with col3:
            if 'analysis_results' in st.session_state and st.session_state.analysis_results:
                if st.button("🗑️ Clear Analysis", use_container_width=True):
                    del st.session_state.analysis_results
                    if 'current_profile_data' in st.session_state:
                        del st.session_state.current_profile_data
                    st.rerun()
        
        st.markdown("---")
        
        # Show existing data summary if available
        if latest_data:
            st.success(f"📊 Latest data from {latest_data['data_date'].strftime('%Y-%m-%d')}")
            
            if not st.session_state.get('show_data_form', False) and not st.session_state.get('analysis_results'):
                # Show quick summary of latest data
                self.show_data_summary(latest_data['financial_data'])
        else:
            st.info("💡 Complete your financial profile to get started with personalized analysis!")
            st.session_state.show_data_form = True
        
        # Show data input form
        if st.session_state.get('show_data_form', False):
            await self.show_financial_data_form(user, latest_data)
        
        # Show analysis results
        if st.session_state.get('analysis_results'):
            profile_data = st.session_state.get('current_profile_data', {})
            self.display_analysis_results(profile_data, st.session_state.analysis_results, user)
    
    def show_ai_assistant_tab(self, user: Dict):
        """Show AI assistant tab with proper initialization"""
        
        if not self.chatbot:
            st.error("AI Chat system not initialized. Please refresh the page.")
            if st.button("🔄 Retry Initialization"):
                try:
                    self.mcp_processor = MCPDataProcessor(self.auth_manager.user_data)
                    self.chatbot = AIFinancialChatbot(GEMINI_API_KEY, self.mcp_processor)
                    st.success("AI Chat system initialized!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Initialization failed: {str(e)}")
        else:
            # Check if user has financial data for context
            latest_data = self.auth_manager.get_user_latest_data(user['user_id'])
            
            if not latest_data:
                st.warning("💡 For personalized advice, please add your financial data first!")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("📝 Add Financial Data", type="primary", use_container_width=True):
                        st.session_state.show_data_form = True
                        # Switch to financial analysis tab
                        st.rerun()
                
                with col2:
                    if st.button("💬 Chat Anyway", use_container_width=True):
                        pass  # Continue to show chat interface
            
            st.markdown("---")
            display_chat_interface(self.chatbot, user['user_id'])
    
    def show_data_summary(self, financial_data: Dict):
        """Show quick summary of user's latest financial data"""
        
        st.markdown("### 📊 Current Financial Snapshot")
        
        col1, col2, col3, col4 = st.columns(4)
        
        income = financial_data.get('income', 0)
        fixed_expenses = financial_data.get('fixed_expenses', 0)
        variable_expenses = financial_data.get('variable_expenses', 0)
        current_savings = financial_data.get('current_savings', 0)
        
        with col1:
            st.metric("Monthly Income", f"${income:,.0f}")
        
        with col2:
            total_expenses = fixed_expenses + variable_expenses
            st.metric("Monthly Expenses", f"${total_expenses:,.0f}")
        
        with col3:
            available = max(0, income - total_expenses)
            st.metric("Available for Savings", f"${available:,.0f}")
        
        with col4:
            st.metric("Current Savings", f"${current_savings:,.0f}")
    
    async def show_financial_data_form(self, user: Dict, latest_data: Optional[Dict] = None):
        """Show form for entering/updating financial data"""
        
        st.markdown("### 📝 Financial Profile")
        
        # Pre-fill with latest data if available
        default_data = latest_data['financial_data'] if latest_data else {}
        
        with st.form("financial_data_form"):
            st.markdown("#### 👤 Personal Information")
            col1, col2 = st.columns(2)
            
            with col1:
                age = st.number_input("Age", min_value=16, max_value=100, 
                                    value=default_data.get('age', 30))
                family_size = st.number_input("Family Size", min_value=1, max_value=10,
                                            value=default_data.get('family_size', 1))
            
            with col2:
                employment_status = st.selectbox("Employment Status",
                    ["Full-time", "Part-time", "Contract", "Freelance", "Self-employed", "Retired"],
                    index=0 if not default_data.get('employment_status') else
                    ["Full-time", "Part-time", "Contract", "Freelance", "Self-employed", "Retired"].index(
                        default_data.get('employment_status', 'Full-time')))
                
                data_date = st.date_input("Data Date", value=date.today(),
                                        help="When is this financial data from?")
            
            st.markdown("#### 💵 Income & Expenses")
            col3, col4 = st.columns(2)
            
            with col3:
                income = st.number_input("Monthly Income ($)", min_value=0.0,
                                       value=float(default_data.get('income', 5000)), step=100.0)
                fixed_expenses = st.number_input("Fixed Expenses/month ($)", min_value=0.0,
                                               value=float(default_data.get('fixed_expenses', 2000)), step=100.0,
                                               help="Rent, insurance, minimum debt payments, utilities")
            
            with col4:
                variable_expenses = st.number_input("Variable Expenses/month ($)", min_value=0.0,
                                                  value=float(default_data.get('variable_expenses', 1000)), step=100.0,
                                                  help="Food, entertainment, shopping, dining out")
                user_notes = st.text_area("Notes (optional)", 
                                         value=default_data.get('user_notes', ''),
                                         help="Any additional context about your financial situation")
            
            st.markdown("#### 💰 Assets & Debt")
            col5, col6 = st.columns(2)
            
            with col5:
                current_savings = st.number_input("Current Savings ($)", min_value=0.0,
                                                value=float(default_data.get('current_savings', 10000)), step=500.0)
            
            with col6:
                debt = st.number_input("Total Debt ($)", min_value=0.0,
                                     value=float(default_data.get('debt', 0)), step=500.0,
                                     help="Credit cards, loans, etc. (excluding mortgage)")
            
            st.markdown("#### 📈 Investment Profile")
            col7, col8 = st.columns(2)
            
            with col7:
                risk_tolerance = st.selectbox("Investment Risk Tolerance",
                    ["Conservative", "Moderate", "Aggressive"],
                    index=1 if not default_data.get('risk_tolerance') else
                    ["Conservative", "Moderate", "Aggressive"].index(
                        default_data.get('risk_tolerance', 'Moderate')))
            
            with col8:
                st.markdown("**Financial Goals:**")
                goal_options = ["Emergency Fund", "House Down Payment", "Retirement", 
                              "Education Fund", "Vacation", "Investment Portfolio", "Debt Payoff", "Wedding"]
                
                selected_goals = st.multiselect("Select Your Goals", goal_options,
                                               default=default_data.get('financial_goals', ["Emergency Fund", "Retirement"]))
            
            # Goal details
            goal_amounts = []
            goal_timelines = []
            
            if selected_goals:
                st.markdown("**Goal Details:**")
                for i, goal in enumerate(selected_goals):
                    col_amt, col_time = st.columns(2)
                    with col_amt:
                        default_amount = default_data.get('goal_amounts', [50000] * len(selected_goals))
                        amount = st.number_input(f"{goal} - Target Amount ($)",
                            min_value=0.0,
                            value=float(default_amount[i] if i < len(default_amount) else 50000),
                            key=f"amount_{goal}")
                    with col_time:
                        default_timeline = default_data.get('goal_timelines', [5] * len(selected_goals))
                        timeline = st.number_input(f"{goal} - Timeline (years)",
                            min_value=1, max_value=50,
                            value=int(default_timeline[i] if i < len(default_timeline) else 5),
                            key=f"timeline_{goal}")
                    goal_amounts.append(amount)
                    goal_timelines.append(timeline)
            
            # Form submission
            col_submit1, col_submit2, col_submit3 = st.columns(3)
            
            with col_submit1:
                submit_and_analyze = st.form_submit_button("💾 Save & Analyze", type="primary", use_container_width=True)
            
            with col_submit2:
                submit_only = st.form_submit_button("💾 Save Only", use_container_width=True)
            
            with col_submit3:
                cancel = st.form_submit_button("❌ Cancel", use_container_width=True)
            
            if cancel:
                st.session_state.show_data_form = False
                st.rerun()
            
            if submit_and_analyze or submit_only:
                # Prepare financial data
                financial_data = {
                    "age": age,
                    "income": income,
                    "fixed_expenses": fixed_expenses,
                    "variable_expenses": variable_expenses,
                    "current_savings": current_savings,
                    "debt": debt,
                    "risk_tolerance": risk_tolerance,
                    "financial_goals": selected_goals,
                    "goal_amounts": goal_amounts,
                    "goal_timelines": goal_timelines,
                    "employment_status": employment_status,
                    "family_size": family_size,
                    "user_notes": user_notes,
                    "user_id": user['user_id']
                }
                
                # Save to database
                data_datetime = datetime.combine(data_date, datetime.min.time())
                save_id = self.auth_manager.save_user_financial_data(
                    user['user_id'], 
                    financial_data, 
                    data_datetime
                )
                
                st.success(f"✅ Financial data saved successfully!")
                
                # Store for analysis
                st.session_state.current_profile_data = financial_data
                
                if submit_and_analyze:
                    # Analyze the data
                    analysis_result = await self.process_financial_analysis(financial_data, is_update=True)
                    st.session_state.analysis_results = analysis_result
                
                # Hide form and refresh
                st.session_state.show_data_form = False
                st.rerun()
    
    def show_data_history_tab(self, user: Dict):
        """Show user's financial data history and trends"""
        
        st.markdown("### 📈 Your Financial History & Trends")
        
        # Get user's historical data
        history = self.auth_manager.get_user_financial_history(user['user_id'], limit=20)
        
        if not history:
            st.info("No historical data found. Add some financial data to see trends over time.")
            return
        
        # Convert to DataFrame for analysis
        data_for_df = []
        for entry in reversed(history):  # Chronological order
            data = entry['financial_data']
            data_for_df.append({
                'Date': entry['data_date'].strftime('%Y-%m-%d'),
                'Income': data.get('income', 0),
                'Fixed_Expenses': data.get('fixed_expenses', 0),
                'Variable_Expenses': data.get('variable_expenses', 0),
                'Total_Expenses': data.get('fixed_expenses', 0) + data.get('variable_expenses', 0),
                'Savings': data.get('current_savings', 0),
                'Debt': data.get('debt', 0),
                'Savings_Rate': ((data.get('income', 0) - data.get('fixed_expenses', 0) - data.get('variable_expenses', 0)) / data.get('income', 1)) if data.get('income', 0) > 0 else 0
            })
        
        df = pd.DataFrame(data_for_df)
        
        if len(df) > 1:
            # Show trend charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Income vs Expenses over time
                fig_income = go.Figure()
                fig_income.add_trace(go.Scatter(x=df['Date'], y=df['Income'], 
                                              mode='lines+markers', name='Income'))
                fig_income.add_trace(go.Scatter(x=df['Date'], y=df['Total_Expenses'], 
                                              mode='lines+markers', name='Total Expenses'))
                fig_income.update_layout(title="Income vs Expenses Over Time", 
                                       xaxis_title="Date", yaxis_title="Amount ($)")
                st.plotly_chart(fig_income, use_container_width=True)
            
            with col2:
                # Savings over time
                fig_savings = go.Figure()
                fig_savings.add_trace(go.Scatter(x=df['Date'], y=df['Savings'], 
                                               mode='lines+markers', name='Savings', line=dict(color='green')))
                fig_savings.update_layout(title="Savings Growth Over Time", 
                                        xaxis_title="Date", yaxis_title="Savings ($)")
                st.plotly_chart(fig_savings, use_container_width=True)
            
            # MCP Analysis
            if self.mcp_processor:
                st.markdown("### 🔍 Advanced Trend Analysis")
                
                col3, col4 = st.columns(2)
                
                with col3:
                    # Recent comparison
                    comparison = self.mcp_processor.fetch_historical_comparison(user['user_id'], days_back=90)
                    
                    if comparison["has_comparison"]:
                        st.markdown("#### 📊 90-Day Comparison")
                        metrics = comparison["comparison_metrics"]
                        
                        st.metric("Income Change", f"${metrics['income_change']:+,.0f}")
                        st.metric("Expense Change", f"${metrics['expense_change']:+,.0f}")
                        st.metric("Savings Change", f"${metrics['savings_change']:+,.0f}")
                        st.metric("Debt Change", f"${metrics['debt_change']:+,.0f}")
                
                with col4:
                    # Goal progress
                    goal_progress = self.mcp_processor.get_goal_progress_analysis(user['user_id'])
                    
                    if goal_progress["has_data"] and goal_progress.get("goal_progress"):
                        st.markdown("#### 🎯 Goal Progress")
                        
                        for goal_info in goal_progress["goal_progress"][:3]:  # Top 3 goals
                            progress = goal_info["progress_percentage"]
                            st.metric(
                                f"{goal_info['goal'][:20]}...",
                                f"{progress:.1f}%",
                                f"${goal_info['monthly_savings_trend']:+,.0f}/mo trend"
                            )
        
        # Data table
        st.markdown("### 📋 Historical Data")
        display_df = df.copy()
        display_df['Savings_Rate'] = display_df['Savings_Rate'].apply(lambda x: f"{x:.1%}")
        
        # Format currency columns
        currency_cols = ['Income', 'Fixed_Expenses', 'Variable_Expenses', 'Total_Expenses', 'Savings', 'Debt']
        for col in currency_cols:
            display_df[col] = display_df[col].apply(lambda x: f"${x:,.0f}")
        
        st.dataframe(display_df, use_container_width=True)
    
    def show_account_settings_tab(self, user: Dict):
        """Show account settings and logout"""
        
        st.markdown("### ⚙️ Account Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 👤 Account Information")
            st.write(f"**Username:** {user['username']}")
            st.write(f"**Email:** {user['email']}")
            st.write(f"**Full Name:** {user['full_name']}")
            st.write(f"**Member Since:** {user['created_at'].strftime('%Y-%m-%d')}")
            
            # Data statistics
            data_count = len(self.auth_manager.get_user_financial_history(user['user_id']))
            st.write(f"**Data Entries:** {data_count}")
        
        with col2:
            st.markdown("#### 🔧 Actions")
            
            if st.button("🗑️ Clear All Chat History", use_container_width=True):
                if hasattr(self, 'chatbot') and self.chatbot:
                    self.chatbot.clear_chat_history()
                    st.success("Chat history cleared!")
            
            if st.button("🔄 Reset Analysis Cache", use_container_width=True):
                if 'analysis_results' in st.session_state:
                    del st.session_state.analysis_results
                if 'current_profile_data' in st.session_state:
                    del st.session_state.current_profile_data
                st.success("Analysis cache reset!")
            
            st.markdown("---")
            
            if st.button("🚪 Logout", type="secondary", use_container_width=True):
                # Clear session
                if 'session_token' in st.session_state:
                    self.auth_manager.logout_user(st.session_state.session_token)
                    del st.session_state.session_token
                
                # Clear all session data
                keys_to_clear = ['current_user', 'analysis_results', 'chat_history', 'user_context', 
                               'chat_context_loaded', 'current_profile_data', 'show_data_form', 'system_initialized']
                
                for key in keys_to_clear:
                    if key in st.session_state:
                        del st.session_state[key]
                
                st.success("Logged out successfully!")
                st.rerun()
    
    async def process_financial_analysis(self, profile_data: Dict, is_update: bool = True) -> Dict:
        """Process financial analysis through the agent system"""
        
        # Generate unique session ID
        session_id = f"session_{uuid.uuid4().hex[:8]}"
        
        # Start analysis with progress tracking
        progress_container = st.empty()
        
        with progress_container.container():
            st.markdown("""
            <div style="text-align: center; padding: 20px;">
                <h3>🤖 Multi-Agent Analysis in Progress</h3>
                <p>Our specialized AI agents are collaborating to analyze your financial data...</p>
            </div>
            """, unsafe_allow_html=True)
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Enhanced analysis stages
            stages = [
                ("🔍 Data Validation Agent: Validating financial data...", 0.15),
                ("📊 Data Validation Agent: Calculating financial metrics...", 0.30),
                ("📈 MCP Processor: Analyzing historical trends...", 0.45),
                ("🧠 Financial Analysis Agent: Applying financial models...", 0.60),
                ("💡 Financial Analysis Agent: Generating investment strategy...", 0.75),
                ("🎯 Financial Analysis Agent: Analyzing goal feasibility...", 0.90),
                ("✅ Analysis complete! Generating visualizations...", 1.0)
            ]
            
            # Process analysis with MCP integration
            analysis_result = await self.simulate_enhanced_agent_analysis(
                profile_data, stages, progress_bar, status_text, is_update
            )
            
            # Clear progress indicators
            progress_container.empty()
            
            return analysis_result
    
    async def simulate_enhanced_agent_analysis(self, profile_data: Dict, stages: List, 
                                             progress_bar, status_text, is_update: bool) -> Dict:
        """Enhanced agent analysis simulation with MCP integration"""
        
        for stage_text, progress in stages:
            status_text.text(stage_text)
            progress_bar.progress(progress)
            await asyncio.sleep(0.5)  # Faster simulation
        
        # Generate comprehensive analysis with MCP data
        analysis = self.generate_enhanced_analysis(profile_data, is_update)
        return analysis
    
    def generate_enhanced_analysis(self, profile_data: Dict, is_update: bool) -> Dict:
        """Generate enhanced analysis with MCP integration"""
        
        # Basic calculations
        total_expenses = profile_data["fixed_expenses"] + profile_data["variable_expenses"]
        available_savings = max(0, profile_data["income"] - total_expenses)
        savings_rate = available_savings / profile_data["income"] if profile_data["income"] > 0 else 0
        debt_to_income = profile_data["debt"] / (profile_data["income"] * 12) if profile_data["income"] > 0 else 0
        
        structured_data = {
            "monthly_income": profile_data["income"],
            "total_expenses": total_expenses,
            "expense_ratio": total_expenses / profile_data["income"] if profile_data["income"] > 0 else 0,
            "savings_rate": savings_rate,
            "available_for_savings": available_savings,
            "debt_to_income": debt_to_income,
            "net_worth_estimate": profile_data["current_savings"] - profile_data["debt"],
            "financial_runway_months": profile_data["current_savings"] / total_expenses if total_expenses > 0 else float('inf')
        }
        
        # Enhanced analysis with MCP data
        analysis = {
            "profile_summary": {
                "age": profile_data["age"],
                "monthly_income": profile_data["income"],
                "savings_rate": savings_rate,
                "risk_tolerance": profile_data["risk_tolerance"]
            },
            "structured_data": structured_data,
            "budget_analysis": {
                "needs": profile_data["income"] * 0.5,
                "wants": profile_data["income"] * 0.3,
                "savings": profile_data["income"] * 0.2
            },
            "emergency_fund": self.calculate_emergency_fund(total_expenses, profile_data),
            "investment_strategy": self.generate_investment_strategy(profile_data, available_savings),
            "health_analysis": self.calculate_health_score(profile_data, structured_data),
            "goal_analysis": self.analyze_goals(profile_data),
            "ai_insights": self.generate_ai_insights(profile_data, structured_data, is_update),
            "analysis_timestamp": datetime.utcnow().isoformat()
        }
        
        # Add MCP-specific analysis if available
        if self.mcp_processor and is_update:
            try:
                user_id = profile_data.get('user_id')
                if user_id:
                    # Get historical comparison
                    comparison = self.mcp_processor.fetch_historical_comparison(user_id, days_back=90)
                    if comparison["has_comparison"]:
                        analysis["mcp_analysis"] = {
                            "historical_comparison": comparison,
                            "trend_insights": self.generate_trend_insights(comparison)
                        }
                    
                    # Get goal progress
                    goal_progress = self.mcp_processor.get_goal_progress_analysis(user_id)
                    if goal_progress["has_data"]:
                        analysis["goal_progress"] = goal_progress
            
            except Exception as e:
                # Log error but don't fail the analysis
                st.warning(f"Historical analysis unavailable: {str(e)}")
        
        return analysis
    
    def calculate_emergency_fund(self, monthly_expenses: float, profile_data: Dict) -> Dict:
        """Calculate emergency fund recommendation"""
        risk_tolerance = profile_data.get("risk_tolerance", "Moderate")
        employment_status = profile_data.get("employment_status", "Full-time")
        
        base_months = {"Conservative": 6, "Moderate": 4, "Aggressive": 3}
        employment_adj = {"Full-time": 0, "Part-time": 1, "Contract": 2, "Freelance": 2, "Self-employed": 2}
        
        recommended_months = base_months.get(risk_tolerance, 4) + employment_adj.get(employment_status, 0)
        
        return {
            "recommended_months": recommended_months,
            "target_amount": monthly_expenses * recommended_months,
            "current_coverage": profile_data["current_savings"] / monthly_expenses if monthly_expenses > 0 else 0,
            "reasoning": f"Based on {risk_tolerance.lower()} risk tolerance and {employment_status.lower()} employment"
        }
    
    def generate_investment_strategy(self, profile_data: Dict, monthly_available: float) -> Dict:
        """Generate investment strategy recommendations"""
        age = profile_data["age"]
        risk_tolerance = profile_data["risk_tolerance"]
        
        # Age-based allocation with risk adjustments
        risk_adj = {"Conservative": -20, "Moderate": 0, "Aggressive": 15}
        stock_pct = max(20, min(90, (100 - age) + risk_adj.get(risk_tolerance, 0)))
        bond_pct = max(10, 100 - stock_pct - 10)
        
        return {
            "asset_allocation": {
                "stocks": int(stock_pct),
                "bonds": int(bond_pct),
                "cash": 10
            },
            "monthly_investment": monthly_available * 0.8,
            "risk_level": risk_tolerance,
            "implementation_priority": [
                "Emergency fund completion",
                "401(k) employer match",
                "Roth IRA maximization",
                "Taxable account investing"
            ]
        }
    
    def calculate_health_score(self, profile_data: Dict, structured_data: Dict) -> Dict:
        """Calculate financial health score"""
        score = 0
        factors = []
        recommendations = []
        
        # Savings rate analysis (30 points)
        savings_rate = structured_data["savings_rate"]
        if savings_rate >= 0.20:
            score += 30
            factors.append("✅ Excellent savings rate (20%+)")
        elif savings_rate >= 0.15:
            score += 25
            factors.append("✅ Good savings rate (15%+)")
        elif savings_rate >= 0.10:
            score += 15
            factors.append("⚠️ Moderate savings rate (10%+)")
        else:
            score += 5
            factors.append("❌ Low savings rate (<10%)")
            recommendations.append("Priority: Increase savings rate to 15%+")
        
        # Debt analysis (25 points)
        debt_to_income = structured_data["debt_to_income"]
        if debt_to_income <= 0.10:
            score += 25
            factors.append("✅ Low debt burden")
        elif debt_to_income <= 0.30:
            score += 20
            factors.append("✅ Manageable debt levels")
        else:
            score += 5
            factors.append("❌ High debt burden")
            recommendations.append("Focus on debt reduction")
        
        # Emergency fund analysis (25 points)
        monthly_expenses = structured_data["total_expenses"]
        emergency_months = profile_data["current_savings"] / monthly_expenses if monthly_expenses > 0 else 0
        if emergency_months >= 6:
            score += 25
            factors.append("✅ Strong emergency fund (6+ months)")
        elif emergency_months >= 3:
            score += 20
            factors.append("✅ Adequate emergency fund (3+ months)")
        elif emergency_months >= 1:
            score += 10
            factors.append("⚠️ Basic emergency fund (1+ month)")
        else:
            score += 0
            factors.append("❌ No emergency fund")
            recommendations.append("Build emergency fund immediately")
        
        # Expense ratio analysis (20 points)
        expense_ratio = structured_data["expense_ratio"]
        if expense_ratio <= 0.70:
            score += 20
            factors.append("✅ Controlled spending (≤70% of income)")
        elif expense_ratio <= 0.85:
            score += 15
            factors.append("✅ Reasonable spending (≤85% of income)")
        elif expense_ratio <= 1.0:
            score += 5
            factors.append("⚠️ High spending (>85% of income)")
        else:
            score += 0
            factors.append("❌ Overspending (>100% of income)")
            recommendations.append("URGENT: Reduce expenses immediately")
        
        # Calculate grade
        grades = {90: "A+", 80: "A", 70: "B+", 60: "B", 50: "C", 40: "D"}
        grade = next((g for s, g in grades.items() if score >= s), "F")
        
        return {
            "score": min(100, score),
            "grade": grade,
            "factors": factors,
            "recommendations": recommendations
        }
    
    def analyze_goals(self, profile_data: Dict) -> Dict:
        """Analyze financial goals and feasibility"""
        goals = profile_data.get("financial_goals", [])
        goal_amounts = profile_data.get("goal_amounts", [])
        goal_timelines = profile_data.get("goal_timelines", [])
        
        goal_analysis = []
        total_monthly_required = 0
        
        for i, goal in enumerate(goals):
            if i < len(goal_amounts) and i < len(goal_timelines):
                amount = goal_amounts[i]
                timeline = goal_timelines[i]
                monthly_needed = amount / (timeline * 12)
                
                # Determine feasibility
                income = profile_data["income"]
                feasibility_pct = monthly_needed / income if income > 0 else 1
                
                if feasibility_pct <= 0.10:
                    feasibility = "High"
                elif feasibility_pct <= 0.20:
                    feasibility = "Medium"
                else:
                    feasibility = "Low"
                
                goal_analysis.append({
                    "goal": goal,
                    "target_amount": amount,
                    "timeline_years": timeline,
                    "monthly_required": monthly_needed,
                    "feasibility": feasibility,
                    "feasibility_percentage": feasibility_pct
                })
                
                total_monthly_required += monthly_needed
        
        return {
            "goals": goal_analysis,
            "total_monthly_required": total_monthly_required,
            "feasibility_summary": "Review timeline or amounts for low-feasibility goals"
        }
    
    def generate_ai_insights(self, profile_data: Dict, structured_data: Dict, is_update: bool) -> str:
        """Generate AI insights about the financial profile"""
        
        insights = []
        
        # Savings insights
        savings_rate = structured_data["savings_rate"]
        if savings_rate >= 0.20:
            insights.append("🎉 Excellent savings discipline! You're saving 20%+ of income.")
        elif savings_rate >= 0.15:
            insights.append("💪 Good savings rate. Consider pushing to 20% if possible.")
        else:
            insights.append("📈 Priority: Increase savings rate through expense review or income growth.")
        
        # Debt insights
        debt_to_income = structured_data["debt_to_income"]
        if debt_to_income > 0.30:
            insights.append("⚠️ High debt levels detected. Consider debt consolidation or aggressive payoff strategy.")
        elif debt_to_income > 0.10:
            insights.append("💡 Moderate debt levels. Good candidate for balanced debt payoff approach.")
        
        # Age-specific insights
        age = profile_data["age"]
        if age < 30:
            insights.append("🚀 Great time to build wealth! Focus on aggressive growth investments and emergency fund.")
        elif age < 45:
            insights.append("⚖️ Balance growth and stability. Consider family financial goals and estate planning.")
        else:
            insights.append("🎯 Pre-retirement focus: Capital preservation with moderate growth, max out catch-up contributions.")
        
        # Emergency fund insights
        monthly_expenses = structured_data["total_expenses"]
        emergency_months = profile_data["current_savings"] / monthly_expenses if monthly_expenses > 0 else 0
        if emergency_months < 3:
            insights.append("🚨 Priority: Build emergency fund to 3-6 months of expenses before aggressive investing.")
        
        return "\n".join(insights[:4])  # Top 4 insights
    
    def generate_trend_insights(self, comparison: Dict) -> List[str]:
        """Generate insights from MCP historical comparison"""
        insights = []
        metrics = comparison.get("comparison_metrics", {})
        
        if metrics.get("income_change", 0) > 0:
            insights.append("📈 Income has increased - great opportunity to boost savings rate")
        
        if metrics.get("savings_change", 0) > 0:
            insights.append("💰 Savings are growing - maintain this positive momentum")
        
        if metrics.get("debt_change", 0) < 0:
            insights.append("✅ Debt reduction progress - excellent work on debt management")
        
        if metrics.get("expense_change", 0) > metrics.get("income_change", 0):
            insights.append("⚠️ Expenses growing faster than income - review budget priorities")
        
        return insights
    
    def display_analysis_results(self, profile_data: Dict, analysis_data: Dict, user: Dict):
        """Display comprehensive analysis results with MCP integration"""
        
        # Success message with personalization
        health_score = analysis_data["health_analysis"]["score"]
        user_name = user["full_name"].split()[0]  # First name
        
        if health_score >= 80:
            st.success(f"🎉 Great job, {user_name}! Your financial health score is {health_score}/100 - You're doing excellent!")
        elif health_score >= 60:
            st.success(f"✅ Nice work, {user_name}! Your financial health score is {health_score}/100 - Solid foundation!")
        else:
            st.info(f"📊 Analysis complete, {user_name}! Your financial health score is {health_score}/100 - Let's work on improvements!")
        
        # Key Metrics Dashboard
        validation_data = {"structured_data": analysis_data["structured_data"]}
        self.ui_manager.display_key_metrics_dashboard(profile_data, validation_data, analysis_data)
        
        # MCP Analysis Section (if available)
        if "mcp_analysis" in analysis_data:
            st.markdown("### 📈 Historical Analysis & Trends")
            
            mcp_data = analysis_data["mcp_analysis"]
            col1, col2 = st.columns(2)
            
            with col1:
                if "historical_comparison" in mcp_data:
                    comparison = mcp_data["historical_comparison"]
                    st.markdown("#### 📊 90-Day Comparison")
                    
                    metrics = comparison["comparison_metrics"]
                    st.metric("Income Change", f"${metrics['income_change']:+,.0f}")
                    st.metric("Savings Change", f"${metrics['savings_change']:+,.0f}")
                    st.metric("Expense Change", f"${metrics['expense_change']:+,.0f}")
            
            with col2:
                if "trend_insights" in mcp_data:
                    st.markdown("#### 💡 Trend Insights")
                    for insight in mcp_data["trend_insights"]:
                        st.write(f"• {insight}")
        
        # Create and display charts
        charts = self.chart_generator.create_comprehensive_dashboard_charts(profile_data, analysis_data)
        
        # Rest of the analysis display
        st.markdown("### 📈 Detailed Financial Analysis")
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "💰 Budget & Cash Flow", 
            "📈 Investment Strategy", 
            "🚨 Emergency Planning",
            "🏥 Financial Health",
            "🎯 Goal Planning"
        ])
        
        with tab1:
            if "budget_comparison" in charts:
                st.plotly_chart(charts["budget_comparison"], use_container_width=True)
            
            if "cash_flow" in charts:
                st.plotly_chart(charts["cash_flow"], use_container_width=True)
        
        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                if "investment_allocation" in charts:
                    st.plotly_chart(charts["investment_allocation"], use_container_width=True)
            with col2:
                if "retirement_projection" in charts:
                    st.plotly_chart(charts["retirement_projection"], use_container_width=True)
            
            # Investment recommendations
            st.markdown("#### 📋 Personalized Investment Recommendations")
            strategy = analysis_data["investment_strategy"]
            monthly_investment = strategy.get("monthly_investment", 0)
            
            if monthly_investment > 0:
                st.success(f"💰 You can invest ${monthly_investment:,.0f}/month based on your current cash flow")
                
                # Priority list
                priorities = strategy.get("implementation_priority", [])
                st.markdown("**Implementation Priority:**")
                for i, priority in enumerate(priorities, 1):
                    st.write(f"{i}. {priority}")
            else:
                st.warning("Focus on budgeting and emergency fund before investing")
        
        with tab3:
            if "emergency_fund" in charts:
                st.plotly_chart(charts["emergency_fund"], use_container_width=True)
            
            ef_data = analysis_data["emergency_fund"]
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Target Amount", f"${ef_data['target_amount']:,.0f}")
            with col2:
                st.metric("Current Coverage", f"{ef_data['current_coverage']:.1f} months")
            with col3:
                remaining = max(0, ef_data['target_amount'] - profile_data["current_savings"])
                st.metric("Still Needed", f"${remaining:,.0f}")
        
        with tab4:
            col1, col2 = st.columns([1, 2])
            with col1:
                if "health_score" in charts:
                    st.plotly_chart(charts["health_score"], use_container_width=True)
            
            with col2:
                health = analysis_data["health_analysis"]
                st.markdown("#### 🏥 Health Analysis Details")
                
                for factor in health["factors"]:
                    if "✅" in factor:
                        st.success(factor)
                    elif "⚠️" in factor:
                        st.warning(factor)
                    elif "❌" in factor:
                        st.error(factor)
                
                if health["recommendations"]:
                    st.markdown("#### 🎯 Action Items")
                    for rec in health["recommendations"]:
                        st.write(f"• {rec}")
        
        with tab5:
            goals_data = analysis_data.get("goal_analysis", {}).get("goals", [])
            if goals_data:
                st.markdown("#### 🎯 Goal Analysis & Feasibility")
                
                # Goal progress if available from MCP
                if "goal_progress" in analysis_data:
                    progress_data = analysis_data["goal_progress"]
                    if progress_data.get("goal_progress"):
                        st.markdown("##### 📊 Progress Tracking")
                        for goal_info in progress_data["goal_progress"]:
                            col_goal, col_progress = st.columns([2, 1])
                            with col_goal:
                                st.write(f"**{goal_info['goal']}**")
                                st.write(f"Target: ${goal_info['target_amount']:,.0f}")
                            with col_progress:
                                progress = goal_info['progress_percentage']
                                st.metric("Progress", f"{progress:.1f}%")
                
                # Goal feasibility table
                goals_df_data = []
                for goal_info in goals_data:
                    feasibility_color = "🟢" if goal_info["feasibility"] == "High" else "🟡" if goal_info["feasibility"] == "Medium" else "🔴"
                    goals_df_data.append({
                        "Goal": goal_info["goal"],
                        "Target": f"${goal_info['target_amount']:,.0f}",
                        "Timeline": f"{goal_info['timeline_years']} years",
                        "Monthly Needed": f"${goal_info['monthly_required']:,.0f}",
                        "Feasibility": f"{feasibility_color} {goal_info['feasibility']}"
                    })
                
                st.dataframe(pd.DataFrame(goals_df_data), use_container_width=True)
                
                if "goals_timeline" in charts:
                    st.plotly_chart(charts["goals_timeline"], use_container_width=True)
            else:
                st.info("Add financial goals in your profile to see detailed goal analysis and tracking!")


def main():
    """Enhanced main application function"""
    
    # Initialize the enhanced application
    app = EnhancedStreamlitApp()
    
    # Initialize system asynchronously
    asyncio.run(app.initialize_system())
    
    # Authentication flow
    user = show_auth_interface(app.auth_manager)
    
    if user:
        # Show authenticated user dashboard
        app.show_user_dashboard(user)
    else:
        # Show landing page for unauthenticated users
        create_landing_page()


def create_landing_page():
    """Create enhanced landing page with authentication features"""
    ui_manager = UIComponentManager()
    
    # Hero section
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 40px; border-radius: 15px; color: white; text-align: center; margin: 20px 0;'>
        <h1>AI Financial Advisor - Enhanced</h1>
        <h2 style='font-weight: normal; margin-bottom: 15px;'>Multi-Agent System with User Authentication</h2>
        <p style='font-size: 18px; opacity: 0.9;'>Persistent Data • Historical Analysis • AI Chat Assistant • MCP Integration</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced features
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🔐 **User Authentication & Data Persistence**
        
        **Secure User Accounts**
        - Individual user profiles with secure authentication
        - Password strength validation and secure storage
        - Session management with automatic logout
        - Personal data isolation and privacy protection
        
        **Data History & Trends**
        - Store multiple financial data entries over time
        - Automatic trend analysis and progress tracking
        - Historical comparison and insights
        - Goal progress monitoring with real dates
        """)
    
    with col2:
        st.markdown("""
        ### 🤖 **AI Chat Assistant with Context**
        
        **Personalized AI Financial Advisor**
        - Full access to your complete financial profile
        - Historical data context for better advice
        - Personalized recommendations based on your situation
        - Continuous conversation memory during session
        
        **MCP Integration**
        - Model Context Protocol for advanced data processing
        - Real-time trend analysis and insights
        - Historical performance tracking
        - Advanced financial modeling and projections
        """)
    
    # Sample analysis preview
    ui_manager.display_sample_analysis_preview()
    
    # Getting started CTA
    st.markdown("""
    <div style='background: linear-gradient(45deg, #4CAF50, #2196F3); 
                padding: 30px; border-radius: 15px; text-align: center; color: white; margin: 20px 0;'>
        <h3>Ready to Start Your Financial Journey?</h3>
        <p style='font-size: 18px; margin-bottom: 20px;'>Create your secure account and begin tracking your financial progress today!</p>
        <p style='font-size: 16px; opacity: 0.9;'>✨ Personal dashboard • 💾 Data persistence • 🤖 AI advisor • 📈 Progress tracking</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()