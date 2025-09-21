"""
AI Chat System with User Financial Context
Provides personalized financial advice using user's data and chat history
"""

import streamlit as st
import google.generativeai as genai
from typing import List, Dict, Optional
from datetime import datetime
import json
import os
from mcp_integration import MCPDataProcessor

class AIFinancialChatbot:
    """AI-powered financial chatbot with user context"""
    
    def __init__(self, api_key: str, mcp_processor: MCPDataProcessor):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')
        self.mcp_processor = mcp_processor
        
        # Initialize chat session in Streamlit session state
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        if 'chat_context_loaded' not in st.session_state:
            st.session_state.chat_context_loaded = False
    
    def load_user_context(self, user_id: str) -> str:
        """Load user's financial context for AI chat"""
        try:
            context_string = self.mcp_processor.prepare_ai_context_string(user_id)
            st.session_state.user_context = context_string
            st.session_state.chat_context_loaded = True
            return context_string
        except Exception as e:
            return f"Error loading user context: {str(e)}"
    
    def get_system_prompt(self, user_context: str) -> str:
        """Generate system prompt with user context"""
        return f"""You are an expert AI Financial Advisor with access to the user's complete financial profile and history. 
        
{user_context}

IMPORTANT GUIDELINES:
1. Always reference specific numbers from their profile when giving advice
2. Be encouraging but realistic about their financial situation
3. Provide actionable, specific recommendations they can implement
4. If they ask about general topics, try to relate it back to their situation
5. Keep responses concise but comprehensive
6. Use their actual data to calculate examples and projections
7. Be supportive of their progress and honest about areas needing improvement

Remember: You have their complete financial context above. Use it to provide personalized advice rather than generic financial tips."""

    def send_message(self, user_message: str, user_id: str) -> str:
        """Send message to AI and get response with user context"""
        try:
            # Load user context if not already loaded
            if not st.session_state.chat_context_loaded:
                self.load_user_context(user_id)
            
            # Prepare conversation history for context
            conversation_context = ""
            if st.session_state.chat_history:
                recent_history = st.session_state.chat_history[-6:]  # Last 3 exchanges
                for msg in recent_history:
                    role = "User" if msg["role"] == "user" else "Assistant"
                    conversation_context += f"{role}: {msg['content']}\n"
            
            # Create full prompt with system context and conversation history
            full_prompt = f"""{self.get_system_prompt(st.session_state.get('user_context', ''))}

CONVERSATION HISTORY:
{conversation_context}

Current User Message: {user_message}

Please respond as their personal financial advisor, using their specific financial data and context:"""
            
            # Generate response
            response = self.model.generate_content(full_prompt)
            ai_response = response.text
            
            # Add to chat history
            st.session_state.chat_history.extend([
                {"role": "user", "content": user_message, "timestamp": datetime.now()},
                {"role": "assistant", "content": ai_response, "timestamp": datetime.now()}
            ])
            
            # Limit chat history to last 50 messages to prevent context overflow
            if len(st.session_state.chat_history) > 50:
                st.session_state.chat_history = st.session_state.chat_history[-50:]
            
            return ai_response
            
        except Exception as e:
            return f"I apologize, but I'm experiencing technical difficulties. Please try again. Error: {str(e)}"
    
    def get_suggested_questions(self, user_id: str) -> List[str]:
        """Generate suggested questions based on user's financial profile"""
        try:
            context = self.mcp_processor.fetch_user_context(user_id)
            
            if not context["has_data"]:
                return [
                    "How should I start building an emergency fund?",
                    "What's the best way to create a budget?",
                    "How much should I save for retirement?",
                    "What investment options are good for beginners?"
                ]
            
            profile = context["current_profile"]
            suggestions = []
            
            # Customize suggestions based on user's situation
            if profile["savings_rate"] < 0.15:
                suggestions.append("How can I increase my savings rate?")
            
            if profile["debt_to_income"] > 0.3:
                suggestions.append("What's the best strategy to pay off my debt?")
            
            if "Emergency Fund" in profile["financial_goals"]:
                suggestions.append("How close am I to my emergency fund goal?")
            
            if "Retirement" in profile["financial_goals"]:
                suggestions.append("Am I on track for retirement at my current savings rate?")
            
            # Add age-specific suggestions
            if profile["age"] < 30:
                suggestions.append("What financial milestones should I hit in my 20s?")
            elif profile["age"] < 45:
                suggestions.append("How should I balance saving for kids' education and retirement?")
            else:
                suggestions.append("Should I adjust my investment strategy as I get closer to retirement?")
            
            # Generic helpful questions
            suggestions.extend([
                "Based on my profile, what's my biggest financial priority?",
                "Can you analyze my spending patterns?",
                "What investment allocation do you recommend for me?"
            ])
            
            return suggestions[:6]  # Return top 6 suggestions
            
        except Exception as e:
            return ["How can I improve my financial situation?"]
    
    def clear_chat_history(self):
        """Clear the current chat session"""
        st.session_state.chat_history = []
        st.session_state.chat_context_loaded = False
        if 'user_context' in st.session_state:
            del st.session_state.user_context

def display_chat_interface(chatbot: AIFinancialChatbot, user_id: str):
    """Display the chat interface in Streamlit"""
    
    st.markdown("### 💬 AI Financial Assistant")
    st.markdown("Ask me anything about your finances - I have access to your complete financial profile!")
    
    # Chat controls
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("🔄 Refresh Context", help="Reload your latest financial data"):
            chatbot.load_user_context(user_id)
            st.success("Context refreshed!")
    
    with col2:
        if st.button("🗑️ Clear Chat", help="Start a new conversation"):
            chatbot.clear_chat_history()
            st.success("Chat cleared!")
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        if st.session_state.chat_history:
            for message in st.session_state.chat_history:
                if message["role"] == "user":
                    with st.chat_message("user"):
                        st.write(message["content"])
                else:
                    with st.chat_message("assistant"):
                        st.write(message["content"])
        else:
            with st.chat_message("assistant"):
                st.write("Hello! I'm your AI Financial Assistant. I have access to your financial profile and can help you with personalized advice. What would you like to know?")
    
    # Suggested questions
    if not st.session_state.chat_history or len(st.session_state.chat_history) < 4:
        suggestions = chatbot.get_suggested_questions(user_id)
        if suggestions:
            st.markdown("#### 💡 Suggested Questions:")
            cols = st.columns(2)
            for i, suggestion in enumerate(suggestions[:4]):
                with cols[i % 2]:
                    if st.button(suggestion, key=f"suggestion_{i}", use_container_width=True):
                        # Add suggestion to chat and get response
                        response = chatbot.send_message(suggestion, user_id)
                        st.rerun()
    
    # Chat input
    if user_input := st.chat_input("Ask me about your finances..."):
        # Display user message immediately
        with st.chat_message("user"):
            st.write(user_input)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = chatbot.send_message(user_input, user_id)
                st.write(response)
        
        # Rerun to update the chat history display
        st.rerun()
    
    # Chat statistics
    with st.expander("📊 Chat Session Info", expanded=False):
        st.write(f"Messages in this session: {len(st.session_state.chat_history)}")
        st.write(f"Context loaded: {'✅ Yes' if st.session_state.chat_context_loaded else '❌ No'}")
        
        if st.session_state.chat_context_loaded and 'user_context' in st.session_state:
            with st.expander("View Current User Context", expanded=False):
                st.code(st.session_state.user_context, language="text")