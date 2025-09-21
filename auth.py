"""
Authentication System with User Management and Data Persistence
Handles user registration, login, and session management with MongoDB
"""

import streamlit as st
import hashlib
import re
import uuid
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from pymongo import MongoClient
from pymongo.collection import Collection
import os
from dotenv import load_dotenv

load_dotenv()

class AuthenticationManager:
    """Handle user authentication and session management"""
    
    def __init__(self, mongodb_url: str):
        self.client = MongoClient(mongodb_url)
        self.db = self.client.financial_advisor
        self.users: Collection = self.db.users
        self.user_sessions: Collection = self.db.user_sessions
        self.user_data: Collection = self.db.user_financial_data
        
        # Create indexes
        self.users.create_index("username", unique=True)
        self.users.create_index("email", unique=True)
        self.user_sessions.create_index("session_token")
        self.user_data.create_index([("user_id", 1), ("data_date", -1)])
    
    def hash_password(self, password: str) -> str:
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def validate_email(self, email: str) -> bool:
        """Validate email format"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    def validate_password(self, password: str) -> tuple[bool, str]:
        """Validate password strength"""
        if len(password) < 8:
            return False, "Password must be at least 8 characters long"
        if not re.search(r'[A-Z]', password):
            return False, "Password must contain at least one uppercase letter"
        if not re.search(r'[a-z]', password):
            return False, "Password must contain at least one lowercase letter"
        if not re.search(r'\d', password):
            return False, "Password must contain at least one number"
        return True, "Password is strong"
    
    def register_user(self, username: str, email: str, password: str, full_name: str) -> tuple[bool, str]:
        """Register a new user"""
        try:
            # Validate inputs
            if not username or len(username) < 3:
                return False, "Username must be at least 3 characters long"
            
            if not self.validate_email(email):
                return False, "Please enter a valid email address"
            
            is_valid, msg = self.validate_password(password)
            if not is_valid:
                return False, msg
            
            # Check if user already exists
            if self.users.find_one({"username": username}):
                return False, "Username already exists"
            
            if self.users.find_one({"email": email}):
                return False, "Email already registered"
            
            # Create user
            user_doc = {
                "user_id": str(uuid.uuid4()),
                "username": username,
                "email": email,
                "password_hash": self.hash_password(password),
                "full_name": full_name,
                "created_at": datetime.utcnow(),
                "last_login": None,
                "is_active": True
            }
            
            self.users.insert_one(user_doc)
            return True, "Registration successful!"
            
        except Exception as e:
            return False, f"Registration failed: {str(e)}"
    
    def authenticate_user(self, username: str, password: str) -> tuple[bool, Optional[Dict]]:
        """Authenticate user login"""
        try:
            user = self.users.find_one({"username": username})
            
            if not user:
                return False, None
            
            if not user.get("is_active", True):
                return False, None
            
            password_hash = self.hash_password(password)
            
            if user["password_hash"] == password_hash:
                # Update last login
                self.users.update_one(
                    {"username": username},
                    {"$set": {"last_login": datetime.utcnow()}}
                )
                return True, user
            
            return False, None
            
        except Exception as e:
            return False, None
    
    def create_session(self, user_id: str) -> str:
        """Create a new user session"""
        session_token = str(uuid.uuid4())
        
        session_doc = {
            "user_id": user_id,
            "session_token": session_token,
            "created_at": datetime.utcnow(),
            "expires_at": datetime.utcnow() + timedelta(days=30),
            "is_active": True
        }
        
        self.user_sessions.insert_one(session_doc)
        return session_token
    
    def validate_session(self, session_token: str) -> Optional[Dict]:
        """Validate an existing session"""
        try:
            session = self.user_sessions.find_one({
                "session_token": session_token,
                "is_active": True,
                "expires_at": {"$gt": datetime.utcnow()}
            })
            
            if session:
                user = self.users.find_one({"user_id": session["user_id"]})
                return user
            
            return None
            
        except Exception as e:
            return None
    
    def logout_user(self, session_token: str):
        """Logout user by invalidating session"""
        self.user_sessions.update_one(
            {"session_token": session_token},
            {"$set": {"is_active": False}}
        )
    
    def save_user_financial_data(self, user_id: str, financial_data: Dict, data_date: datetime = None) -> str:
        """Save user's financial data with date"""
        if data_date is None:
            data_date = datetime.utcnow()
        
        data_doc = {
            "user_id": user_id,
            "financial_data": financial_data,
            "data_date": data_date,
            "created_at": datetime.utcnow()
        }
        
        result = self.user_data.insert_one(data_doc)
        return str(result.inserted_id)
    
    def get_user_financial_history(self, user_id: str, limit: int = 10) -> List[Dict]:
        """Get user's financial data history"""
        return list(self.user_data.find(
            {"user_id": user_id}
        ).sort("data_date", -1).limit(limit))
    
    def get_user_latest_data(self, user_id: str) -> Optional[Dict]:
        """Get user's most recent financial data"""
        latest = self.user_data.find_one(
            {"user_id": user_id},
            sort=[("data_date", -1)]
        )
        return latest

def show_auth_interface(auth_manager: AuthenticationManager) -> Optional[Dict]:
    """Show authentication interface"""
    
    # Check if user is already logged in
    if 'session_token' in st.session_state and st.session_state.session_token:
        user = auth_manager.validate_session(st.session_state.session_token)
        if user:
            return user
        else:
            # Invalid session, clear it
            if 'session_token' in st.session_state:
                del st.session_state.session_token
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 40px; border-radius: 15px; color: white; text-align: center; margin: 20px 0;">
        <h1>Welcome to Finmate an AI Financial Advisor</h1>
        <h3>Multi-Agent Financial Analysis System</h3>
        <p>Please sign in or create an account to continue</p>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["🔑 Sign In", "📝 Sign Up"])
    
    with tab1:
        st.subheader("Sign In to Your Account")
        
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit_login = st.form_submit_button("Sign In", use_container_width=True)
            
            if submit_login:
                if username and password:
                    success, user = auth_manager.authenticate_user(username, password)
                    
                    if success:
                        session_token = auth_manager.create_session(user["user_id"])
                        st.session_state.session_token = session_token
                        st.session_state.current_user = user
                        st.success(f"Welcome back, {user['full_name']}!")
                        st.rerun()
                    else:
                        st.error("Invalid username or password")
                else:
                    st.error("Please enter both username and password")
    
    with tab2:
        st.subheader("Create New Account")
        
        with st.form("register_form"):
            reg_username = st.text_input("Username", help="Minimum 3 characters")
            reg_email = st.text_input("Email Address")
            reg_full_name = st.text_input("Full Name")
            reg_password = st.text_input("Password", type="password", 
                                       help="Minimum 8 characters with uppercase, lowercase, and number")
            reg_password_confirm = st.text_input("Confirm Password", type="password")
            
            submit_register = st.form_submit_button("Create Account", use_container_width=True)
            
            if submit_register:
                if not all([reg_username, reg_email, reg_full_name, reg_password, reg_password_confirm]):
                    st.error("Please fill in all fields")
                elif reg_password != reg_password_confirm:
                    st.error("Passwords do not match")
                else:
                    success, message = auth_manager.register_user(
                        reg_username, reg_email, reg_password, reg_full_name
                    )
                    
                    if success:
                        st.success(message)
                        st.info("Please use the Sign In tab to log in with your new account")
                    else:
                        st.error(message)
    
    return None