"""
Authentication module for the Avalanche Modeling Application.
Centralizes login logic for use across all pages.
"""

import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
from pathlib import Path
from typing import Optional, Tuple

# Path to config file
CONFIG_PATH = Path(__file__).parent.parent.parent / '.streamlit' / 'config.yaml'


def load_auth_config() -> dict:
    """Load authentication configuration."""
    with open(CONFIG_PATH) as file:
        return yaml.load(file, Loader=SafeLoader)


def get_authenticator() -> stauth.Authenticate:
    """Get or create the authenticator instance."""
    # Cache in session_state to avoid duplicate CookieManager component keys
    if 'authenticator' not in st.session_state:
        config = load_auth_config()
        st.session_state.authenticator = stauth.Authenticate(
            config['credentials'],
            config['cookie']['name'],
            config['cookie']['key'],
            config['cookie']['expiry_days']
        )
    return st.session_state.authenticator


def require_authentication() -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Ensure user is authenticated. Call at the start of each page.
    
    Returns:
        Tuple of (is_authenticated, username, name)
    """
    authenticator = get_authenticator()
    
    # Always call login with location parameter to check cookie
    try:
        authenticator.login(location='main')
    except Exception as e:
        st.error(f"Authentication error: {e}")
        return False, None, None
    
    # Check authentication status after login attempt
    if st.session_state.get('authentication_status') == False:
        st.error('Username/password is incorrect')
        return False, None, None
    elif st.session_state.get('authentication_status') is None:
        st.warning('Please enter your username and password')
        return False, None, None
    
    return True, st.session_state.get('username'), st.session_state.get('name')


def show_user_info_sidebar():
    """Display user info and logout button in sidebar."""
    if st.session_state.get('authentication_status'):
        authenticator = get_authenticator()
        with st.sidebar:
            st.write(f"👤 **{st.session_state.get('name')}**")
            authenticator.logout(location='sidebar')
            st.divider()
