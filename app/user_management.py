#!/usr/bin/env python3
"""
User Management Utility for Avalanche App

This script allows admin to:
1. Add new users
2. List all users
3. Delete users
4. Change passwords

Usage: 
    source venv/bin/activate  # Activate virtualenv first!
    python app/user_management.py list
    python app/user_management.py add username "Name" "email" "password"
    python app/user_management.py delete username
    python app/user_management.py change-password username "new_password"
"""

import sys
import os
import yaml
from yaml.loader import SafeLoader

# Suppress Streamlit warnings when running in CLI mode
os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
import warnings
warnings.filterwarnings('ignore')

try:
    import streamlit_authenticator as stauth
except ImportError:
    print("ERROR: streamlit_authenticator not found!")
    print("")
    print("Please activate the virtualenv first:")
    print("  cd /home/gustav/avalanche-app")
    print("  source venv/bin/activate")
    print("  python app/user_management.py list")
    sys.exit(1)

from pathlib import Path

config_file = Path(__file__).parent.parent / '.streamlit' / 'config.yaml'

def load_config():
    """Load configuration from TOML file."""
    with open(config_file) as file:
        return yaml.load(file, Loader=SafeLoader)

def save_config(config):
    """Save configuration to TOML file."""
    with open(config_file, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)

def list_users():
    """List all users."""
    config = load_config()
    print("\n=== Current Users ===")
    for username, user_data in config['credentials']['usernames'].items():
        print(f"  - {username}: {user_data['name']} ({user_data['email']})")
    print()

def add_user(username, name, email, password):
    """Add a new user."""
    config = load_config()
    
    if username in config['credentials']['usernames']:
        print(f"Error: User '{username}' already exists!")
        return False
    
    # Hash the password
    hashed_password = stauth.Hasher.hash(password)
    
    config['credentials']['usernames'][username] = {
        'name': name,
        'email': email,
        'password': hashed_password
    }
    
    save_config(config)
    print(f"✓ User '{username}' added successfully!")
    return True

def delete_user(username):
    """Delete a user."""
    config = load_config()
    
    if username == 'admin':
        print("Error: Cannot delete admin user!")
        return False
    
    if username not in config['credentials']['usernames']:
        print(f"Error: User '{username}' does not exist!")
        return False
    
    del config['credentials']['usernames'][username]
    save_config(config)
    print(f"✓ User '{username}' deleted successfully!")
    return True

def change_password(username, new_password):
    """Change user password."""
    config = load_config()
    
    if username not in config['credentials']['usernames']:
        print(f"Error: User '{username}' does not exist!")
        return False
    
    # Hash the new password
    hashed_password = stauth.Hasher.hash(new_password)
    config['credentials']['usernames'][username]['password'] = hashed_password
    
    save_config(config)
    print(f"✓ Password changed for user '{username}'!")
    return True

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("""
Usage:
  python user_management.py list
  python user_management.py add <username> <name> <email> <password>
  python user_management.py delete <username>
  python user_management.py change-password <username> <new_password>

Examples:
  python user_management.py list
  python user_management.py add john "John Doe" "john@example.com" "mypassword"
  python user_management.py delete john
  python user_management.py change-password admin "new_admin_pass"
        """)
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == 'list':
        list_users()
    elif command == 'add':
        if len(sys.argv) != 6:
            print("Error: add requires <username> <name> <email> <password>")
            sys.exit(1)
        add_user(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    elif command == 'delete':
        if len(sys.argv) != 3:
            print("Error: delete requires <username>")
            sys.exit(1)
        delete_user(sys.argv[2])
    elif command == 'change-password':
        if len(sys.argv) != 4:
            print("Error: change-password requires <username> <new_password>")
            sys.exit(1)
        change_password(sys.argv[2], sys.argv[3])
    else:
        print(f"Error: Unknown command '{command}'")
        sys.exit(1)
