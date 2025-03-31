#!/usr/bin/env python3
import os
import jwt
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from pathlib import Path

def generate_dev_token():
    # Load environment variables from config/.env
    env_path = Path(__file__).parent.parent / 'config' / '.env'
    load_dotenv(env_path)
    
    # Get JWT settings from environment
    secret_key = os.getenv("SECRET_KEY")
    algorithm = os.getenv("ALGORITHM", "HS256")
    access_token_expire_minutes = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))
    
    # Get development user credentials
    dev_email = os.getenv("DEV_USER_EMAIL")
    dev_role = os.getenv("DEV_USER_ROLE")
    
    if not all([secret_key, dev_email, dev_role]):
        print("Error: Missing required environment variables. Please check your config/.env file.")
        return
    
    # Create token data
    token_data = {
        "sub": dev_email,
        "role": dev_role,
        "exp": datetime.now(timezone.utc) + timedelta(minutes=access_token_expire_minutes)
    }
    
    # Generate token
    token = jwt.encode(token_data, secret_key, algorithm=algorithm)
    
    print("\nDevelopment Token Generated:")
    print("-" * 50)
    print(token)
    print("-" * 50)
    print("\nTo use this token in your requests, add it to the Authorization header:")
    print(f"Authorization: Bearer {token}")
    print("\nExample curl command:")
    print(f'curl -H "Authorization: Bearer {token}" http://localhost:8000/your-endpoint')
    print("\nToken will expire in", access_token_expire_minutes, "minutes")

if __name__ == "__main__":
    generate_dev_token() 