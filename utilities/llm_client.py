# utilities/llm_client.py
import os
from typing import Dict, List, Optional, Union, Any
import httpx
import json
import asyncio
from enum import Enum

class LLMProvider(str, Enum):
    GPT4O = "gpt4o"
    CLAUDE = "claude-sonnet-3.5"

class LLMClient:
    """Generic LLM client that can work with different providers."""
    
    def __init__(self, provider: LLMProvider = LLMProvider.CLAUDE):
        self.provider = provider
        self.api_keys = {
            LLMProvider.GPT4O: os.getenv("OPENAI_API_KEY"),
            LLMProvider.CLAUDE: os.getenv("ANTHROPIC_API_KEY")
        }
        self.endpoints = {
            LLMProvider.GPT4O: "https://api.openai.com/v1/chat/completions",
            LLMProvider.CLAUDE: "https://api.anthropic.com/v1/messages"
        }
        
    async def generate(self, 
                      prompt: str, 
                      system_prompt: Optional[str] = None,
                      temperature: float = 0.7,
                      max_tokens: int = 1000,
                      tools: Optional[List[Dict[str, Any]]] = None) -> str:
        """Generate text from LLM."""
        if self.provider == LLMProvider.GPT4O:
            return await self._generate_openai(prompt, system_prompt, temperature, max_tokens, tools)
        elif self.provider == LLMProvider.CLAUDE:
            return await self._generate_anthropic(prompt, system_prompt, temperature, max_tokens)
    
    async def _generate_openai(self, 
                              prompt: str, 
                              system_prompt: Optional[str] = None,
                              temperature: float = 0.7,
                              max_tokens: int = 1000,
                              tools: Optional[List[Dict[str, Any]]] = None) -> str:
        """Generate text from OpenAI's API."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": "gpt-4o",
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        if tools:
            payload["tools"] = tools
            
        async with httpx.AsyncClient() as client:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_keys[LLMProvider.GPT4O]}"
            }
            response = await client.post(
                self.endpoints[LLMProvider.GPT4O],
                headers=headers,
                json=payload,
                timeout=60.0
            )
            response_data = response.json()
            return response_data["choices"][0]["message"]["content"]
    
    async def _generate_anthropic(self, 
                                 prompt: str, 
                                 system_prompt: Optional[str] = None,
                                 temperature: float = 0.7,
                                 max_tokens: int = 1000) -> str:
        """Generate text from Anthropic's API."""
        payload = {
            "model": "claude-3-sonnet-20240229",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        if system_prompt:
            payload["system"] = system_prompt
            
        async with httpx.AsyncClient() as client:
            headers = {
                "Content-Type": "application/json",
                "x-api-key": self.api_keys[LLMProvider.CLAUDE],
                "anthropic-version": "2023-06-01"
            }
            response = await client.post(
                self.endpoints[LLMProvider.CLAUDE],
                headers=headers,
                json=payload,
                timeout=60.0
            )
            response_data = response.json()
            return response_data["content"][0]["text"]


# utilities/logging_utils.py
import logging
import json
from datetime import datetime
import uuid
from typing import Dict, Any, Optional

class StructuredLogger:
    """Structured logging utility for consistent logging across services."""
    
    def __init__(self, service_name: str, log_level: int = logging.INFO):
        self.logger = logging.getLogger(service_name)
        self.logger.setLevel(log_level)
        
        # Add console handler if not already present
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            
        self.service_name = service_name
        
    def log(self, 
            level: int, 
            message: str, 
            request_id: Optional[str] = None, 
            extra_data: Optional[Dict[str, Any]] = None) -> None:
        """Log a structured message."""
        if not request_id:
            request_id = str(uuid.uuid4())
            
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "service": self.service_name,
            "request_id": request_id,
            "message": message
        }
        
        if extra_data:
            log_data.update(extra_data)
            
        self.logger.log(level, json.dumps(log_data))
        
    def info(self, message: str, request_id: Optional[str] = None, extra_data: Optional[Dict[str, Any]] = None) -> None:
        """Log an info message."""
        self.log(logging.INFO, message, request_id, extra_data)
        
    def error(self, message: str, request_id: Optional[str] = None, extra_data: Optional[Dict[str, Any]] = None) -> None:
        """Log an error message."""
        self.log(logging.ERROR, message, request_id, extra_data)
        
    def warning(self, message: str, request_id: Optional[str] = None, extra_data: Optional[Dict[str, Any]] = None) -> None:
        """Log a warning message."""
        self.log(logging.WARNING, message, request_id, extra_data)
        
    def debug(self, message: str, request_id: Optional[str] = None, extra_data: Optional[Dict[str, Any]] = None) -> None:
        """Log a debug message."""
        self.log(logging.DEBUG, message, request_id, extra_data)


# utilities/database.py
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./app.db")

# Create engine
engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base model class
Base = declarative_base()

# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# utilities/auth.py
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from pydantic import BaseModel
import os

# Secret key for JWT
SECRET_KEY = os.getenv("SECRET_KEY", "")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Password context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hashed password."""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Hash password."""
    return pwd_context.hash(password)

def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)) -> Dict[str, Any]:
    """Get current user from JWT token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    
    # In a real app, you would lookup the user in a database
    # For this example, we'll just return the username
    return {"username": token_data.username}


# utilities/prompts.py
class PromptTemplates:
    """Collection of prompt templates for different LLM tasks."""
    
    @staticmethod
    def document_extraction(document_text: str, items_to_extract: list, output_format: str) -> str:
        """Generate prompt for document extraction."""
        return f"""
Extract the following items from the document:
{', '.join(items_to_extract)}

Output the extracted information in this format: {output_format}

Document content:
{document_text}
"""

    @staticmethod
    def categorization(content: str, categories: list, instructions: str) -> str:
        """Generate prompt for content categorization."""
        categories_str = '\n'.join([f"- {category}" for category in categories])
        return f"""
Categorize the following content according to these categories:
{categories_str}

Additional instructions:
{instructions}

Content to categorize:
{content}
"""

    @staticmethod
    def code_generation(specifications: str) -> str:
        """Generate prompt for code generation."""
        return f"""
Generate code based on the following specifications:
{specifications}

Write clean, well-documented, and efficient code.
"""

    @staticmethod
    def code_documentation(code: str) -> str:
        """Generate prompt for code documentation."""
        return f"""
Write comprehensive documentation for the following code:
{code}

Include:
- Purpose of the code
- Function/class descriptions
- Parameter explanations
- Return value information
- Usage examples
"""

    @staticmethod
    def data_cleaning(data_sample: str, cleaning_instructions: str) -> str:
        """Generate prompt for data cleaning instructions."""
        return f"""
Generate data cleaning instructions for the following data:
{data_sample}

Specific cleaning requirements:
{cleaning_instructions}

Provide a clear step-by-step process to clean this data.
"""

    @staticmethod
    def summarization(content: str, summary_type: str, length: str) -> str:
        """Generate prompt for content summarization."""
        return f"""
Summarize the following content:
{content}

Type of summary: {summary_type}
Length: {length}
"""

    @staticmethod
    def conversation_generation(context: str, query: str) -> str:
        """Generate prompt for conversational response generation."""
        return f"""
Generate a conversational response to the following query:

Context information:
{context}

User query:
{query}

Your response should be natural, helpful, and address the user's query directly.
"""

    @staticmethod
    def semantic_search(query: str, documents: list) -> str:
        """Generate prompt for semantic search."""
        docs_str = '\n\n---\n\n'.join([f"Document {i+1}:\n{doc}" for i, doc in enumerate(documents)])
        return f"""
Find the most relevant information for the following query:
{query}

Documents to search:
{docs_str}

Return only the most relevant information that answers the query.
"""

    @staticmethod
    def workflow_automation(context: str, task_description: str) -> str:
        """Generate prompt for workflow automation."""
        return f"""
Based on the following context and task description, determine the next steps in the workflow:

Context:
{context}

Task description:
{task_description}

Provide a detailed plan for automating this workflow.
"""

    @staticmethod
    def personalization(user_data: str, content_to_personalize: str) -> str:
        """Generate prompt for content personalization."""
        return f"""
Personalize the following content for the user:

User data:
{user_data}

Content to personalize:
{content_to_personalize}

The personalized content should reflect the user's preferences and needs.
"""

    @staticmethod
    def code_quality(code: str, quality_criteria: str) -> str:
        """Generate prompt for code quality analysis."""
        return f"""
Analyze the quality of the following code based on these criteria:
{quality_criteria}

Code to analyze:
{code}

Provide specific issues found and suggestions for improvement.
"""