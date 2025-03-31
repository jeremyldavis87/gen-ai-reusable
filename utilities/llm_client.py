# utilities/llm_client.py
import os
from typing import Dict, List, Optional, Union, Any
import httpx
import json
import asyncio
import logging
from tenacity import retry, stop_after_attempt, wait_exponential

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.embeddings import OpenAIEmbeddings

from utilities.llm_models import (
    LLMProvider,
    get_model_info,
    get_recommended_models,
    get_provider_for_model
)

logger = logging.getLogger(__name__)

class LLMClient:
    """Client for interacting with LLM providers using LangChain."""
    
    def __init__(
        self,
        provider: Optional[LLMProvider] = None,
        model: Optional[str] = None,
        task: str = "general",
        temperature: float = 0.7,
        max_tokens: int = 4096
    ):
        """Initialize LLM client.
        
        Args:
            provider: LLM provider to use (optional, will be determined from model if not provided)
            model: Model to use (optional, will use recommended model for task if not provided)
            task: Task type for model selection (general, code, vision, embeddings, long_context, cost_effective)
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens to generate
        """
        # Get recommended models for the task
        recommended = get_recommended_models(task)
        
        # Set model if not provided
        if not model:
            model = recommended["default"]
        
        # Get provider from model if not provided
        if not provider:
            provider = get_provider_for_model(model)
            if not provider:
                raise ValueError(f"Could not determine provider for model: {model}")
        
        # Get model info
        model_info = get_model_info(model)
        if not model_info:
            raise ValueError(f"Unknown model: {model}")
        
        # Validate provider matches model
        if model_info["provider"] != provider:
            raise ValueError(f"Model {model} is not from provider {provider}")
        
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model_info = model_info
        self._setup_client()
    
    def _setup_client(self):
        """Set up the LLM client based on the provider."""
        if self.provider == LLMProvider.OPENAI:
            if not os.getenv("OPENAI_API_KEY"):
                raise ValueError("OPENAI_API_KEY environment variable is required")
            self.llm = ChatOpenAI(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
        elif self.provider == LLMProvider.ANTHROPIC:
            if not os.getenv("ANTHROPIC_API_KEY"):
                raise ValueError("ANTHROPIC_API_KEY environment variable is required")
            self.llm = ChatAnthropic(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
        elif self.provider == LLMProvider.MISTRAL:
            # TODO: Add Mistral client setup when available
            raise NotImplementedError("Mistral provider not yet implemented")
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
            
        # Set up embeddings using OpenAI (for all providers)
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable is required for embeddings")
        self.embeddings = OpenAIEmbeddings()
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate text using the LLM.
        
        Args:
            prompt: Input prompt
            system_prompt: Optional system prompt
            temperature: Optional temperature override
            max_tokens: Optional max_tokens override
            
        Returns:
            Generated text
        """
        try:
            # Create messages
            messages = []
            if system_prompt:
                messages.append(SystemMessage(content=system_prompt))
            messages.append(HumanMessage(content=prompt))
            
            # Create chat prompt template
            chat_prompt = ChatPromptTemplate.from_messages(messages)
            
            # Create chain
            chain = chat_prompt | self.llm | StrOutputParser()
            
            # Generate response
            response = await chain.ainvoke({})
            return response
            
        except Exception as e:
            logger.error(f"Error in LLM generation: {str(e)}")
            raise
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """Chat with the LLM.
        
        Args:
            messages: List of messages in the conversation
            temperature: Optional temperature override
            max_tokens: Optional max_tokens override
            
        Returns:
            Generated response
        """
        try:
            # Convert messages to LangChain format
            langchain_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    langchain_messages.append(SystemMessage(content=msg["content"]))
                elif msg["role"] == "user":
                    langchain_messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    langchain_messages.append(AIMessage(content=msg["content"]))
            
            # Create chat prompt template
            chat_prompt = ChatPromptTemplate.from_messages(langchain_messages)
            
            # Create chain
            chain = chat_prompt | self.llm | StrOutputParser()
            
            # Generate response
            response = await chain.ainvoke({})
            return response
            
        except Exception as e:
            logger.error(f"Error in LLM chat: {str(e)}")
            raise
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def embed(
        self,
        text: Union[str, List[str]],
        model: Optional[str] = None
    ) -> List[List[float]]:
        """Get embeddings for text.
        
        Args:
            text: Text to embed
            model: Model to use (provider-specific)
            
        Returns:
            List of embeddings
        """
        try:
            # Convert text to list if string
            if isinstance(text, str):
                text = [text]
            
            # Generate embeddings
            embeddings = await self.embeddings.aembed_documents(text)
            return embeddings
            
        except Exception as e:
            logger.error(f"Error in embedding generation: {str(e)}")
            raise

    def _generate_mock_response(self, prompt: str, system_prompt: str = None) -> str:
        """Generate a mock response for testing without API keys."""
        # Extract the format from the system prompt if available
        format_type = "json"  # Default format
        if system_prompt and "JSON formatter" in system_prompt:
            format_type = "json"
        elif system_prompt and "YAML formatter" in system_prompt:
            format_type = "yaml"
        elif system_prompt and "CSV formatter" in system_prompt:
            format_type = "csv"
        elif system_prompt and "XML formatter" in system_prompt:
            format_type = "xml"
        elif system_prompt and "Markdown formatter" in system_prompt:
            format_type = "markdown"
        elif system_prompt and "HTML formatter" in system_prompt:
            format_type = "html"
        elif system_prompt and "SQL formatter" in system_prompt:
            format_type = "sql"
            
        # Generate mock responses based on format type
        if format_type == "json":
            return '{"name": "John Smith", "age": 35, "occupation": "Software Engineer", "skills": ["Python", "JavaScript", "Docker"]}'
        elif format_type == "yaml":
            return 'name: John Smith\nage: 35\noccupation: Software Engineer\nskills:\n  - Python\n  - JavaScript\n  - Docker'
        elif format_type == "csv":
            return 'name,age,occupation,skills\nJohn Smith,35,Software Engineer,"Python, JavaScript, Docker"'
        elif format_type == "xml":
            return '<person>\n  <n>John Smith</n>\n  <age>35</age>\n  <occupation>Software Engineer</occupation>\n  <skills>\n    <skill>Python</skill>\n    <skill>JavaScript</skill>\n    <skill>Docker</skill>\n  </skills>\n</person>'
        elif format_type == "markdown":
            return '# John Smith\n\n**Age:** 35\n\n**Occupation:** Software Engineer\n\n## Skills\n- Python\n- JavaScript\n- Docker'
        elif format_type == "html":
            return '<div class="person">\n  <h1>John Smith</h1>\n  <p><strong>Age:</strong> 35</p>\n  <p><strong>Occupation:</strong> Software Engineer</p>\n  <h2>Skills</h2>\n  <ul>\n    <li>Python</li>\n    <li>JavaScript</li>\n    <li>Docker</li>\n  </ul>\n</div>'
        elif format_type == "sql":
            return 'CREATE TABLE person (\n  name VARCHAR(100),\n  age INT,\n  occupation VARCHAR(100)\n);\n\nINSERT INTO person (name, age, occupation) VALUES ("John Smith", 35, "Software Engineer");\n\nCREATE TABLE skills (\n  person_name VARCHAR(100),\n  skill VARCHAR(100)\n);\n\nINSERT INTO skills (person_name, skill) VALUES ("John Smith", "Python");\nINSERT INTO skills (person_name, skill) VALUES ("John Smith", "JavaScript");\nINSERT INTO skills (person_name, skill) VALUES ("John Smith", "Docker");'
        else:
            return "Mock response for testing purposes."