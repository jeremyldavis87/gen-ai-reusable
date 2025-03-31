# utilities/llm_models.py
from typing import Dict, List, Optional
from enum import Enum

class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    MISTRAL = "mistral"
    META = "meta"
    GOOGLE = "google"
    AZURE = "azure"
    BEDROCK = "bedrock"

# OpenAI Models (Direct API)
OPENAI_MODELS = {
    # GPT-4 Family
    "gpt-4o": {
        "description": "GPT-4o - Latest version with vision capabilities",
        "context_window": 128000,
        "is_multimodal": True,
        "provider": LLMProvider.OPENAI,
    },
    "gpt-4o-mini": {
        "description": "GPT-4o mini - Smaller, faster version of GPT-4o",
        "context_window": 128000,
        "is_multimodal": True,
        "provider": LLMProvider.OPENAI,
    },
    "gpt-4-turbo": {
        "description": "GPT-4 Turbo - Cost-effective GPT-4 variant",
        "context_window": 128000,
        "is_multimodal": True,
        "provider": LLMProvider.OPENAI,
    },
    "gpt-4": {
        "description": "GPT-4 - Original GPT-4 model",
        "context_window": 8192,
        "is_multimodal": False,
        "provider": LLMProvider.OPENAI,
    },
    "gpt-4-vision-preview": {
        "description": "GPT-4 Vision - Image understanding capabilities",
        "context_window": 128000,
        "is_multimodal": True,
        "provider": LLMProvider.OPENAI,
    },
    # GPT-3.5 Family
    "gpt-3.5-turbo": {
        "description": "GPT-3.5 Turbo - Cost-effective model for most tasks",
        "context_window": 16385,
        "is_multimodal": False,
        "provider": LLMProvider.OPENAI,
    },
    # Embeddings models
    "text-embedding-3-large": {
        "description": "Large text embedding model",
        "is_embedding": True,
        "provider": LLMProvider.OPENAI,
    },
    "text-embedding-3-small": {
        "description": "Small text embedding model",
        "is_embedding": True,
        "provider": LLMProvider.OPENAI,
    },
}

# Anthropic Models (Direct API)
ANTHROPIC_MODELS = {
    "claude-3-7-sonnet-20250219": {
        "description": "Claude 3.7 Sonnet - Latest high-performance model",
        "context_window": 200000,
        "is_multimodal": True,
        "provider": LLMProvider.ANTHROPIC,
    },
    "claude-3-5-sonnet-20240620": {
        "description": "Claude 3.5 Sonnet - Updated in June 2024",
        "context_window": 200000,
        "is_multimodal": True,
        "provider": LLMProvider.ANTHROPIC,
    },
    "claude-3-5-sonnet-20241030": {
        "description": "Claude 3.5 Sonnet - Updated in October 2024",
        "context_window": 200000,
        "is_multimodal": True,
        "provider": LLMProvider.ANTHROPIC,
    },
    "claude-3-5-haiku-20240307": {
        "description": "Claude 3.5 Haiku - Fast, cost-effective model",
        "context_window": 200000,
        "is_multimodal": True,
        "provider": LLMProvider.ANTHROPIC,
    },
    "claude-3-opus-20240229": {
        "description": "Claude 3 Opus - High-capability model",
        "context_window": 200000,
        "is_multimodal": True,
        "provider": LLMProvider.ANTHROPIC,
    },
    "claude-3-sonnet-20240229": {
        "description": "Claude 3 Sonnet - Balanced performance",
        "context_window": 200000,
        "is_multimodal": True,
        "provider": LLMProvider.ANTHROPIC,
    },
    "claude-3-haiku-20240307": {
        "description": "Claude 3 Haiku - Fast, cost-effective model",
        "context_window": 200000,
        "is_multimodal": True,
        "provider": LLMProvider.ANTHROPIC,
    },
}

# Mistral Models (Direct API)
MISTRAL_MODELS = {
    "mistral-large-2-2411": {
        "description": "Mistral Large 2 (November 2024)",
        "context_window": 32768,
        "is_multimodal": True,
        "provider": LLMProvider.MISTRAL,
    },
    "mistral-large-2-2407": {
        "description": "Mistral Large 2 (July 2024)",
        "context_window": 32768,
        "is_multimodal": True,
        "provider": LLMProvider.MISTRAL,
    },
    "mistral-large-2402": {
        "description": "Mistral Large (February 2024)",
        "context_window": 32768,
        "is_multimodal": False,
        "provider": LLMProvider.MISTRAL,
    },
    "mistral-medium": {
        "description": "Mistral Medium - Balanced performance",
        "context_window": 32768,
        "is_multimodal": False,
        "provider": LLMProvider.MISTRAL,
    },
    "mistral-small-3.1": {
        "description": "Mistral Small 3.1 - Latest small model",
        "context_window": 32768,
        "is_multimodal": False,
        "provider": LLMProvider.MISTRAL,
    },
    "mistral-embed": {
        "description": "Mistral Embedding model",
        "is_embedding": True,
        "provider": LLMProvider.MISTRAL,
    },
}

# Default recommended models by task
RECOMMENDED_DEFAULTS = {
    "general": {
        "default": "claude-3-5-sonnet-20240620",
        "fallbacks": ["gpt-4o", "claude-3-7-sonnet-20250219", "mistral-large-2-2411"],
    },
    "code": {
        "default": "gpt-4o",
        "fallbacks": ["claude-3-7-sonnet-20250219", "mistral-large-2-2411"],
    },
    "vision": {
        "default": "gpt-4o",
        "fallbacks": ["claude-3-7-sonnet-20250219", "mistral-large-2-2411"],
    },
    "embeddings": {
        "default": "text-embedding-3-large",
        "fallbacks": ["mistral-embed"],
    },
    "long_context": {
        "default": "claude-3-7-sonnet-20250219",
        "fallbacks": ["gpt-4o", "mistral-large-2-2411"],
    },
    "cost_effective": {
        "default": "mistral-small-3.1",
        "fallbacks": ["gpt-3.5-turbo", "claude-3-5-haiku-20240307"],
    },
}

def get_model_info(model_id: str) -> Optional[Dict]:
    """
    Get information about a specific model by its ID
    
    Args:
        model_id (str): The model identifier
        
    Returns:
        dict: Model information or None if not found
    """
    # Check in all model collections
    all_models = {
        **OPENAI_MODELS,
        **ANTHROPIC_MODELS,
        **MISTRAL_MODELS,
    }
    
    if model_id in all_models:
        return all_models[model_id]
    
    return None

def get_recommended_models(task: str = "general") -> Dict[str, List[str]]:
    """
    Get recommended models for a specific task
    
    Args:
        task (str): The task type (general, code, vision, embeddings, long_context, cost_effective)
        
    Returns:
        dict: Default and fallback models for the task
    """
    if task in RECOMMENDED_DEFAULTS:
        return RECOMMENDED_DEFAULTS[task]
    return RECOMMENDED_DEFAULTS["general"]

def get_provider_for_model(model_id: str) -> Optional[LLMProvider]:
    """
    Get the provider for a specific model ID
    
    Args:
        model_id (str): The model identifier
        
    Returns:
        LLMProvider: The provider enum or None if not found
    """
    model_info = get_model_info(model_id)
    if model_info:
        return model_info["provider"]
    return None 