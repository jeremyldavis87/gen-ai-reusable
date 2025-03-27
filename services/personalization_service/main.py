# services/personalization_service/main.py
from fastapi import FastAPI, Depends, HTTPException, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
import uuid
import json
import asyncio
from enum import Enum
from datetime import datetime

from utilities.llm_client import LLMClient, LLMProvider
from utilities.logging_utils import StructuredLogger
from utilities.auth import get_current_user
from utilities.prompts import PromptTemplates

# Create app
app = FastAPI(
    title="Content Personalization Service",
    description="API for personalizing content based on user preferences and behavior"
)

# Setup logger
logger = StructuredLogger("personalization_service")

# Models
class ContentType(str, Enum):
    TEXT = "text"
    MARKETING = "marketing"
    EMAIL = "email"
    PRODUCT_DESCRIPTION = "product_description"
    NEWS = "news"
    EDUCATIONAL = "educational"
    DOCUMENTATION = "documentation"
    
class TargetAudience(str, Enum):
    GENERAL = "general"
    TECHNICAL = "technical"
    EXECUTIVE = "executive"
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    CUSTOMER = "customer"
    PROSPECT = "prospect"
    
class PersonalizationProfile(BaseModel):
    user_id: Optional[str] = Field(None, description="Unique identifier for the user")
    demographics: Optional[Dict[str, Any]] = Field(None, description="Demographic information")
    preferences: Optional[Dict[str, Any]] = Field(None, description="User preferences")
    behavior: Optional[Dict[str, Any]] = Field(None, description="User behavior data")
    interests: Optional[List[str]] = Field(None, description="User interests")
    history: Optional[List[Dict[str, Any]]] = Field(None, description="User interaction history")
    
class PersonalizationRequest(BaseModel):
    content: str = Field(..., description="Content to personalize")
    content_type: ContentType = Field(..., description="Type of content")
    user_profile: PersonalizationProfile = Field(..., description="User profile for personalization")
    target_audience: Optional[TargetAudience] = Field(None, description="Target audience for the content")
    personalization_aspects: Optional[List[str]] = Field(None, description="Specific aspects to personalize")
    constraints: Optional[Dict[str, Any]] = Field(None, description="Constraints for personalization")
    
class PersonalizationResponse(BaseModel):
    request_id: str = Field(..., description="Unique identifier for the request")
    personalized_content: str = Field(..., description="Personalized content")
    personalization_summary: Optional[str] = Field(None, description="Summary of personalizations applied")
    variations: Optional[List[Dict[str, Any]]] = Field(None, description="Alternative variations")

class ContentVariation(BaseModel):
    original_content: str = Field(..., description="Original content")
    variations: List[Dict[str, Any]] = Field(..., description="Generated variations")

class ABTestRequest(BaseModel):
    content: str = Field(..., description="Original content to generate variations for")
    content_type: ContentType = Field(..., description="Type of content")
    test_variables: List[Dict[str, Any]] = Field(..., description="Variables to test")
    audience_segments: Optional[List[Dict[str, Any]]] = Field(None, description="Audience segments to target")
    constraints: Optional[Dict[str, Any]] = Field(None, description="Constraints for variation generation")
    
class LocalizationRequest(BaseModel):
    content: str = Field(..., description="Content to localize")
    content_type: ContentType = Field(..., description="Type of content")
    source_locale: str = Field(..., description="Source locale (e.g., 'en-US')")
    target_locale: str = Field(..., description="Target locale (e.g., 'fr-FR')")
    cultural_notes: Optional[Dict[str, Any]] = Field(None, description="Cultural considerations")
    preserve_elements: Optional[List[str]] = Field(None, description="Elements to preserve untranslated")

# Routes
@app.post("/personalize", response_model=PersonalizationResponse)
async def personalize_content(
    request: PersonalizationRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Personalize content based on user profile and preferences."""
    request_id = str(uuid.uuid4())
    
    try:
        logger.info(
            "Content personalization request received", 
            request_id, 
            {"content_type": request.content_type}
        )
        
        # Create LLM client
        llm_client = LLMClient()
        
        # Format user profile for the prompt
        user_profile_str = "User Profile:\n"
        
        if request.user_profile.demographics:
            user_profile_str += "Demographics:\n"
            for key, value in request.user_profile.demographics.items():
                user_profile_str += f"- {key}: {value}\n"
        
        if request.user_profile.preferences:
            user_profile_str += "Preferences:\n"
            for key, value in request.user_profile.preferences.items():
                user_profile_str += f"- {key}: {value}\n"
        
        if request.user_profile.interests:
            user_profile_str += f"Interests: {', '.join(request.user_profile.interests)}\n"
        
        if request.user_profile.behavior:
            user_profile_str += "Behavior:\n"
            for key, value in request.user_profile.behavior.items():
                user_profile_str += f"- {key}: {value}\n"
        
        # Build the system prompt
        system_prompt = f"""
        You are an expert content personalization system. Personalize the provided {request.content_type.value} content based on the user profile.
        Make the content more relevant, engaging, and effective for this specific user.
        {f"The target audience is {request.target_audience.value}." if request.target_audience else ""}
        
        Adapt tone, examples, terminology, and structure to match the user's preferences and characteristics.
        Preserve the core message and intent of the original content.
        """
        
        # Build the user prompt using the personalization template
        prompt = PromptTemplates.personalization(user_profile_str, request.content)
        
        # Add personalization aspects if provided
        if request.personalization_aspects:
            aspects_str = "Focus on personalizing these specific aspects:\n"
            for aspect in request.personalization_aspects:
                aspects_str += f"- {aspect}\n"
            prompt += f"\n{aspects_str}"
        
        # Add constraints if provided
        if request.constraints:
            constraints_str = "Consider these constraints:\n"
            for key, value in request.constraints.items():
                constraints_str += f"- {key}: {value}\n"
            prompt += f"\n{constraints_str}"
        
        # Append format instructions
        prompt += """
        
        Format your response as a valid JSON object with this structure:
        {
            "personalized_content": "The personalized content",
            "personalization_summary": "Summary of personalizations applied",
            "variations": [
                {
                    "title": "Variation title/focus",
                    "content": "Alternative version of the content"
                },
                ...
            ]
        }
        """
        
        # Generate the personalized content
        result = await llm_client.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.7,  # Higher temperature for more creative personalization
            max_tokens=2000
        )
        
        # Parse the result
        try:
            # Find JSON in the response
            json_start = result.find("{")
            json_end = result.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = result[json_start:json_end]
                personalization_data = json.loads(json_str)
            else:
                # If no JSON found, try to parse the entire response
                personalization_data = json.loads(result)
            
            return PersonalizationResponse(
                request_id=request_id,
                personalized_content=personalization_data.get("personalized_content", ""),
                personalization_summary=personalization_data.get("personalization_summary"),
                variations=personalization_data.get("variations")
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing personalization result: {str(e)}", request_id)
            
            # If parsing fails, assume the result is the personalized content directly
            return PersonalizationResponse(
                request_id=request_id,
                personalized_content=result,
                personalization_summary="Personalization applied directly"
            )
            
    except Exception as e:
        logger.error(f"Error personalizing content: {str(e)}", request_id)
        raise HTTPException(status_code=500, detail=f"Error personalizing content: {str(e)}")

@app.post("/generate-ab-test", response_model=ContentVariation)
async def generate_ab_test_variations(
    request: ABTestRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Generate variations of content for A/B testing."""
    request_id = str(uuid.uuid4())
    
    try:
        logger.info("A/B test generation request received", request_id)
        
        # Create LLM client
        llm_client = LLMClient()
        
        # Format test variables for the prompt
        variables_str = "Test Variables:\n"
        for i, variable in enumerate(request.test_variables):
            variables_str += f"Variable {i+1}: {variable['name']}\n"
            if 'options' in variable:
                variables_str += f"Options: {', '.join(variable['options'])}\n"
            if 'description' in variable:
                variables_str += f"Description: {variable['description']}\n"
            variables_str += "\n"
        
        # Format audience segments if provided
        segments_str = ""
        if request.audience_segments:
            segments_str = "Audience Segments:\n"
            for i, segment in enumerate(request.audience_segments):
                segments_str += f"Segment {i+1}: {segment['name']}\n"
                if 'description' in segment:
                    segments_str += f"Description: {segment['description']}\n"
                if 'characteristics' in segment:
                    segments_str += f"Characteristics: {', '.join(segment['characteristics'])}\n"
                segments_str += "\n"
        
        # Build the system prompt
        system_prompt = f"""
        You are an expert in A/B testing content optimization. Generate diverse, testable variations of the provided {request.content_type.value} content.
        Each variation should test specific variables while maintaining the core message and purpose.
        Create variations that are meaningfully different and testable.
        """
        
        # Build the user prompt
        prompt = f"""
        Please generate variations of this content for A/B testing:
        
        Original Content:
        {request.content}
        
        {variables_str}
        
        {segments_str}
        
        """
        
        # Add constraints if provided
        if request.constraints:
            constraints_str = "Consider these constraints:\n"
            for key, value in request.constraints.items():
                constraints_str += f"- {key}: {value}\n"
            prompt += f"\n{constraints_str}"
        
        # Append format instructions
        prompt += """
        
        Format your response as a valid JSON object with this structure:
        {
            "variations": [
                {
                    "title": "Variation title",
                    "content": "The variation content",
                    "test_variables": {
                        "variable_name": "value_being_tested",
                        ...
                    },
                    "target_segments": ["Segment 1", ...],
                    "hypothesis": "Hypothesis being tested"
                },
                ...
            ]
        }
        """
        
        # Generate the variations
        result = await llm_client.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.8,  # Higher temperature for more creative variations
            max_tokens=2500
        )
        
        # Parse the result
        try:
            # Find JSON in the response
            json_start = result.find("{")
            json_end = result.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = result[json_start:json_end]
                variations_data = json.loads(json_str)
            else:
                # If no JSON found, try to parse the entire response
                variations_data = json.loads(result)
            
            return ContentVariation(
                original_content=request.content,
                variations=variations_data.get("variations", [])
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing A/B test variations result: {str(e)}", request_id)
            raise HTTPException(status_code=500, detail=f"Error parsing A/B test variations result: {str(e)}")
            
    except Exception as e:
        logger.error(f"Error generating A/B test variations: {str(e)}", request_id)
        raise HTTPException(status_code=500, detail=f"Error generating A/B test variations: {str(e)}")

@app.post("/localize", response_model=JSONResponse)
async def localize_content(
    request: LocalizationRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Localize content for a specific target locale with cultural adaptation."""
    request_id = str(uuid.uuid4())
    
    try:
        logger.info(
            "Content localization request received", 
            request_id, 
            {"source_locale": request.source_locale, "target_locale": request.target_locale}
        )
        
        # Create LLM client
        llm_client = LLMClient()
        
        # Format preserve elements if provided
        preserve_str = ""
        if request.preserve_elements:
            preserve_str = "Elements to preserve untranslated:\n"
            for element in request.preserve_elements:
                preserve_str += f"- {element}\n"
        
        # Format cultural notes if provided
        cultural_str = ""
        if request.cultural_notes:
            cultural_str = "Cultural Considerations:\n"
            for key, value in request.cultural_notes.items():
                cultural_str += f"- {key}: {value}\n"
        
        # Build the system prompt
        system_prompt = f"""
        You are an expert content localization specialist. Adapt the provided {request.content_type.value} content from {request.source_locale} to {request.target_locale}.
        This is not just translation - adapt cultural references, idioms, examples, and tone to be appropriate for the target locale.
        Preserve the core message and intent while making the content feel natural and culturally appropriate.
        """
        
        # Build the user prompt
        prompt = f"""
        Please localize this content from {request.source_locale} to {request.target_locale}:
        
        Original Content ({request.source_locale}):
        {request.content}
        
        {cultural_str}
        
        {preserve_str}
        
        Format your response as a valid JSON object with this structure:
        {{
            "localized_content": "The localized content",
            "adaptations": [
                {{
                    "original": "Original text/reference",
                    "adapted": "Adapted text/reference",
                    "explanation": "Explanation of the adaptation"
                }},
                ...
            ],
            "notes": "Any additional notes or considerations"
        }}
        """
        
        # Generate the localized content
        result = await llm_client.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.4,
            max_tokens=2000
        )
        
        # Parse the result
        try:
            # Find JSON in the response
            json_start = result.find("{")
            json_end = result.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = result[json_start:json_end]
                localization_data = json.loads(json_str)
            else:
                # If no JSON found, try to parse the entire response
                localization_data = json.loads(result)
            
            return JSONResponse(
                content={
                    "request_id": request_id,
                    "source_locale": request.source_locale,
                    "target_locale": request.target_locale,
                    "original_content": request.content,
                    "localization_result": localization_data
                }
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing localization result: {str(e)}", request_id)
            raise HTTPException(status_code=500, detail=f"Error parsing localization result: {str(e)}")
            
    except Exception as e:
        logger.error(f"Error localizing content: {str(e)}", request_id)
        raise HTTPException(status_code=500, detail=f"Error localizing content: {str(e)}")

@app.post("/personalize-recommendation")
async def personalize_recommendation(
    user_profile: PersonalizationProfile = Body(...),
    items: List[Dict[str, Any]] = Body(...),
    recommendation_context: Optional[Dict[str, Any]] = Body(None),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Generate personalized recommendations from a list of items based on user profile."""
    request_id = str(uuid.uuid4())
    
    try:
        logger.info("Recommendation personalization request received", request_id)
        
        # Create LLM client
        llm_client = LLMClient()
        
        # Format user profile (similar to personalize endpoint)
        user_profile_str = "User Profile:\n"
        
        if user_profile.demographics:
            user_profile_str += "Demographics:\n"
            for key, value in user_profile.demographics.items():
                user_profile_str += f"- {key}: {value}\n"
        
        if user_profile.preferences:
            user_profile_str += "Preferences:\n"
            for key, value in user_profile.preferences.items():
                user_profile_str += f"- {key}: {value}\n"
        
        if user_profile.interests:
            user_profile_str += f"Interests: {', '.join(user_profile.interests)}\n"
        
        if user_profile.behavior:
            user_profile_str += "Behavior:\n"
            for key, value in user_profile.behavior.items():
                user_profile_str += f"- {key}: {value}\n"
        
        if user_profile.history:
            user_profile_str += "Recent History:\n"
            for i, item in enumerate(user_profile.history[:5]):  # Limit to 5 most recent items
                user_profile_str += f"- {item.get('type', 'Item')}: {item.get('name', '')}\n"
        
        # Format items for recommendation
        items_str = "Available Items:\n"
        for i, item in enumerate(items):
            items_str += f"Item {i+1}: {item.get('name', '')}\n"
            if 'description' in item:
                items_str += f"Description: {item['description']}\n"
            if 'category' in item:
                items_str += f"Category: {item['category']}\n"
            if 'tags' in item:
                items_str += f"Tags: {', '.join(item['tags'])}\n"
            if 'attributes' in item:
                items_str += "Attributes:\n"
                for key, value in item['attributes'].items():
                    items_str += f"  - {key}: {value}\n"
            items_str += "\n"
        
        # Format recommendation context if provided
        context_str = ""
        if recommendation_context:
            context_str = "Recommendation Context:\n"
            for key, value in recommendation_context.items():
                context_str += f"- {key}: {value}\n"
        
        # Build the system prompt
        system_prompt = """
        You are an expert recommendation system. Generate personalized recommendations based on the user profile and available items.
        Rank items by relevance to the user's preferences, interests, and behavior.
        Provide explanations for why each item is recommended.
        Consider both obvious matches and potential discoveries the user might enjoy.
        """
        
        # Build the user prompt
        prompt = f"""
        Please generate personalized recommendations:
        
        {user_profile_str}
        
        {items_str}
        
        {context_str}
        
        Format your response as a valid JSON object with this structure:
        {{
            "recommendations": [
                {{
                    "item_id": "item_index or item_name",
                    "relevance_score": 0.95,
                    "explanation": "Why this item is recommended",
                    "matches": ["preference1", "interest2", ...]
                }},
                ...
            ],
            "personalization_factors": ["factor1", "factor2", ...],
            "recommendation_strategy": "Description of the strategy used"
        }}
        
        Order recommendations by relevance_score in descending order.
        """
        
        # Generate the recommendations
        result = await llm_client.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.4,
            max_tokens=2000
        )
        
        # Parse the result
        try:
            # Find JSON in the response
            json_start = result.find("{")
            json_end = result.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = result[json_start:json_end]
                recommendation_data = json.loads(json_str)
            else:
                # If no JSON found, try to parse the entire response
                recommendation_data = json.loads(result)
            
            return JSONResponse(
                content={
                    "request_id": request_id,
                    "user_id": user_profile.user_id,
                    "recommendations": recommendation_data
                }
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing recommendation result: {str(e)}", request_id)
            raise HTTPException(status_code=500, detail=f"Error parsing recommendation result: {str(e)}")
            
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}", request_id)
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")