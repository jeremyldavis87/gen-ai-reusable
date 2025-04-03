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
from fastapi.security import HTTPBearer
from fastapi.openapi.utils import get_openapi

from utilities.llm_client import LLMClient, LLMProvider
from utilities.logging_utils import StructuredLogger
from utilities.auth import get_current_user
from utilities.prompts import PromptTemplates

# Create app with detailed documentation
app = FastAPI(
    title="Content Personalization Service",
    description="""
    The Content Personalization Service provides AI-powered content personalization capabilities through a RESTful API.
    
    ## Key Features
    
    * **Content Personalization**: Customize content based on user profiles and preferences
    * **A/B Testing**: Generate and test multiple content variations
    * **Localization**: Adapt content for different locales and cultures
    * **Personalized Recommendations**: Get content recommendations based on user behavior
    
    ## Content Types Supported
    - Text: General text content
    - Marketing: Marketing and promotional content
    - Email: Email communications
    - Product Descriptions: Product and service descriptions
    - News: News articles and updates
    - Educational: Learning materials and courses
    - Documentation: Technical and user documentation
    
    ## Personalization Features
    - User Profile Based: Demographics, preferences, behavior
    - Context Aware: Time, location, device
    - A/B Testing: Multiple variations for testing
    - Localization: Language and cultural adaptation
    - Recommendations: Personalized content suggestions
    
    ## Target Audiences
    - General: General audience
    - Technical: Technical users
    - Executive: Business executives
    - Beginner: New users
    - Intermediate: Regular users
    - Advanced: Expert users
    - Customer: Existing customers
    - Prospect: Potential customers
    
    ## Authentication
    
    All endpoints require authentication using a valid API key or token.
    
    ## Rate Limiting
    
    The API is rate-limited to ensure fair usage. Please check the response headers for rate limit information.
    
    ## Best Practices
    
    1. Always provide comprehensive user profiles for better personalization
    2. Use specific content types and constraints for more accurate results
    3. Include relevant context in recommendation requests
    4. Consider cultural factors in localization requests
    5. Test personalized content with target audience
    6. Monitor performance metrics
    7. Regularly update user profiles
    8. Use appropriate content constraints
    9. Consider mobile and desktop contexts
    10. Follow accessibility guidelines
    
    ## Example Use Cases
    - Marketing Campaign Personalization
    - Email Content Customization
    - Product Description Adaptation
    - Learning Path Recommendations
    - Technical Documentation Personalization
    - News Feed Customization
    """,
    version="1.0.0",
    contact={
        "name": "AI Services Team",
        "email": "ai-services@example.com",
    },
    license_info={
        "name": "Proprietary",
        "url": "https://example.com/license",
    },
    openapi_tags=[
        {
            "name": "personalization",
            "description": "Endpoints for personalizing content based on user profiles and preferences"
        },
        {
            "name": "testing",
            "description": "Endpoints for A/B testing and content variation generation"
        },
        {
            "name": "localization",
            "description": "Endpoints for content localization and cultural adaptation"
        },
        {
            "name": "recommendations",
            "description": "Endpoints for generating personalized content recommendations"
        }
    ]
)

# Security scheme
security = HTTPBearer(
    scheme_name="Bearer",
    description="Enter your JWT token",
    auto_error=True
)

# Setup logger
logger = StructuredLogger("personalization_service")

# Models
class ContentType(str, Enum):
    """
    Supported content types for personalization.
    
    Attributes:
        TEXT: General text content
        MARKETING: Marketing and promotional content
        EMAIL: Email communications
        PRODUCT_DESCRIPTION: Product and service descriptions
        NEWS: News articles and updates
        EDUCATIONAL: Learning materials and courses
        DOCUMENTATION: Technical documentation
    """
    TEXT = "text"
    MARKETING = "marketing"
    EMAIL = "email"
    PRODUCT_DESCRIPTION = "product_description"
    NEWS = "news"
    EDUCATIONAL = "educational"
    DOCUMENTATION = "documentation"
    
    class Config:
        schema_extra = {
            "description": "Type of content to be personalized",
            "example": "marketing"
        }
    
class TargetAudience(str, Enum):
    """
    Target audience categories for content personalization.
    
    Attributes:
        GENERAL: General audience
        TECHNICAL: Technical users
        EXECUTIVE: Business executives
        BEGINNER: New users
        INTERMEDIATE: Regular users
        ADVANCED: Expert users
        CUSTOMER: Existing customers
        PROSPECT: Potential customers
    """
    GENERAL = "general"
    TECHNICAL = "technical"
    EXECUTIVE = "executive"
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    CUSTOMER = "customer"
    PROSPECT = "prospect"
    
    class Config:
        schema_extra = {
            "description": "Target audience for the content",
            "example": "technical"
        }
    
class PersonalizationProfile(BaseModel):
    """
    User profile model for content personalization.
    
    Attributes:
        user_id: Unique identifier for the user
        demographics: Demographic information (age, gender, location, etc.)
        preferences: User preferences (tone, format, topics, etc.)
        behavior: User behavior data (engagement, history, etc.)
        interests: List of user interests
        history: User interaction history
    """
    user_id: Optional[str] = Field(None, description="Unique identifier for the user")
    demographics: Optional[Dict[str, Any]] = Field(None, description="Demographic information")
    preferences: Optional[Dict[str, Any]] = Field(None, description="User preferences")
    behavior: Optional[Dict[str, Any]] = Field(None, description="User behavior data")
    interests: Optional[List[str]] = Field(None, description="User interests")
    history: Optional[List[Dict[str, Any]]] = Field(None, description="User interaction history")
    
    class Config:
        schema_extra = {
            "example": {
                "user_id": "user123",
                "demographics": {
                    "age": 35,
                    "gender": "female",
                    "location": "New York"
                },
                "preferences": {
                    "tone": "professional",
                    "format": "video",
                    "topics": ["technology", "business"]
                },
                "interests": ["AI", "machine learning", "data science"],
                "behavior": {
                    "avg_session_duration": 15,
                    "preferred_time": "morning"
                }
            }
        }

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

class LocalizationResponse(BaseModel):
    request_id: str = Field(..., description="Unique identifier for the request")
    localized_content: str = Field(..., description="Localized content")
    source_locale: str = Field(..., description="Source locale")
    target_locale: str = Field(..., description="Target locale")
    cultural_adaptations: Optional[List[str]] = Field(None, description="List of cultural adaptations made")
    warnings: Optional[List[str]] = Field(None, description="Any warnings or notes about the localization")

class RecommendationResponse(BaseModel):
    request_id: str = Field(..., description="Unique identifier for the request")
    recommendations: List[Dict[str, Any]] = Field(..., description="List of recommended items with scores and explanations")
    personalization_factors: Optional[List[str]] = Field(None, description="Factors used in personalization")
    recommendation_strategy: Optional[str] = Field(None, description="Description of the recommendation strategy used")

# Routes
@app.post("/personalize", 
    response_model=PersonalizationResponse,
    tags=["personalization"],
    summary="Personalize content based on user profile",
    description="""
    Personalizes content by adapting it to match user preferences, demographics, and behavior.
    
    ## Features
    - User profile-based personalization
    - Multiple content type support
    - Target audience adaptation
    - Constraint handling
    - Confidence scoring
    
    ## Content Processing
    - Text analysis
    - User profile matching
    - Content adaptation
    - Format preservation
    - Quality validation
    
    ## Example Use Cases
    - Marketing message personalization
    - Email content customization
    - Product description adaptation
    - Documentation level adjustment
    - News content personalization
    
    ## Request Format
    ```json
    {
        "content": "Welcome to our platform!",
        "content_type": "marketing",
        "user_profile": {
            "user_id": "user123",
            "demographics": {...},
            "preferences": {...}
        },
        "target_audience": "technical",
        "personalization_aspects": ["tone", "examples"]
    }
    ```
    
    ## Response Format
    ```json
    {
        "request_id": "uuid",
        "personalized_content": "...",
        "personalization_summary": "...",
        "variations": [...]
    }
    ```
    """,
    response_description="The personalized content with summary of changes"
)
async def personalize_content(
    request: PersonalizationRequest = Body(
        ...,
        example={
            "content": "Welcome to our platform! We offer a wide range of services to help you achieve your goals.",
            "content_type": "marketing",
            "user_profile": {
                "user_id": "user123",
                "demographics": {
                    "age": 35,
                    "gender": "female",
                    "location": "New York"
                },
                "preferences": {
                    "tone": "professional",
                    "length": "concise",
                    "format": "bullet_points"
                },
                "interests": ["technology", "business", "personal development"],
                "behavior": {
                    "preferred_content_type": "video",
                    "average_session_duration": 15
                }
            },
            "target_audience": "executive",
            "personalization_aspects": ["tone", "examples", "call_to_action"],
            "constraints": {
                "max_length": 500,
                "must_include": ["free trial", "contact us"]
            }
        }
    ),
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

@app.post("/generate-ab-test", 
    response_model=ContentVariation,
    tags=["testing"],
    summary="Generate A/B test variations",
    description="""
    Creates multiple variations of content for A/B testing purposes.
    
    ## Features
    - Multiple variation generation
    - Audience segment targeting
    - Performance hypothesis
    - Content constraints
    - Quality validation
    
    ## Testing Variables
    - Content elements
    - Tone and style
    - Call to action
    - Value proposition
    - Visual elements
    
    ## Example Use Cases
    - Marketing campaign testing
    - Landing page optimization
    - Email subject line testing
    - Product description testing
    - Call-to-action optimization
    
    ## Request Format
    ```json
    {
        "content": "Get 20% off your first purchase!",
        "content_type": "marketing",
        "test_variables": [
            {
                "name": "discount_amount",
                "options": ["20%", "25%", "30%"]
            }
        ],
        "audience_segments": [...]
    }
    ```
    
    ## Response Format
    ```json
    {
        "original_content": "...",
        "variations": [
            {
                "title": "Variation A",
                "content": "...",
                "test_variables": {...}
            }
        ]
    }
    ```
    """,
    response_description="Generated content variations with test metadata"
)
async def generate_ab_test_variations(
    request: ABTestRequest = Body(
        ...,
        example={
            "content": "Get 20% off your first purchase! Limited time offer.",
            "content_type": "marketing",
            "test_variables": [
                {
                    "name": "discount_amount",
                    "options": ["20%", "25%", "30%"]
                },
                {
                    "name": "urgency",
                    "options": ["Limited time offer", "Act now", "Don't miss out"]
                }
            ],
            "audience_segments": [
                {
                    "name": "new_customers",
                    "description": "First-time visitors",
                    "characteristics": ["price_sensitive", "risk_averse"]
                },
                {
                    "name": "returning_customers",
                    "description": "Previous buyers",
                    "characteristics": ["brand_loyal", "value_quality"]
                }
            ],
            "constraints": {
                "max_length": 100,
                "must_include": ["discount", "call_to_action"]
            }
        }
    ),
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

@app.post("/localize", 
    response_model=LocalizationResponse,
    tags=["localization"],
    summary="Localize content for different locales",
    description="""
    Localizes content for specific locales while considering cultural factors.
    
    ## Features
    - Language translation
    - Cultural adaptation
    - Idiom handling
    - Format preservation
    - Quality validation
    
    ## Localization Aspects
    - Language translation
    - Cultural nuances
    - Regional preferences
    - Format adaptation
    - Regulatory compliance
    
    ## Example Use Cases
    - Global marketing campaigns
    - International product launches
    - Multi-language documentation
    - Regional content adaptation
    - Cultural customization
    
    ## Request Format
    ```json
    {
        "content": "Welcome to our platform!",
        "content_type": "marketing",
        "source_locale": "en-US",
        "target_locale": "es-ES",
        "cultural_notes": {
            "formality": "high"
        }
    }
    ```
    
    ## Response Format
    ```json
    {
        "request_id": "uuid",
        "localized_content": "...",
        "cultural_adaptations": [...],
        "warnings": [...]
    }
    ```
    """,
    response_description="Localized content with cultural adaptations"
)
async def localize_content(
    request: LocalizationRequest = Body(
        ...,
        example={
            "content": "Welcome to our platform! We're excited to help you achieve your goals.",
            "content_type": "marketing",
            "source_locale": "en-US",
            "target_locale": "es-ES",
            "cultural_notes": {
                "formality": "high",
                "idioms": "avoid",
                "humor": "professional"
            },
            "preserve_elements": ["company_name", "product_names"]
        }
    ),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Localize content for a specific locale and culture."""
    request_id = str(uuid.uuid4())
    
    try:
        logger.info(
            "Content localization request received", 
            request_id, 
            {
                "content_type": request.content_type,
                "source_locale": request.source_locale,
                "target_locale": request.target_locale
            }
        )
        
        # Create LLM client
        llm_client = LLMClient()
        
        # Build the system prompt
        system_prompt = f"""
        You are an expert content localization system. Localize the provided {request.content_type.value} content from {request.source_locale} to {request.target_locale}.
        Consider cultural nuances, idioms, and regional preferences while maintaining the original meaning and intent.
        """
        
        # Format cultural notes if provided
        cultural_notes_str = ""
        if request.cultural_notes:
            cultural_notes_str = "Cultural Considerations:\n"
            for key, value in request.cultural_notes.items():
                cultural_notes_str += f"- {key}: {value}\n"
        
        # Format elements to preserve if provided
        preserve_str = ""
        if request.preserve_elements:
            preserve_str = "Elements to preserve untranslated:\n"
            for element in request.preserve_elements:
                preserve_str += f"- {element}\n"
        
        # Build the user prompt
        prompt = f"""
        Localize this content from {request.source_locale} to {request.target_locale}:

        {request.content}

        {cultural_notes_str}
        {preserve_str}
        
        Format your response as a valid JSON object with this structure:
        {{
            "localized_content": "The localized content",
            "cultural_adaptations": ["List of cultural adaptations made"],
            "warnings": ["Any warnings or notes about the localization"]
        }}
        """
        
        # Generate the localized content
        result = await llm_client.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.3,  # Lower temperature for more accurate translation
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
            
            return LocalizationResponse(
                request_id=request_id,
                localized_content=localization_data.get("localized_content", ""),
                source_locale=request.source_locale,
                target_locale=request.target_locale,
                cultural_adaptations=localization_data.get("cultural_adaptations"),
                warnings=localization_data.get("warnings")
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing localization result: {str(e)}", request_id)
            
            # If parsing fails, assume the result is the localized content directly
            return LocalizationResponse(
                request_id=request_id,
                localized_content=result,
                source_locale=request.source_locale,
                target_locale=request.target_locale,
                warnings=["Error parsing structured response, using raw output"]
            )
            
    except Exception as e:
        logger.error(f"Error localizing content: {str(e)}", request_id)
        raise HTTPException(status_code=500, detail=f"Error localizing content: {str(e)}")

@app.post("/personalize-recommendation", 
    response_model=RecommendationResponse,
    tags=["recommendations"],
    summary="Get personalized content recommendations",
    description="""
    Generates personalized content recommendations based on user profile and available items.
    
    ## Features
    - Profile-based matching
    - Content relevance scoring
    - Contextual awareness
    - Explanation generation
    - Diversity handling
    
    ## Recommendation Factors
    - User preferences
    - Historical behavior
    - Content attributes
    - Context awareness
    - Time sensitivity
    
    ## Example Use Cases
    - Learning path recommendations
    - Content discovery
    - Product recommendations
    - Reading suggestions
    - Course recommendations
    
    ## Request Format
    ```json
    {
        "user_profile": {
            "user_id": "user123",
            "preferences": {...},
            "history": [...]
        },
        "items": [
            {
                "name": "Advanced Python Course",
                "attributes": {...}
            }
        ],
        "recommendation_context": {
            "purpose": "learning"
        }
    }
    ```
    
    ## Response Format
    ```json
    {
        "request_id": "uuid",
        "recommendations": [
            {
                "item_id": "course123",
                "relevance_score": 0.95,
                "explanation": "..."
            }
        ],
        "personalization_factors": [...]
    }
    ```
    """,
    response_description="Personalized recommendations with explanations and scores"
)
async def personalize_recommendation(
    request: Dict[str, Any] = Body(
        ...,
        example={
            "user_profile": {
                "user_id": "user123",
                "demographics": {
                    "age": 35,
                    "gender": "female",
                    "location": "New York"
                },
                "preferences": {
                    "content_type": "video",
                    "difficulty": "intermediate",
                    "topics": ["AI", "machine learning"]
                },
                "interests": ["technology", "data science", "programming"],
                "behavior": {
                    "average_watch_time": 15,
                    "preferred_duration": "short"
                },
                "history": [
                    {
                        "type": "course",
                        "name": "Introduction to Python",
                        "completion_date": "2023-01-15"
                    },
                    {
                        "type": "article",
                        "name": "Machine Learning Basics",
                        "view_date": "2023-02-01"
                    }
                ]
            },
            "items": [
                {
                    "name": "Advanced Python Programming",
                    "description": "Deep dive into Python's advanced features",
                    "category": "Programming",
                    "tags": ["python", "advanced", "programming"],
                    "attributes": {
                        "difficulty": "advanced",
                        "duration": "2 hours",
                        "format": "video"
                    }
                },
                {
                    "name": "Machine Learning Fundamentals",
                    "description": "Introduction to ML concepts and algorithms",
                    "category": "Data Science",
                    "tags": ["machine learning", "beginner", "data science"],
                    "attributes": {
                        "difficulty": "beginner",
                        "duration": "3 hours",
                        "format": "video"
                    }
                }
            ],
            "recommendation_context": {
                "purpose": "skill_development",
                "time_available": 60,
                "preferred_format": "video",
                "current_skill_level": "intermediate"
            }
        }
    ),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Generate personalized recommendations from a list of items based on user profile."""
    request_id = str(uuid.uuid4())
    
    try:
        logger.info("Recommendation personalization request received", request_id)
        
        # Extract components from the request
        user_profile = PersonalizationProfile(**request["user_profile"])
        items = request["items"]
        recommendation_context = request.get("recommendation_context")
        
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
            
            return RecommendationResponse(
                request_id=request_id,
                recommendations=recommendation_data.get("recommendations", []),
                personalization_factors=recommendation_data.get("personalization_factors"),
                recommendation_strategy=recommendation_data.get("recommendation_strategy")
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing recommendation result: {str(e)}", request_id)
            raise HTTPException(status_code=500, detail=f"Error parsing recommendation result: {str(e)}")
            
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}", request_id)
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    # Let FastAPI generate the base schema
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
        tags=app.openapi_tags,
    )

    # Add security scheme
    if "components" not in openapi_schema:
        openapi_schema["components"] = {}
    
    openapi_schema["components"]["securitySchemes"] = {
        "Bearer": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": "Enter your JWT token"
        }
    }

    # Add global security requirement
    openapi_schema["security"] = [{"Bearer": []}]

    # Update examples and descriptions for each endpoint
    if "paths" in openapi_schema:
        for path, path_item in openapi_schema["paths"].items():
            if "post" not in path_item:
                continue
                
            post_op = path_item["post"]
            
            # Enhance descriptions based on the endpoint
            if path == "/personalize":
                post_op["description"] = """
                Personalizes content by adapting it to match user preferences, demographics, and behavior.
                
                ## Features
                - User profile-based personalization
                - Multiple content type support
                - Target audience adaptation
                - Constraint handling
                - Confidence scoring
                
                ## Content Processing
                - Text analysis
                - User profile matching
                - Content adaptation
                - Format preservation
                - Quality validation
                
                ## Example Use Cases
                - Marketing message personalization
                - Email content customization
                - Product description adaptation
                - Documentation level adjustment
                - News content personalization
                """
                
            elif path == "/generate-ab-test":
                post_op["description"] = """
                Creates multiple variations of content for A/B testing purposes.
                
                ## Features
                - Multiple variation generation
                - Audience segment targeting
                - Performance hypothesis
                - Content constraints
                - Quality validation
                
                ## Testing Variables
                - Content elements
                - Tone and style
                - Call to action
                - Value proposition
                - Visual elements
                
                ## Example Use Cases
                - Marketing campaign testing
                - Landing page optimization
                - Email subject line testing
                - Product description testing
                - Call-to-action optimization
                """
                
            elif path == "/localize":
                post_op["description"] = """
                Localizes content for specific locales while considering cultural factors.
                
                ## Features
                - Language translation
                - Cultural adaptation
                - Idiom handling
                - Format preservation
                - Quality validation
                
                ## Localization Aspects
                - Language translation
                - Cultural nuances
                - Regional preferences
                - Format adaptation
                - Regulatory compliance
                
                ## Example Use Cases
                - Global marketing campaigns
                - International product launches
                - Multi-language documentation
                - Regional content adaptation
                - Cultural customization
                """
                
            elif path == "/personalize-recommendation":
                post_op["description"] = """
                Generates personalized content recommendations based on user profile and available items.
                
                ## Features
                - Profile-based matching
                - Content relevance scoring
                - Contextual awareness
                - Explanation generation
                - Diversity handling
                
                ## Recommendation Factors
                - User preferences
                - Historical behavior
                - Content attributes
                - Context awareness
                - Time sensitivity
                
                ## Example Use Cases
                - Learning path recommendations
                - Content discovery
                - Product recommendations
                - Reading suggestions
                - Course recommendations
                """

    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi