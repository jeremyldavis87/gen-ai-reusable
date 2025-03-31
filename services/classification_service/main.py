# services/classification_service/main.py
from fastapi import FastAPI, Depends, HTTPException, File, UploadFile, Form, Body
from fastapi.responses import JSONResponse, StreamingResponse, Response
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
import uuid
import json
import io
import csv
from enum import Enum
from datetime import datetime
import asyncio

from utilities.llm_client import LLMClient, LLMProvider
from utilities.logging_utils import StructuredLogger
from utilities.auth import get_current_user
from utilities.prompts import PromptTemplates
from utilities.document_utils import extract_text_from_file

# Document parsers
import PyPDF2
import docx
import pandas as pd
from pdfminer.high_level import extract_text as extract_text_pdfminer

# Create app
app = FastAPI(
    title="Classification and Categorization Service",
    description="""
    A service for classifying and categorizing various types of content using AI.
    
    ## Features
    - Single document classification
    - Batch document classification
    - Tabular data classification
    - Support for multiple content types (text, documents, emails, etc.)
    - Various classification schemes (categories, multi-label, hierarchical, etc.)
    - Confidence scoring for classifications
    - Configurable thresholds and limits
    - Enhanced classification instructions with:
        - Context and domain information
        - Classification guidelines
        - Example classifications
        - Constraints and requirements
        - User preferences
    
    ## Content Types
    - TEXT: Plain text content
    - DOCUMENT: Structured documents (PDF, DOCX, etc.)
    - IMAGE: Image files (requires OCR)
    - TABULAR: Tabular data (CSV, Excel)
    - EMAIL: Email messages
    - SOCIAL_MEDIA: Social media posts
    - REVIEW: Customer reviews
    - SUPPORT_TICKET: Support ticket content
    - NEWS_ARTICLE: News articles
    
    ## Classification Types
    - CATEGORY: Single, mutually exclusive category
    - MULTI_LABEL: Multiple tags/labels can apply
    - HIERARCHICAL: Categories with parent-child relationships
    - SENTIMENT: Positive, negative, neutral sentiment
    - INTENT: User intent classification
    - ENTITY: Named entity recognition
    - LANGUAGE: Language detection
    - TOPIC: Topic modeling
    - CUSTOM: Custom classification scheme
    
    ## Classification Instructions
    The service supports detailed classification instructions to guide the AI model:
    
    ### Context
    Provide domain-specific context about the content being classified.
    
    ### Guidelines
    Specify rules and best practices for classification.
    
    ### Examples
    Include example classifications to demonstrate desired outcomes.
    
    ### Constraints
    Define requirements and limitations for classification.
    
    ### Preferences
    Set user preferences for classification approach.
    
    ### Custom Instructions
    Add free-form instructions for additional guidance.
    """
)

# Setup logger
logger = StructuredLogger("classification_service")

# Models
class ContentType(str, Enum):
    TEXT = "text"
    DOCUMENT = "document"
    IMAGE = "image"
    TABULAR = "tabular"
    EMAIL = "email"
    SOCIAL_MEDIA = "social_media"
    REVIEW = "review"
    SUPPORT_TICKET = "support_ticket"
    NEWS_ARTICLE = "news_article"
    
    class Config:
        title = "Content Type"
        description = "Type of content being classified"
        json_schema_extra = {
            "examples": [
                {"value": "document", "description": "PDF, DOCX, or other document format"},
                {"value": "email", "description": "Email message content"},
                {"value": "tabular", "description": "CSV, Excel, or other tabular data"}
            ]
        }
    
class ClassificationType(str, Enum):
    CATEGORY = "category"  # Mutually exclusive categories
    MULTI_LABEL = "multi_label"  # Multiple tags/labels can apply
    HIERARCHICAL = "hierarchical"  # Categories with parent-child relationships
    SENTIMENT = "sentiment"  # Positive, negative, neutral sentiment
    INTENT = "intent"  # User intent classification
    ENTITY = "entity"  # Named entity recognition
    LANGUAGE = "language"  # Language detection
    TOPIC = "topic"  # Topic modeling
    CUSTOM = "custom"  # Custom classification scheme
    
    class Config:
        title = "Classification Type"
        description = "Type of classification to perform"
        json_schema_extra = {
            "examples": [
                {"value": "category", "description": "Single category classification"},
                {"value": "multi_label", "description": "Multiple label classification"},
                {"value": "sentiment", "description": "Sentiment analysis"}
            ]
        }
    
class Category(BaseModel):
    id: str = Field(..., description="Unique identifier for the category")
    name: str = Field(..., description="Name of the category")
    description: Optional[str] = Field(None, description="Description of the category")
    parent_id: Optional[str] = Field(None, description="Parent category ID for hierarchical classifications")
    
    class Config:
        title = "Category"
        description = "Category definition for classification"
        json_schema_extra = {
            "examples": [
                {
                    "id": "urgent",
                    "name": "Urgent",
                    "description": "Requires immediate attention",
                    "parent_id": None
                },
                {
                    "id": "legal",
                    "name": "Legal",
                    "description": "Legal or compliance related",
                    "parent_id": None
                }
            ]
        }
    
class ClassificationRequest(BaseModel):
    content: Optional[str] = Field(None, description="Content to classify (for direct text input)")
    content_type: ContentType = Field(..., description="Type of content being classified")
    classification_type: ClassificationType = Field(..., description="Type of classification to perform")
    categories: List[Category] = Field(..., description="Available categories for classification")
    confidence_threshold: Optional[float] = Field(
        0.5, 
        description="Minimum confidence threshold (0.0-1.0). Only classifications with confidence above this threshold will be included in the results.",
        ge=0.0,
        le=1.0
    )
    max_categories: Optional[int] = Field(
        None, 
        description="Maximum number of categories to assign. If not specified, all categories above the confidence threshold will be included."
    )
    classification_instructions: Optional[Dict[str, Any]] = Field(
        None,
        description="Detailed instructions for the classification process. Can include:"
                   "\n- context: Additional context about the content or domain"
                   "\n- guidelines: Specific guidelines for classification"
                   "\n- examples: Example classifications for reference"
                   "\n- constraints: Any constraints or special considerations"
                   "\n- preferences: User preferences for classification approach"
    )
    custom_instructions: Optional[str] = Field(
        None, 
        description="Additional free-form instructions for classification. This will be appended to the structured instructions."
    )
    
    class Config:
        title = "Classification Request"
        description = "Request parameters for content classification"
        json_schema_extra = {
            "examples": [
                {
                    "content_type": "document",
                    "classification_type": "multi_label",
                    "categories": [
                        {
                            "id": "urgent",
                            "name": "Urgent",
                            "description": "Requires immediate attention"
                        },
                        {
                            "id": "legal",
                            "name": "Legal",
                            "description": "Legal or compliance related"
                        }
                    ],
                    "confidence_threshold": 0.9,
                    "max_categories": 3,
                    "classification_instructions": {
                        "context": "These are internal company documents that need to be classified for compliance purposes.",
                        "guidelines": [
                            "Consider both the content and metadata when classifying",
                            "When in doubt, assign multiple categories rather than missing important classifications"
                        ],
                        "examples": [
                            {
                                "content": "Contract renewal notice for Q4",
                                "classifications": [
                                    {"category_id": "legal", "confidence": 0.95},
                                    {"category_id": "urgent", "confidence": 0.85}
                                ]
                            }
                        ],
                        "constraints": [
                            "All legal documents must be classified with at least one legal category",
                            "Documents with deadlines within 24 hours should be marked as urgent"
                        ],
                        "preferences": {
                            "bias_towards": "safety",
                            "consider_metadata": True,
                            "include_explanation": True
                        }
                    },
                    "custom_instructions": "Please pay special attention to any mentions of deadlines or compliance requirements."
                }
            ]
        }

class ClassificationResult(BaseModel):
    category_id: str = Field(..., description="ID of the assigned category")
    category_name: str = Field(..., description="Name of the assigned category")
    confidence: float = Field(..., description="Confidence score (0.0-1.0)")
    
    class Config:
        title = "Classification Result"
        description = "Result of a single classification"
        json_schema_extra = {
            "examples": [
                {
                    "category_id": "urgent",
                    "category_name": "Urgent",
                    "confidence": 0.95
                }
            ]
        }

class ClassificationResponse(BaseModel):
    request_id: str = Field(..., description="Unique identifier for the request")
    results: List[ClassificationResult] = Field(..., description="Classification results")
    content_excerpt: Optional[str] = Field(None, description="Short excerpt of the classified content")
    processing_time: float = Field(..., description="Processing time in seconds")
    warnings: Optional[List[str]] = Field(None, description="Warnings or issues encountered")
    
    class Config:
        title = "Classification Response"
        description = "Response containing classification results"
        json_schema_extra = {
            "examples": [
                {
                    "request_id": "a534a8c5-3309-4790-935b-6dc1a2e59ba0",
                    "results": [
                        {
                            "category_id": "urgent",
                            "category_name": "Urgent",
                            "confidence": 0.95
                        }
                    ],
                    "content_excerpt": "Subject: Urgent: Contract Review Required...",
                    "processing_time": 2.5,
                    "warnings": None
                }
            ]
        }

class BatchClassificationResponse(BaseModel):
    request_id: str = Field(..., description="Unique identifier for the request")
    results: List[Dict[str, Any]] = Field(..., description="Classification results for each item")
    summary: Optional[Dict[str, Any]] = Field(None, description="Summary of the batch classification")
    
    class Config:
        title = "Batch Classification Response"
        description = "Response containing batch classification results"
        json_schema_extra = {
            "examples": [
                {
                    "request_id": "a534a8c5-3309-4790-935b-6dc1a2e59ba0",
                    "results": [
                        {
                            "filename": "document1.txt",
                            "classifications": [
                                {
                                    "category_id": "urgent",
                                    "category_name": "Urgent",
                                    "confidence": 0.95
                                }
                            ],
                            "content_excerpt": "Subject: Urgent: Contract Review Required...",
                            "success": True
                        }
                    ],
                    "summary": {
                        "total_files": 1,
                        "successful_classifications": 1,
                        "failed_classifications": 0,
                        "failed_files": [],
                        "processing_time": 5.895121
                    }
                }
            ]
        }

class TabularClassificationResponse(BaseModel):
    request_id: str = Field(..., description="Unique identifier for the request")
    results: List[Dict[str, Any]] = Field(..., description="Classification results for each row")
    summary: Optional[Dict[str, Any]] = Field(None, description="Summary of the tabular classification")
    warnings: Optional[List[str]] = Field(None, description="Warnings or issues encountered")
    original_data: Optional[List[Dict[str, Any]]] = Field(None, description="Original data with additional classification columns")
    
    class Config:
        title = "Tabular Classification Response"
        description = "Response containing tabular classification results"
        json_schema_extra = {
            "examples": [
                {
                    "request_id": "a534a8c5-3309-4790-935b-6dc1a2e59ba0",
                    "results": [
                        {
                            "id": "row1",
                            "content": "Sample content",
                            "classifications": [
                                {
                                    "category_id": "urgent",
                                    "category_name": "Urgent",
                                    "confidence": 0.95
                                }
                            ],
                            "success": True
                        }
                    ],
                    "summary": {
                        "total_rows": 1,
                        "successful_classifications": 1,
                        "failed_classifications": 0
                    },
                    "original_data": [
                        {
                            "feedback_id": "row1",
                            "feedback_text": "Sample content",
                            "date": "2024-03-15",
                            "user_id": "user123",
                            "category_id": "urgent",
                            "category_name": "Urgent",
                            "confidence": 0.95,
                            "success": True
                        }
                    ],
                    "warnings": None
                }
            ]
        }

# Helper functions
def validate_categories(categories: List[Category]) -> None:
    """Validate the category structure."""
    # Check for duplicate IDs
    category_ids = [category.id for category in categories]
    if len(category_ids) != len(set(category_ids)):
        raise ValueError("Duplicate category IDs found")
    
    # For hierarchical classifications, validate parent-child relationships
    category_dict = {category.id: category for category in categories}
    for category in categories:
        if category.parent_id and category.parent_id not in category_dict:
            raise ValueError(f"Parent category ID '{category.parent_id}' not found for category '{category.id}'")

def format_categories_for_prompt(categories: List[Category], classification_type: ClassificationType) -> str:
    """Format categories for the LLM prompt."""
    if classification_type == ClassificationType.HIERARCHICAL:
        # Format hierarchical categories to show the structure
        # First, build a hierarchy map
        hierarchy = {}
        roots = []
        for category in categories:
            if not category.parent_id:
                roots.append(category)
            else:
                if category.parent_id not in hierarchy:
                    hierarchy[category.parent_id] = []
                hierarchy[category.parent_id].append(category)
        
        # Recursive function to build the formatted string
        def format_hierarchy(category, level=0):
            indent = "  " * level
            result = f"{indent}- {category.name}: {category.description or ''}\n"
            for child in hierarchy.get(category.id, []):
                result += format_hierarchy(child, level + 1)
            return result
        
        result = "Categories hierarchy:\n"
        for root in roots:
            result += format_hierarchy(root)
        
        return result
    else:
        # For non-hierarchical classifications, just list the categories
        result = "Categories:\n"
        for category in categories:
            result += f"- {category.name}: {category.description or ''}\n"
        
        return result

# Routes
@app.post("/classify", response_model=ClassificationResponse)
async def classify_content(
    request: Union[str, ClassificationRequest] = Body(...),
    file: Optional[UploadFile] = File(None),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Classify a single piece of content according to specified categories.
    
    This endpoint supports both direct text input and file uploads.
    
    ## Parameters
    - **request**: Classification parameters including:
        - Content type and classification type
        - Available categories
        - Confidence threshold and max categories
        - Enhanced classification instructions:
            - Context about the content/domain
            - Classification guidelines
            - Example classifications
            - Constraints and requirements
            - User preferences
        - Custom free-form instructions
    - **file**: Optional file to classify (if not using direct text input)
    
    ## Returns
    - Classification results with confidence scores
    - Content excerpt
    - Processing time
    - Any warnings or issues encountered
    
    ## Example
    ```json
    {
        "content_type": "document",
        "classification_type": "multi_label",
        "categories": [
            {
                "id": "urgent",
                "name": "Urgent",
                "description": "Requires immediate attention"
            },
            {
                "id": "legal",
                "name": "Legal",
                "description": "Legal or compliance related"
            }
        ],
        "confidence_threshold": 0.9,
        "max_categories": 3,
        "classification_instructions": {
            "context": "These are internal company documents that need to be classified for compliance purposes.",
            "guidelines": [
                "Consider both the content and metadata when classifying",
                "When in doubt, assign multiple categories rather than missing important classifications"
            ],
            "examples": [
                {
                    "content": "Contract renewal notice for Q4",
                    "classifications": [
                        {"category_id": "legal", "confidence": 0.95},
                        {"category_id": "urgent", "confidence": 0.85}
                    ]
                }
            ],
            "constraints": [
                "All legal documents must be classified with at least one legal category",
                "Documents with deadlines within 24 hours should be marked as urgent"
            ],
            "preferences": {
                "bias_towards": "safety",
                "consider_metadata": true,
                "include_explanation": true
            }
        },
        "custom_instructions": "Please pay special attention to any mentions of deadlines or compliance requirements."
    }
    ```
    """
    request_id = str(uuid.uuid4())
    start_time = datetime.now()
    
    try:
        # Handle stringified JSON request
        if isinstance(request, str):
            try:
                request_data = json.loads(request)
                request = ClassificationRequest(**request_data)
            except json.JSONDecodeError as e:
                raise HTTPException(status_code=422, detail=f"Invalid JSON string: {str(e)}")
        
        # Validate categories
        validate_categories(request.categories)
        
        logger.info(
            "Classification request received", 
            request_id, 
            {"content_type": request.content_type, "classification_type": request.classification_type}
        )
        
        # Get content from file or request
        content_text = request.content
        if file:
            content_text = await extract_text_from_file(file)
        
        if not content_text:
            raise ValueError("No content provided for classification")
        
        # Format categories for the prompt
        categories_formatted = format_categories_for_prompt(request.categories, request.classification_type)
        
        # Create LLM client
        llm_client = LLMClient()
        
        # Build instructions based on classification type
        classification_instructions = ""
        if request.classification_type == ClassificationType.CATEGORY:
            classification_instructions = "Assign exactly one category that best fits the content."
        elif request.classification_type == ClassificationType.MULTI_LABEL:
            max_categories = request.max_categories or len(request.categories)
            classification_instructions = f"Assign up to {max_categories} categories that apply to the content."
        elif request.classification_type == ClassificationType.HIERARCHICAL:
            classification_instructions = "Assign categories considering the hierarchical structure. Start with the most general category and proceed to more specific subcategories."
        elif request.classification_type == ClassificationType.SENTIMENT:
            classification_instructions = "Determine the sentiment expressed in the content (positive, negative, or neutral)."
        elif request.classification_type == ClassificationType.INTENT:
            classification_instructions = "Identify the primary intent behind the content."
        elif request.classification_type == ClassificationType.ENTITY:
            classification_instructions = "Identify and categorize named entities in the content."
        elif request.classification_type == ClassificationType.LANGUAGE:
            classification_instructions = "Identify the language of the content."
        elif request.classification_type == ClassificationType.TOPIC:
            classification_instructions = "Identify the main topics discussed in the content."
        elif request.classification_type == ClassificationType.CUSTOM:
            classification_instructions = "Classify the content according to the provided categories."
        
        # Build enhanced instructions from classification_instructions field
        enhanced_instructions = []
        if request.classification_instructions:
            if "context" in request.classification_instructions:
                enhanced_instructions.append(f"Context: {request.classification_instructions['context']}")
            
            if "guidelines" in request.classification_instructions:
                enhanced_instructions.append("Guidelines:")
                for guideline in request.classification_instructions["guidelines"]:
                    enhanced_instructions.append(f"- {guideline}")
            
            if "examples" in request.classification_instructions:
                enhanced_instructions.append("Example Classifications:")
                for example in request.classification_instructions["examples"]:
                    enhanced_instructions.append(f"Content: {example['content']}")
                    enhanced_instructions.append("Classifications:")
                    for cls in example["classifications"]:
                        enhanced_instructions.append(f"  - {cls['category_id']} (confidence: {cls['confidence']})")
            
            if "constraints" in request.classification_instructions:
                enhanced_instructions.append("Constraints:")
                for constraint in request.classification_instructions["constraints"]:
                    enhanced_instructions.append(f"- {constraint}")
            
            if "preferences" in request.classification_instructions:
                enhanced_instructions.append("Preferences:")
                for key, value in request.classification_instructions["preferences"].items():
                    enhanced_instructions.append(f"- {key}: {value}")
        
        # Combine all instructions
        final_instructions = classification_instructions
        if enhanced_instructions:
            final_instructions += "\n\n" + "\n".join(enhanced_instructions)
        if request.custom_instructions:
            final_instructions += f"\n\nAdditional Instructions:\n{request.custom_instructions}"
        
        # Build the system prompt
        system_prompt = f"""
        You are an expert content classifier specialized in {request.classification_type.value} classification.
        You will analyze {request.content_type.value} content and assign appropriate categories based on the provided classification scheme.
        
        {final_instructions}
        
        For each assigned category, provide a confidence score between 0.0 and 1.0.
        Only assign categories with confidence above {request.confidence_threshold}.
        """
        
        # Build the user prompt using the categorization template
        prompt = PromptTemplates.categorization(
            content_text,
            [f"{cat.id}: {cat.name}" for cat in request.categories],
            final_instructions
        )
        
        # Append format instructions
        prompt += """
        
        Format your response as a valid JSON object with this structure:
        {
            "classifications": [
                {
                    "category_id": "category_id",
                    "category_name": "Category Name",
                    "confidence": 0.95
                },
                ...
            ],
            "explanation": "Brief explanation of why these categories were assigned"
        }
        """
        
        # Generate the classification
        result = await llm_client.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.1,  # Lower temperature for more deterministic classification
            max_tokens=1000
        )
        
        # Parse the result
        try:
            # Find JSON in the response (the LLM might include explanatory text before/after the JSON)
            json_start = result.find("{")
            json_end = result.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = result[json_start:json_end]
                classification_data = json.loads(json_str)
            else:
                # If no JSON found, try to parse the entire response
                classification_data = json.loads(result)
            
            classifications = classification_data.get("classifications", [])
            
            # Filter classifications by confidence threshold
            filtered_classifications = [
                cls for cls in classifications 
                if cls["confidence"] >= request.confidence_threshold
            ]
            
            # Sort by confidence (highest first)
            filtered_classifications.sort(key=lambda x: x["confidence"], reverse=True)
            
            # Limit to max_categories if specified
            if request.max_categories:
                filtered_classifications = filtered_classifications[:request.max_categories]
            
            # Create a short excerpt of the content
            excerpt = content_text[:150] + "..." if len(content_text) > 150 else content_text
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return ClassificationResponse(
                request_id=request_id,
                results=filtered_classifications,
                content_excerpt=excerpt,
                processing_time=processing_time,
                warnings=None
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing classification result: {str(e)}", request_id)
            return ClassificationResponse(
                request_id=request_id,
                results=[],
                content_excerpt=content_text[:150] + "..." if len(content_text) > 150 else content_text,
                processing_time=(datetime.now() - start_time).total_seconds(),
                warnings=[f"Error parsing classification result: {str(e)}", "Raw result: " + result[:100] + "..."]
            )
            
    except Exception as e:
        logger.error(f"Error classifying content: {str(e)}", request_id)
        raise HTTPException(status_code=500, detail=f"Error classifying content: {str(e)}")

@app.post("/batch-classify", response_model=BatchClassificationResponse)
async def batch_classify(
    request: Union[str, ClassificationRequest] = Body(...),
    files: List[UploadFile] = File(...),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Batch classify multiple files according to specified categories.
    
    ## Parameters
    - **request**: Classification parameters including:
        - Content type and classification type
        - Available categories
        - Confidence threshold and max categories
        - Enhanced classification instructions:
            - Context about the content/domain
            - Classification guidelines
            - Example classifications
            - Constraints and requirements
            - User preferences
        - Custom free-form instructions
    - **files**: List of files to classify
    
    ## Returns
    - Classification results for each file
    - Summary including:
        - Total number of files processed
        - Number of successful classifications
        - Number of failed classifications
        - List of failed files
        - Total processing time
    
    ## Example
    ```json
    {
        "content_type": "document",
        "classification_type": "multi_label",
        "categories": [
            {
                "id": "urgent",
                "name": "Urgent",
                "description": "Requires immediate attention"
            },
            {
                "id": "legal",
                "name": "Legal",
                "description": "Legal or compliance related"
            }
        ],
        "confidence_threshold": 0.9,
        "max_categories": 3,
        "classification_instructions": {
            "context": "These are internal company documents that need to be classified for compliance purposes.",
            "guidelines": [
                "Consider both the content and metadata when classifying",
                "When in doubt, assign multiple categories rather than missing important classifications"
            ],
            "examples": [
                {
                    "content": "Contract renewal notice for Q4",
                    "classifications": [
                        {"category_id": "legal", "confidence": 0.95},
                        {"category_id": "urgent", "confidence": 0.85}
                    ]
                }
            ],
            "constraints": [
                "All legal documents must be classified with at least one legal category",
                "Documents with deadlines within 24 hours should be marked as urgent"
            ],
            "preferences": {
                "bias_towards": "safety",
                "consider_metadata": true,
                "include_explanation": true
            }
        },
        "custom_instructions": "Please pay special attention to any mentions of deadlines or compliance requirements."
    }
    ```
    """
    request_id = str(uuid.uuid4())
    start_time = datetime.now()
    
    try:
        # Handle stringified JSON request
        if isinstance(request, str):
            try:
                request_data = json.loads(request)
                request = ClassificationRequest(**request_data)
            except json.JSONDecodeError as e:
                raise HTTPException(status_code=422, detail=f"Invalid JSON string: {str(e)}")
        
        # Validate categories
        validate_categories(request.categories)
        
        logger.info(
            "Batch classification request received", 
            request_id, 
            {"content_type": request.content_type, "classification_type": request.classification_type, "file_count": len(files)}
        )
        
        results = []
        
        # Process each file
        for file in files:
            try:
                # Extract text from the file
                content_text = await extract_text_from_file(file)
                
                # Create LLM client
                llm_client = LLMClient()
                
                # Format categories
                categories_formatted = format_categories_for_prompt(request.categories, request.classification_type)
                
                # Build instructions based on classification type
                classification_instructions = ""
                if request.classification_type == ClassificationType.CATEGORY:
                    classification_instructions = "Assign exactly one category that best fits the content."
                elif request.classification_type == ClassificationType.MULTI_LABEL:
                    max_categories = request.max_categories or len(request.categories)
                    classification_instructions = f"Assign up to {max_categories} categories that apply to the content."
                elif request.classification_type == ClassificationType.HIERARCHICAL:
                    classification_instructions = "Assign categories considering the hierarchical structure."
                elif request.classification_type == ClassificationType.SENTIMENT:
                    classification_instructions = "Determine the sentiment expressed in the content (positive, negative, or neutral)."
                elif request.classification_type == ClassificationType.INTENT:
                    classification_instructions = "Identify the primary intent behind the content."
                elif request.classification_type == ClassificationType.ENTITY:
                    classification_instructions = "Identify and categorize named entities in the content."
                elif request.classification_type == ClassificationType.LANGUAGE:
                    classification_instructions = "Identify the language of the content."
                elif request.classification_type == ClassificationType.TOPIC:
                    classification_instructions = "Identify the main topics discussed in the content."
                elif request.classification_type == ClassificationType.CUSTOM:
                    classification_instructions = "Classify the content according to the provided categories."
                
                # Build enhanced instructions from classification_instructions field
                enhanced_instructions = []
                if request.classification_instructions:
                    if "context" in request.classification_instructions:
                        enhanced_instructions.append(f"Context: {request.classification_instructions['context']}")
                    
                    if "guidelines" in request.classification_instructions:
                        enhanced_instructions.append("Guidelines:")
                        for guideline in request.classification_instructions["guidelines"]:
                            enhanced_instructions.append(f"- {guideline}")
                    
                    if "examples" in request.classification_instructions:
                        enhanced_instructions.append("Example Classifications:")
                        for example in request.classification_instructions["examples"]:
                            enhanced_instructions.append(f"Content: {example['content']}")
                            enhanced_instructions.append("Classifications:")
                            for cls in example["classifications"]:
                                enhanced_instructions.append(f"  - {cls['category_id']} (confidence: {cls['confidence']})")
                    
                    if "constraints" in request.classification_instructions:
                        enhanced_instructions.append("Constraints:")
                        for constraint in request.classification_instructions["constraints"]:
                            enhanced_instructions.append(f"- {constraint}")
                    
                    if "preferences" in request.classification_instructions:
                        enhanced_instructions.append("Preferences:")
                        for key, value in request.classification_instructions["preferences"].items():
                            enhanced_instructions.append(f"- {key}: {value}")
                
                # Combine all instructions
                final_instructions = classification_instructions
                if enhanced_instructions:
                    final_instructions += "\n\n" + "\n".join(enhanced_instructions)
                if request.custom_instructions:
                    final_instructions += f"\n\nAdditional Instructions:\n{request.custom_instructions}"
                
                # Build the user prompt
                prompt = PromptTemplates.categorization(
                    content_text,
                    [f"{cat.id}: {cat.name}" for cat in request.categories],
                    final_instructions
                )
                
                # Append format instructions
                prompt += """
                
                Format your response as a valid JSON object with this structure:
                {
                    "classifications": [
                        {
                            "category_id": "category_id",
                            "category_name": "Category Name",
                            "confidence": 0.95
                        },
                        ...
                    ],
                    "explanation": "Brief explanation of why these categories were assigned"
                }
                """
                
                # Generate the classification
                result = await llm_client.generate(
                    prompt=prompt,
                    system_prompt=f"""
                    You are classifying content from the file {file.filename}.
                    
                    {final_instructions}
                    
                    For each assigned category, provide a confidence score between 0.0 and 1.0.
                    Only assign categories with confidence above {request.confidence_threshold}.
                    """,
                    temperature=0.1,
                    max_tokens=1000
                )
                
                # Parse the result
                try:
                    # Find JSON in the response
                    json_start = result.find("{")
                    json_end = result.rfind("}") + 1
                    if json_start >= 0 and json_end > json_start:
                        json_str = result[json_start:json_end]
                        classification_data = json.loads(json_str)
                    else:
                        classification_data = json.loads(result)
                    
                    classifications = classification_data.get("classifications", [])
                    
                    # Filter classifications by confidence threshold
                    filtered_classifications = [
                        cls for cls in classifications 
                        if cls["confidence"] >= request.confidence_threshold
                    ]
                    
                    # Sort by confidence (highest first)
                    filtered_classifications.sort(key=lambda x: x["confidence"], reverse=True)
                    
                    # Limit to max_categories if specified
                    if request.max_categories:
                        filtered_classifications = filtered_classifications[:request.max_categories]
                    
                    # Create a short excerpt of the content
                    excerpt = content_text[:150] + "..." if len(content_text) > 150 else content_text
                    
                    results.append({
                        "filename": file.filename,
                        "classifications": filtered_classifications,
                        "content_excerpt": excerpt,
                        "success": True
                    })
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing classification result for {file.filename}: {str(e)}", request_id)
                    results.append({
                        "filename": file.filename,
                        "error": f"Error parsing classification result: {str(e)}",
                        "success": False
                    })
                    
            except Exception as e:
                # Log error but continue processing other files
                logger.error(f"Error processing file {file.filename}: {str(e)}", request_id)
                results.append({
                    "filename": file.filename,
                    "error": str(e),
                    "success": False
                })
        
        # Create a summary
        successful = sum(1 for result in results if result.get("success", False))
        failed = len(results) - successful
        failed_files = [result["filename"] for result in results if not result.get("success", False)]
        
        summary = {
            "total_files": len(files),
            "successful_classifications": successful,
            "failed_classifications": failed,
            "failed_files": failed_files,
            "processing_time": (datetime.now() - start_time).total_seconds()
        }
        
        return BatchClassificationResponse(
            request_id=request_id,
            results=results,
            summary=summary
        )
        
    except Exception as e:
        logger.error(f"Error in batch classification: {str(e)}", request_id)
        raise HTTPException(status_code=500, detail=f"Error in batch classification: {str(e)}")

@app.post("/classify-tabular", response_model=None)
async def classify_tabular_data(
    file: UploadFile = File(...),
    request: Union[str, ClassificationRequest] = Body(...),
    column_to_classify: str = Body(..., description="Name of the column containing text to classify"),
    id_column: Optional[str] = Body(None, description="Optional column to use as row identifier"),
    return_original_format: bool = Body(False, description="If true, returns data in original format with additional classification columns"),
    response_format: str = Body("json", description="Response format: 'json' or 'csv'"),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Classify data from a specific column in a tabular file (CSV, Excel, etc.).
    
    ## Parameters
    - **file**: Tabular data file (CSV, Excel)
    - **request**: Classification parameters including:
        - Content type and classification type
        - Available categories
        - Confidence threshold and max categories
        - Enhanced classification instructions:
            - Context about the content/domain
            - Classification guidelines
            - Example classifications
            - Constraints and requirements
            - User preferences
        - Custom free-form instructions
    - **column_to_classify**: Name of the column containing text to classify
    - **id_column**: Optional column to use as row identifier
    - **return_original_format**: If true, returns data in original format with additional classification columns
    - **response_format**: Response format: 'json' or 'csv'
    
    ## Returns
    - Classification results for each row
    - Summary including:
        - Total number of rows processed
        - Number of successful classifications
        - Number of failed classifications
    - If return_original_format is true, also includes original data with classification columns
    - If response_format is 'csv', returns a CSV file with the results
    
    ## Example
    ```json
    {
        "content_type": "tabular",
        "classification_type": "sentiment",
        "categories": [
            {
                "id": "positive",
                "name": "Positive",
                "description": "Expresses positive sentiment or satisfaction"
            },
            {
                "id": "negative",
                "name": "Negative",
                "description": "Expresses negative sentiment or dissatisfaction"
            },
            {
                "id": "neutral",
                "name": "Neutral",
                "description": "Expresses neutral sentiment or balanced feedback"
            }
        ],
        "confidence_threshold": 0.5,
        "max_categories": 1,
        "classification_instructions": {
            "context": "These are customer feedback entries that need to be classified for sentiment analysis.",
            "guidelines": [
                "Consider the overall tone and emotional content of the feedback",
                "Look for explicit positive or negative language",
                "Consider context and implications of the feedback",
                "Handle mixed sentiment by focusing on the dominant tone"
            ],
            "examples": [
                {
                    "content": "The product is excellent and works perfectly!",
                    "classifications": [
                        {"category_id": "positive", "confidence": 0.95}
                    ]
                },
                {
                    "content": "I'm having issues with the system and it's frustrating.",
                    "classifications": [
                        {"category_id": "negative", "confidence": 0.85}
                    ]
                },
                {
                    "content": "The features are good but could use some improvements.",
                    "classifications": [
                        {"category_id": "neutral", "confidence": 0.75}
                    ]
                }
            ],
            "constraints": [
                "Assign exactly one sentiment category per feedback",
                "Consider both explicit and implicit sentiment",
                "Handle sarcasm appropriately"
            ],
            "preferences": {
                "bias_towards": "accuracy",
                "consider_context": true,
                "handle_sarcasm": true
            }
        }
    }
    ```
    
    Additional parameters:
    ```json
    {
        "column_to_classify": "feedback_text",
        "id_column": "feedback_id",
        "return_original_format": true,
        "response_format": "csv"
    }
    ```
    
    Sample CSV file (sample_feedback.csv):
    ```csv
    feedback_id,feedback_text,date,user_id
    1,"The new interface is amazing! It's much more intuitive and faster to use.",2024-03-15,user123
    2,"Having trouble with the payment system. It keeps showing an error message.",2024-03-15,user456
    3,"The product meets my needs, but the documentation could be improved.",2024-03-15,user789
    ```
    
    Expected CSV response:
    ```csv
    feedback_id,feedback_text,date,user_id,category_id,category_name,confidence,success
    1,"The new interface is amazing! It's much more intuitive and faster to use.",2024-03-15,user123,positive,Positive,0.95,true
    2,"Having trouble with the payment system. It keeps showing an error message.",2024-03-15,user456,negative,Negative,0.85,true
    3,"The product meets my needs, but the documentation could be improved.",2024-03-15,user789,neutral,Neutral,0.75,true
    ```
    """
    request_id = str(uuid.uuid4())
    
    try:
        # Handle stringified JSON request
        if isinstance(request, str):
            try:
                request_data = json.loads(request)
                request = ClassificationRequest(**request_data)
            except json.JSONDecodeError as e:
                raise HTTPException(status_code=422, detail=f"Invalid JSON string: {str(e)}")
        
        # Validate request
        if request.content_type != ContentType.TABULAR:
            raise ValueError("Content type must be 'tabular' for this endpoint")
        
        logger.info(
            "Tabular classification request received", 
            request_id, 
            {"filename": file.filename, "column_to_classify": column_to_classify}
        )
        
        # Read the file
        content = await file.read()
        file_extension = file.filename.split('.')[-1].lower()
        
        # Parse the tabular data
        data = []
        original_data = []  # Store original data for return_original_format
        column_indices = {}
        
        if file_extension in ['csv', 'txt', 'tsv', 'psv', 'ssv']:
            # Determine delimiter based on file extension
            delimiter = {
                'csv': ',',
                'txt': ',',
                'tsv': '\t',
                'psv': '|',
                'ssv': ';'
            }[file_extension]
            
            # Parse delimited text file
            text_content = content.decode('utf-8')
            csv_reader = csv.reader(io.StringIO(text_content), delimiter=delimiter)
            headers = next(csv_reader)
            
            # Find the column indices
            for i, header in enumerate(headers):
                column_indices[header] = i
            
            if column_to_classify not in column_indices:
                raise ValueError(f"Column '{column_to_classify}' not found in the file")
            
            classify_index = column_indices[column_to_classify]
            id_index = column_indices.get(id_column, None) if id_column else None
            
            # Read the data
            for row in csv_reader:
                if len(row) > classify_index:
                    # Create original data row
                    original_row = {}
                    for header, index in column_indices.items():
                        if index < len(row):
                            original_row[header] = row[index]
                    
                    item = {
                        "content": row[classify_index],
                        "id": row[id_index] if id_index is not None and id_index < len(row) else None
                    }
                    data.append(item)
                    original_data.append(original_row)
        
        elif file_extension in ['xls', 'xlsx']:
            # Parse Excel using pandas
            try:
                # Create a BytesIO object from the content
                excel_file = io.BytesIO(content)
                
                # Read Excel file into pandas DataFrame
                df = pd.read_excel(excel_file)
                
                # Get column names
                headers = df.columns.tolist()
                
                # Find the column indices
                for i, header in enumerate(headers):
                    column_indices[header] = i
                
                if column_to_classify not in column_indices:
                    raise ValueError(f"Column '{column_to_classify}' not found in the Excel file")
                
                classify_index = column_indices[column_to_classify]
                id_index = column_indices.get(id_column, None) if id_column else None
                
                # Convert DataFrame to list of dictionaries
                for _, row in df.iterrows():
                    # Create original data row
                    original_row = {}
                    for header in headers:
                        original_row[header] = str(row[header]) if pd.notna(row[header]) else ""
                    
                    item = {
                        "content": str(row[headers[classify_index]]) if pd.notna(row[headers[classify_index]]) else "",
                        "id": str(row[headers[id_index]]) if id_index is not None and pd.notna(row[headers[id_index]]) else None
                    }
                    data.append(item)
                    original_data.append(original_row)
                    
            except Exception as e:
                raise ValueError(f"Error parsing Excel file: {str(e)}")
        
        elif file_extension == 'json':
            try:
                # Parse JSON file
                json_data = json.loads(content.decode('utf-8'))
                
                # Handle both array of objects and single object
                if isinstance(json_data, dict):
                    json_data = [json_data]
                
                # Get headers from first object
                if not json_data:
                    raise ValueError("Empty JSON file")
                
                headers = list(json_data[0].keys())
                
                # Find the column indices
                for i, header in enumerate(headers):
                    column_indices[header] = i
                
                if column_to_classify not in column_indices:
                    raise ValueError(f"Column '{column_to_classify}' not found in the JSON file")
                
                classify_index = column_indices[column_to_classify]
                id_index = column_indices.get(id_column, None) if id_column else None
                
                # Process each row
                for row in json_data:
                    # Create original data row
                    original_row = {}
                    for header in headers:
                        original_row[header] = str(row.get(header, ""))
                    
                    item = {
                        "content": str(row.get(headers[classify_index], "")),
                        "id": str(row.get(headers[id_index])) if id_index is not None else None
                    }
                    data.append(item)
                    original_data.append(original_row)
                    
            except json.JSONDecodeError as e:
                raise ValueError(f"Error parsing JSON file: {str(e)}")
        
        elif file_extension == 'jsonl':
            try:
                # Parse JSON Lines file
                text_content = content.decode('utf-8')
                lines = text_content.strip().split('\n')
                
                if not lines:
                    raise ValueError("Empty JSONL file")
                
                # Get headers from first line
                first_row = json.loads(lines[0])
                headers = list(first_row.keys())
                
                # Find the column indices
                for i, header in enumerate(headers):
                    column_indices[header] = i
                
                if column_to_classify not in column_indices:
                    raise ValueError(f"Column '{column_to_classify}' not found in the JSONL file")
                
                classify_index = column_indices[column_to_classify]
                id_index = column_indices.get(id_column, None) if id_column else None
                
                # Process each line
                for line in lines:
                    row = json.loads(line)
                    
                    # Create original data row
                    original_row = {}
                    for header in headers:
                        original_row[header] = str(row.get(header, ""))
                    
                    item = {
                        "content": str(row.get(headers[classify_index], "")),
                        "id": str(row.get(headers[id_index])) if id_index is not None else None
                    }
                    data.append(item)
                    original_data.append(original_row)
                    
            except json.JSONDecodeError as e:
                raise ValueError(f"Error parsing JSONL file: {str(e)}")
        
        else:
            raise ValueError(f"Unsupported file type: {file_extension}. Supported types are: csv, txt, tsv, psv, ssv, xls, xlsx, json, jsonl")
        
        # Process each row (in a real service, you might want to batch these)
        results = []
        llm_client = LLMClient()
        
        for i, item in enumerate(data):
            content_text = item["content"]
            
            # Skip empty cells
            if not content_text:
                result = {
                    "id": item["id"],
                    "content": content_text,
                    "classifications": [],
                    "error": "Empty content",
                    "success": False
                }
                results.append(result)
                if return_original_format:
                    original_data[i].update({
                        "category_id": None,
                        "category_name": None,
                        "confidence": None,
                        "success": False,
                        "error": "Empty content"
                    })
                continue
            
            # Format categories
            categories_formatted = format_categories_for_prompt(request.categories, request.classification_type)
            
            # Build instructions based on classification type
            classification_instructions = ""
            if request.classification_type == ClassificationType.CATEGORY:
                classification_instructions = "Assign exactly one category that best fits the content."
            elif request.classification_type == ClassificationType.MULTI_LABEL:
                max_categories = request.max_categories or len(request.categories)
                classification_instructions = f"Assign up to {max_categories} categories that apply to the content."
            elif request.classification_type == ClassificationType.HIERARCHICAL:
                classification_instructions = "Assign categories considering the hierarchical structure."
            elif request.classification_type == ClassificationType.SENTIMENT:
                classification_instructions = "Determine the sentiment expressed in the content (positive, negative, or neutral)."
            elif request.classification_type == ClassificationType.INTENT:
                classification_instructions = "Identify the primary intent behind the content."
            elif request.classification_type == ClassificationType.ENTITY:
                classification_instructions = "Identify and categorize named entities in the content."
            elif request.classification_type == ClassificationType.LANGUAGE:
                classification_instructions = "Identify the language of the content."
            elif request.classification_type == ClassificationType.TOPIC:
                classification_instructions = "Identify the main topics discussed in the content."
            elif request.classification_type == ClassificationType.CUSTOM:
                classification_instructions = "Classify the content according to the provided categories."
            
            # Build enhanced instructions from classification_instructions field
            enhanced_instructions = []
            if request.classification_instructions:
                if "context" in request.classification_instructions:
                    enhanced_instructions.append(f"Context: {request.classification_instructions['context']}")
                
                if "guidelines" in request.classification_instructions:
                    enhanced_instructions.append("Guidelines:")
                    for guideline in request.classification_instructions["guidelines"]:
                        enhanced_instructions.append(f"- {guideline}")
                
                if "examples" in request.classification_instructions:
                    enhanced_instructions.append("Example Classifications:")
                    for example in request.classification_instructions["examples"]:
                        enhanced_instructions.append(f"Content: {example['content']}")
                        enhanced_instructions.append("Classifications:")
                        for cls in example["classifications"]:
                            enhanced_instructions.append(f"  - {cls['category_id']} (confidence: {cls['confidence']})")
                
                if "constraints" in request.classification_instructions:
                    enhanced_instructions.append("Constraints:")
                    for constraint in request.classification_instructions["constraints"]:
                        enhanced_instructions.append(f"- {constraint}")
                
                if "preferences" in request.classification_instructions:
                    enhanced_instructions.append("Preferences:")
                    for key, value in request.classification_instructions["preferences"].items():
                        enhanced_instructions.append(f"- {key}: {value}")
            
            # Combine all instructions
            final_instructions = classification_instructions
            if enhanced_instructions:
                final_instructions += "\n\n" + "\n".join(enhanced_instructions)
            if request.custom_instructions:
                final_instructions += f"\n\nAdditional Instructions:\n{request.custom_instructions}"
            
            # Build the user prompt
            prompt = PromptTemplates.categorization(
                content_text,
                [f"{cat.id}: {cat.name}" for cat in request.categories],
                final_instructions
            )
            
            # Append format instructions
            prompt += """
            
            Format your response as a valid JSON object with this structure:
            {
                "classifications": [
                    {
                        "category_id": "category_id",
                        "category_name": "Category Name",
                        "confidence": 0.95
                    },
                    ...
                ]
            }
            """
            
            try:
                # Generate the classification
                result = await llm_client.generate(
                    prompt=prompt,
                    system_prompt=f"""
                    You are classifying tabular data from column '{column_to_classify}'.
                    
                    {final_instructions}
                    
                    For each assigned category, provide a confidence score between 0.0 and 1.0.
                    Only assign categories with confidence above {request.confidence_threshold}.
                    """,
                    temperature=0.1,
                    max_tokens=500
                )
                
                # Parse the result
                try:
                    # Find JSON in the response
                    json_start = result.find("{")
                    json_end = result.rfind("}") + 1
                    if json_start >= 0 and json_end > json_start:
                        json_str = result[json_start:json_end]
                        classification_data = json.loads(json_str)
                    else:
                        classification_data = json.loads(result)
                    
                    classifications = classification_data.get("classifications", [])
                    
                    # Filter classifications by confidence threshold
                    filtered_classifications = [
                        cls for cls in classifications 
                        if cls["confidence"] >= request.confidence_threshold
                    ]
                    
                    # Sort by confidence (highest first)
                    filtered_classifications.sort(key=lambda x: x["confidence"], reverse=True)
                    
                    # Limit to max_categories if specified
                    if request.max_categories:
                        filtered_classifications = filtered_classifications[:request.max_categories]
                    
                    result = {
                        "id": item["id"],
                        "content": content_text,
                        "classifications": filtered_classifications,
                        "success": True
                    }
                    results.append(result)
                    
                    # Update original data if requested
                    if return_original_format:
                        if filtered_classifications:
                            original_data[i].update({
                                "category_id": filtered_classifications[0]["category_id"],
                                "category_name": filtered_classifications[0]["category_name"],
                                "confidence": filtered_classifications[0]["confidence"],
                                "success": True
                            })
                        else:
                            original_data[i].update({
                                "category_id": None,
                                "category_name": None,
                                "confidence": None,
                                "success": True
                            })
                    
                except json.JSONDecodeError as e:
                    result = {
                        "id": item["id"],
                        "content": content_text,
                        "error": f"Error parsing classification result: {str(e)}",
                        "success": False
                    }
                    results.append(result)
                    if return_original_format:
                        original_data[i].update({
                            "category_id": None,
                            "category_name": None,
                            "confidence": None,
                            "success": False,
                            "error": f"Error parsing classification result: {str(e)}"
                        })
                    
            except Exception as e:
                result = {
                    "id": item["id"],
                    "content": content_text,
                    "error": f"Error generating classification: {str(e)}",
                    "success": False
                }
                results.append(result)
                if return_original_format:
                    original_data[i].update({
                        "category_id": None,
                        "category_name": None,
                        "confidence": None,
                        "success": False,
                        "error": f"Error generating classification: {str(e)}"
                    })
        
        # Create a summary
        successful = sum(1 for result in results if result.get("success", False))
        failed = len(results) - successful
        
        summary = {
            "total_rows": len(data),
            "successful_classifications": successful,
            "failed_classifications": failed
        }
        
        # Create the final response
        response_data = {
            "request_id": request_id,
            "results": results,
            "summary": summary
        }
        
        # Add original data if requested
        if return_original_format:
            response_data["original_data"] = original_data
        
        # Handle CSV response format
        if response_format.lower() == "csv":
            if not return_original_format:
                raise ValueError("CSV response format requires return_original_format to be true")
            
            # Create CSV in memory
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=original_data[0].keys())
            writer.writeheader()
            writer.writerows(original_data)
            
            # Create response
            response = StreamingResponse(
                iter([output.getvalue()]),
                media_type="text/csv",
                headers={
                    "Content-Disposition": f"attachment; filename=classified_{file.filename}"
                }
            )
            return response
        
        # Return JSON response
        return TabularClassificationResponse(**response_data)
        
    except Exception as e:
        logger.error(f"Error in tabular classification: {str(e)}", request_id)
        raise HTTPException(status_code=500, detail=f"Error in tabular classification: {str(e)}")