# services/classification_service/main.py
from fastapi import FastAPI, Depends, HTTPException, File, UploadFile, Form, Body
from fastapi.responses import JSONResponse
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

# Create app
app = FastAPI(
    title="Classification and Categorization Service",
    description="API for classifying and categorizing various types of content"
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
    
class Category(BaseModel):
    id: str = Field(..., description="Unique identifier for the category")
    name: str = Field(..., description="Name of the category")
    description: Optional[str] = Field(None, description="Description of the category")
    parent_id: Optional[str] = Field(None, description="Parent category ID for hierarchical classifications")
    
class ClassificationRequest(BaseModel):
    content: Optional[str] = Field(None, description="Content to classify (for direct text input)")
    content_type: ContentType = Field(..., description="Type of content being classified")
    classification_type: ClassificationType = Field(..., description="Type of classification to perform")
    categories: List[Category] = Field(..., description="Available categories for classification")
    confidence_threshold: Optional[float] = Field(0.5, description="Minimum confidence threshold (0.0-1.0)")
    max_categories: Optional[int] = Field(None, description="Maximum number of categories to assign")
    custom_instructions: Optional[str] = Field(None, description="Additional instructions for classification")
    
class ClassificationResult(BaseModel):
    category_id: str = Field(..., description="ID of the assigned category")
    category_name: str = Field(..., description="Name of the assigned category")
    confidence: float = Field(..., description="Confidence score (0.0-1.0)")
    
class ClassificationResponse(BaseModel):
    request_id: str = Field(..., description="Unique identifier for the request")
    results: List[ClassificationResult] = Field(..., description="Classification results")
    content_excerpt: Optional[str] = Field(None, description="Short excerpt of the classified content")
    processing_time: float = Field(..., description="Processing time in seconds")
    warnings: Optional[List[str]] = Field(None, description="Warnings or issues encountered")

class BatchClassificationResponse(BaseModel):
    request_id: str = Field(..., description="Unique identifier for the request")
    results: List[Dict[str, Any]] = Field(..., description="Classification results for each item")
    summary: Optional[Dict[str, Any]] = Field(None, description="Summary of the batch classification")

# Helper functions
async def extract_text_from_file(file: UploadFile) -> str:
    """Extract text content from various file types."""
    content = await file.read()
    file_extension = file.filename.split('.')[-1].lower()
    
    # Use the same text extraction logic from the document extraction service
    # In a real implementation, this would likely be a shared utility
    
    # For simplicity, just return the content as UTF-8 text
    try:
        return content.decode('utf-8')
    except UnicodeDecodeError:
        raise ValueError(f"Unsupported file type or encoding: {file_extension}")

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
    request: ClassificationRequest = Body(...),
    file: Optional[UploadFile] = File(None),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Classify content according to specified categories."""
    request_id = str(uuid.uuid4())
    start_time = datetime.now()
    
    try:
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
        
        # Add custom instructions if provided
        if request.custom_instructions:
            classification_instructions += f"\n\n{request.custom_instructions}"
        
        # Build the system prompt
        system_prompt = f"""
        You are an expert content classifier specialized in {request.classification_type.value} classification.
        You will analyze {request.content_type.value} content and assign appropriate categories based on the provided classification scheme.
        {classification_instructions}
        For each assigned category, provide a confidence score between 0.0 and 1.0.
        Only assign categories with confidence above {request.confidence_threshold}.
        """
        
        # Build the user prompt using the categorization template
        prompt = PromptTemplates.categorization(
            content_text,
            [f"{cat.id}: {cat.name}" for cat in request.categories],
            classification_instructions
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
            
            # Convert to ClassificationResult objects
            results = []
            for cls in classifications:
                results.append(ClassificationResult(
                    category_id=cls["category_id"],
                    category_name=cls["category_name"],
                    confidence=cls["confidence"]
                ))
            
            # Sort by confidence (highest first)
            results.sort(key=lambda x: x.confidence, reverse=True)
            
            # Limit to max_categories if specified
            if request.max_categories:
                results = results[:request.max_categories]
            
            # Create a short excerpt of the content
            excerpt = content_text[:150] + "..." if len(content_text) > 150 else content_text
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return ClassificationResponse(
                request_id=request_id,
                results=results,
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
    request: ClassificationRequest = Body(...),
    files: List[UploadFile] = File(...),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Batch classify multiple content items according to specified categories."""
    request_id = str(uuid.uuid4())
    start_time = datetime.now()
    
    try:
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
                
                # Use the same classification logic as the single item endpoint
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
                # ... (similar logic for other classification types)
                
                # Add custom instructions if provided
                if request.custom_instructions:
                    classification_instructions += f"\n\n{request.custom_instructions}"
                
                # Build the user prompt
                prompt = PromptTemplates.categorization(
                    content_text,
                    [f"{cat.id}: {cat.name}" for cat in request.categories],
                    classification_instructions
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
                    system_prompt=f"You are classifying content from the file {file.filename}.",
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
                    
                    # Convert to ClassificationResult objects
                    file_results = []
                    for cls in classifications:
                        file_results.append({
                            "category_id": cls["category_id"],
                            "category_name": cls["category_name"],
                            "confidence": cls["confidence"]
                        })
                    
                    # Sort by confidence (highest first)
                    file_results.sort(key=lambda x: x["confidence"], reverse=True)
                    
                    # Limit to max_categories if specified
                    if request.max_categories:
                        file_results = file_results[:request.max_categories]
                    
                    # Create a short excerpt of the content
                    excerpt = content_text[:150] + "..." if len(content_text) > 150 else content_text
                    
                    results.append({
                        "filename": file.filename,
                        "classifications": file_results,
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
        
        summary = {
            "total_files": len(files),
            "successful_classifications": successful,
            "failed_classifications": failed,
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

@app.post("/classify-tabular", response_model=JSONResponse)
async def classify_tabular_data(
    file: UploadFile = File(...),
    request: ClassificationRequest = Body(...),
    column_to_classify: str = Body(...),
    id_column: Optional[str] = Body(None),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Classify data from a specific column in a tabular file (CSV, Excel, etc.)."""
    request_id = str(uuid.uuid4())
    
    try:
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
        column_indices = {}
        
        if file_extension == 'csv':
            # Parse CSV
            text_content = content.decode('utf-8')
            csv_reader = csv.reader(io.StringIO(text_content))
            headers = next(csv_reader)
            
            # Find the column indices
            for i, header in enumerate(headers):
                column_indices[header] = i
            
            if column_to_classify not in column_indices:
                raise ValueError(f"Column '{column_to_classify}' not found in the CSV file")
            
            classify_index = column_indices[column_to_classify]
            id_index = column_indices.get(id_column, None) if id_column else None
            
            # Read the data
            for row in csv_reader:
                if len(row) > classify_index:
                    item = {
                        "content": row[classify_index],
                        "id": row[id_index] if id_index is not None and id_index < len(row) else None
                    }
                    data.append(item)
        
        elif file_extension in ['xls', 'xlsx']:
            # Parse Excel (placeholder - in a real service, you'd use a library like pandas or openpyxl)
            raise NotImplementedError("Excel parsing not implemented in this example")
        
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        # Process each row (in a real service, you might want to batch these)
        results = []
        llm_client = LLMClient()
        
        for item in data:
            content_text = item["content"]
            
            # Skip empty cells
            if not content_text:
                results.append({
                    "id": item["id"],
                    "content": content_text,
                    "classifications": [],
                    "error": "Empty content"
                })
                continue
            
            # Format categories
            categories_formatted = format_categories_for_prompt(request.categories, request.classification_type)
            
            # Build the user prompt
            prompt = PromptTemplates.categorization(
                content_text,
                [f"{cat.id}: {cat.name}" for cat in request.categories],
                request.custom_instructions or ""
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
                    system_prompt=f"You are classifying tabular data from column '{column_to_classify}'.",
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
                    
                    results.append({
                        "id": item["id"],
                        "content": content_text,
                        "classifications": classifications,
                        "success": True
                    })
                    
                except json.JSONDecodeError as e:
                    results.append({
                        "id": item["id"],
                        "content": content_text,
                        "error": f"Error parsing classification result: {str(e)}",
                        "success": False
                    })
                    
            except Exception as e:
                results.append({
                    "id": item["id"],
                    "content": content_text,
                    "error": f"Error generating classification: {str(e)}",
                    "success": False
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
        
        return JSONResponse(content=response_data)
        
    except Exception as e:
        logger.error(f"Error in tabular classification: {str(e)}", request_id)
        raise HTTPException(status_code=500, detail=f"Error in tabular classification: {str(e)}")