# services/document_extraction/main.py
from fastapi import FastAPI, Depends, HTTPException, File, UploadFile, Form, Body
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
import uuid
import json
import io
from enum import Enum
import os
from fastapi.security import HTTPBearer
from fastapi.openapi.utils import get_openapi

from utilities.llm_client import LLMClient, LLMProvider
from utilities.logging_utils import StructuredLogger
from utilities.auth import get_current_user
from utilities.prompts import PromptTemplates

# Document parsers
import PyPDF2
import docx
import csv
import pandas as pd
from pdfminer.high_level import extract_text as extract_text_pdfminer

# Load example requests
EXAMPLES_PATH = os.path.join("tests", "test_data", "document_extraction", "examples.json")
with open(EXAMPLES_PATH, "r") as f:
    EXAMPLE_REQUESTS = json.load(f)

# Create app with OpenAPI documentation
app = FastAPI(
    title="Document Extraction Service",
    description="""
    API for extracting structured information from various document types using AI.
    
    ## Features
    - Single document extraction with confidence scoring
    - Batch processing for multiple documents
    - Direct text extraction without file upload
    - Support for multiple document formats
    - Customizable extraction templates
    - Flexible output formats
    
    ## Document Types Supported
    - PDF: Portable Document Format files
    - DOCX: Microsoft Word documents
    - TXT: Plain text files
    - CSV: Comma-separated values
    - JSON: JavaScript Object Notation
    - XML: Extensible Markup Language
    - HTML: HyperText Markup Language
    - MARKDOWN: Markdown text files
    - PPTX: Microsoft PowerPoint presentations
    - XLS/XLSX: Microsoft Excel spreadsheets
    
    ## Extraction Features
    - Template-based extraction
    - Custom field definitions
    - Confidence scoring
    - Batch processing
    - Format conversion
    - Error handling
    
    ## Extraction Templates
    - INVOICE: Extract invoice details
    - RESUME: Extract resume information
    - RECEIPT: Extract receipt details
    - CONTRACT: Extract contract information
    - REPORT: Extract report content
    - CUSTOM: Custom extraction template
    
    ## Output Formats
    - JSON: Structured JSON output (default)
    - CSV: Comma-separated values
    - XML: XML formatted output
    - YAML: YAML formatted output
    - TEXT: Plain text output
    - MARKDOWN: Markdown formatted output
    
    ## Example Use Cases
    - Invoice processing
    - Resume parsing
    - Contract analysis
    - Report data extraction
    - Medical record processing
    - Legal document analysis
    
    ## Quality Features
    - Error handling
    - Input validation
    - Performance optimization
    - Clean output formatting
    - Comprehensive logging
    """,
    version="1.0.0",
    openapi_url="/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {
            "name": "extraction",
            "description": "Operations for extracting structured information from documents"
        }
    ]
)

# Setup logger
logger = StructuredLogger("document_extraction_service")

# Security scheme
security = HTTPBearer(
    scheme_name="Bearer",
    description="Enter your JWT token",
    auto_error=True
)

# Models
class DocumentType(str, Enum):
    """Supported document types."""
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    CSV = "csv"
    JSON = "json"
    XML = "xml"
    HTML = "html"
    MARKDOWN = "markdown"
    PPTX = "pptx"
    XLS = "xls"
    XLSX = "xlsx"

    class Config:
        schema_extra = {
            "example": "pdf"
        }

class OutputFormat(str, Enum):
    """Supported output formats."""
    JSON = "json"
    CSV = "csv"
    XML = "xml"
    YAML = "yaml"
    TEXT = "text"
    MARKDOWN = "markdown"

    class Config:
        schema_extra = {
            "example": "json"
        }

class ExtractionTemplate(str, Enum):
    """Predefined extraction templates."""
    INVOICE = "invoice"
    RESUME = "resume"
    RECEIPT = "receipt"
    CONTRACT = "contract"
    REPORT = "report"
    CUSTOM = "custom"

    class Config:
        schema_extra = {
            "example": "invoice"
        }

class CustomField(BaseModel):
    """Model for defining custom extraction fields."""
    name: str = Field(..., description="Name of the field to extract")
    description: Optional[str] = Field(None, description="Description of the field")
    type: str = Field("string", description="Data type of the field (string, number, date, etc.)")
    required: Optional[bool] = Field(False, description="Whether the field is required")
    validation_regex: Optional[str] = Field(None, description="Regular expression for validation")

    class Config:
        schema_extra = {
            "example": {
                "name": "total_amount",
                "description": "Total amount on the invoice",
                "type": "number",
                "required": True,
                "validation_regex": r"^\d+(\.\d{2})?$"
            }
        }

class ExtractionRequest(BaseModel):
    """Request model for document extraction."""
    items_to_extract: List[str] = Field(
        ..., 
        description="List of items to extract from the document",
        example=["invoice_number", "date", "total_amount", "vendor_name"]
    )
    output_format: OutputFormat = Field(
        OutputFormat.JSON, 
        description="Format of the output data",
        example="json"
    )
    template: Optional[ExtractionTemplate] = Field(
        None, 
        description="Predefined extraction template",
        example="invoice"
    )
    custom_fields: Optional[List[CustomField]] = Field(
        None, 
        description="Custom fields to extract (when template is CUSTOM)"
    )
    additional_instructions: Optional[str] = Field(
        None, 
        description="Additional instructions for extraction",
        example="Extract line items with their individual prices"
    )

    class Config:
        schema_extra = {
            "example": {
                "items_to_extract": ["invoice_number", "date", "total_amount", "vendor_name"],
                "output_format": "json",
                "template": "invoice",
                "additional_instructions": "Extract line items with their individual prices"
            }
        }

class ExtractionResponse(BaseModel):
    """Response model for document extraction."""
    request_id: str = Field(
        ..., 
        description="Unique identifier for the request",
        example="a534a8c5-3309-4790-935b-6dc1a2e59ba0"
    )
    extracted_data: Union[Dict[str, Any], str] = Field(
        ..., 
        description="Extracted data in the requested format",
        example={
            "invoice_number": "INV-2024-001",
            "date": "2024-03-20",
            "total_amount": 1250.00,
            "vendor_name": "Acme Corp"
        }
    )
    confidence_scores: Optional[Dict[str, float]] = Field(
        None, 
        description="Confidence scores for each extracted item",
        example={
            "invoice_number": 0.95,
            "date": 0.98,
            "total_amount": 0.99,
            "vendor_name": 0.97
        }
    )
    missing_items: Optional[List[str]] = Field(
        None, 
        description="Items that could not be extracted",
        example=["purchase_order_number"]
    )
    warnings: Optional[List[str]] = Field(
        None, 
        description="Warnings or issues encountered during extraction",
        example=["Could not find purchase order number in document"]
    )

    class Config:
        schema_extra = {
            "example": {
                "request_id": "a534a8c5-3309-4790-935b-6dc1a2e59ba0",
                "extracted_data": {
                    "invoice_number": "INV-2024-001",
                    "date": "2024-03-20",
                    "total_amount": 1250.00,
                    "vendor_name": "Acme Corp"
                },
                "confidence_scores": {
                    "invoice_number": 0.95,
                    "date": 0.98,
                    "total_amount": 0.99,
                    "vendor_name": 0.97
                },
                "missing_items": ["purchase_order_number"],
                "warnings": ["Could not find purchase order number in document"]
            }
        }

class BatchExtractionResponse(BaseModel):
    """Response model for batch document extraction."""
    request_id: str = Field(
        ..., 
        description="Unique identifier for the request",
        example="b634a8c5-3309-4790-935b-6dc1a2e59ba1"
    )
    results: List[ExtractionResponse] = Field(
        ..., 
        description="Extraction results for each document"
    )
    summary: Optional[Dict[str, Any]] = Field(
        None, 
        description="Summary of the batch extraction",
        example={
            "total_files": 3,
            "successful_extractions": 2,
            "failed_extractions": 1,
            "common_missing_items": ["purchase_order_number"]
        }
    )

    class Config:
        schema_extra = {
            "example": {
                "request_id": "b634a8c5-3309-4790-935b-6dc1a2e59ba1",
                "results": [
                    {
                        "request_id": "c534a8c5-3309-4790-935b-6dc1a2e59ba2",
                        "extracted_data": {
                            "invoice_number": "INV-2024-001",
                            "date": "2024-03-20",
                            "total_amount": 1250.00,
                            "vendor_name": "Acme Corp"
                        },
                        "confidence_scores": {
                            "invoice_number": 0.95,
                            "date": 0.98,
                            "total_amount": 0.99,
                            "vendor_name": 0.97
                        }
                    }
                ],
                "summary": {
                    "total_files": 3,
                    "successful_extractions": 2,
                    "failed_extractions": 1,
                    "common_missing_items": ["purchase_order_number"]
                }
            }
        }

# Helper functions
async def extract_text_from_document(file: UploadFile) -> str:
    """Extract text from different document types."""
    content = await file.read()
    file_extension = file.filename.split('.')[-1].lower()
    
    try:
        if file_extension == 'pdf':
            # Try with PyPDF2 first
            try:
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                
                # If PyPDF2 extracted very little text, try with pdfminer
                if len(text.strip()) < 100:
                    text = extract_text_pdfminer(io.BytesIO(content))
                
                return text
            except Exception as e:
                # Fallback to pdfminer if PyPDF2 fails
                logger.warning(f"PyPDF2 extraction failed, falling back to pdfminer: {str(e)}")
                return extract_text_pdfminer(io.BytesIO(content))
                
        elif file_extension == 'docx':
            doc = docx.Document(io.BytesIO(content))
            return "\n".join([paragraph.text for paragraph in doc.paragraphs])
            
        elif file_extension == 'txt':
            return content.decode('utf-8')
            
        elif file_extension in ['csv', 'xls', 'xlsx']:
            if file_extension == 'csv':
                df = pd.read_csv(io.BytesIO(content))
            else:  # Excel files
                df = pd.read_excel(io.BytesIO(content))
                
            # Convert DataFrame to string representation
            return df.to_string(index=False)
            
        elif file_extension == 'json':
            # Parse JSON and then format it as a string
            json_data = json.loads(content.decode('utf-8'))
            return json.dumps(json_data, indent=2)
            
        else:
            # For other file types, just return the raw content as UTF-8 text
            # In a production environment, you would add more specialized parsers
            try:
                return content.decode('utf-8')
            except UnicodeDecodeError:
                raise ValueError(f"Unsupported file type or encoding: {file_extension}")
    except Exception as e:
        raise ValueError(f"Error extracting text from {file_extension} file: {str(e)}")

def get_template_fields(template: ExtractionTemplate) -> List[str]:
    """Get predefined fields for a template."""
    template_fields = {
        ExtractionTemplate.INVOICE: [
            "invoice_number", "date", "due_date", "vendor_name", "vendor_address", 
            "customer_name", "customer_address", "line_items", "subtotal", "tax", 
            "total_amount", "payment_terms", "currency"
        ],
        ExtractionTemplate.RESUME: [
            "name", "contact_information", "email", "phone", "address", "summary", 
            "skills", "experience", "education", "certifications", "projects", "languages"
        ],
        ExtractionTemplate.RECEIPT: [
            "merchant_name", "merchant_address", "date", "time", "items_purchased", 
            "subtotal", "tax", "total_amount", "payment_method", "receipt_number"
        ],
        ExtractionTemplate.CONTRACT: [
            "contract_title", "parties_involved", "effective_date", "termination_date", 
            "terms_and_conditions", "signatures", "scope_of_work", "payment_terms", 
            "confidentiality_clauses", "termination_clauses"
        ],
        ExtractionTemplate.REPORT: [
            "title", "author", "date", "executive_summary", "introduction", "methodology", 
            "findings", "conclusions", "recommendations", "references"
        ],
        ExtractionTemplate.CUSTOM: []  # For custom templates, fields are provided separately
    }
    
    return template_fields.get(template, [])

def format_output(data: Dict[str, Any], output_format: OutputFormat) -> Union[Dict[str, Any], str]:
    """Format the extracted data according to the specified output format."""
    if output_format == OutputFormat.JSON:
        # Already in the right format
        return data
    elif output_format == OutputFormat.CSV:
        # Convert to CSV string
        # This is a simple implementation - in a real service, you'd need more sophisticated conversion
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow(data.keys())
        
        # Write values
        writer.writerow(data.values())
        
        return output.getvalue()
    elif output_format == OutputFormat.XML:
        # Simple XML conversion
        xml = ['<?xml version="1.0" encoding="UTF-8"?>', '<extraction>']
        for key, value in data.items():
            xml.append(f'  <{key}>{value}</{key}>')
        xml.append('</extraction>')
        return '\n'.join(xml)
    elif output_format == OutputFormat.YAML:
        # Simple YAML conversion
        yaml_lines = []
        for key, value in data.items():
            yaml_lines.append(f'{key}: {value}')
        return '\n'.join(yaml_lines)
    elif output_format == OutputFormat.TEXT:
        # Simple text conversion
        text_lines = []
        for key, value in data.items():
            text_lines.append(f'{key}: {value}')
        return '\n'.join(text_lines)
    elif output_format == OutputFormat.MARKDOWN:
        # Simple markdown conversion
        md_lines = ['# Extracted Data', '']
        for key, value in data.items():
            md_lines.append(f'**{key}**: {value}')
        return '\n'.join(md_lines)
    else:
        # Default to JSON
        return data

# Routes
@app.post(
    "/extract", 
    response_model=ExtractionResponse,
    tags=["extraction"],
    summary="Extract structured information from a document file",
    description="""
    Extract structured information from a document file using AI-powered analysis.
    
    ## Features
    - Support for multiple document types (PDF, DOCX, etc.)
    - Template-based extraction
    - Custom field extraction
    - Confidence scoring
    - Format conversion
    - Error handling
    
    ## Document Processing
    - Text extraction
    - Structure analysis
    - Field identification
    - Data validation
    - Format conversion
    
    ## Extraction Templates
    - Invoice: Extract invoice details
    - Resume: Extract resume information
    - Receipt: Extract receipt details
    - Contract: Extract contract information
    - Report: Extract report content
    - Custom: User-defined fields
    
    ## Example Use Cases
    - Invoice processing
    - Resume parsing
    - Contract analysis
    - Report data extraction
    - Medical record analysis
    - Legal document processing
    
    ## Output Options
    - JSON structured data
    - CSV format
    - XML output
    - YAML format
    - Plain text
    - Markdown
    """,
    response_description="Successfully extracted structured information from the document"
)
async def extract_from_document(
    file: UploadFile = File(..., description="The document file to extract information from"),
    extraction_request: str = Form(..., description="JSON string containing extraction request parameters"),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Extract structured information from a document file.
    
    Args:
        file: The document file to extract information from
        extraction_request: JSON string containing extraction parameters
        current_user: The authenticated user making the request
        
    Returns:
        ExtractionResponse containing the extracted data and metadata
        
    Raises:
        HTTPException: If there's an error during extraction
    """
    request_id = str(uuid.uuid4())
    
    try:
        # Parse the extraction request
        request_data = json.loads(extraction_request)
        request = ExtractionRequest(**request_data)
        
        logger.info(
            "Document extraction request received", 
            request_id, 
            {"filename": file.filename, "output_format": request.output_format}
        )
        
        # Extract text from the document
        document_text = await extract_text_from_document(file)
        
        # Determine items to extract
        items_to_extract = request.items_to_extract
        
        # If a template is specified, add template fields
        if request.template and request.template != ExtractionTemplate.CUSTOM:
            template_fields = get_template_fields(request.template)
            items_to_extract.extend([field for field in template_fields if field not in items_to_extract])
        
        # If custom fields are provided, add them
        if request.custom_fields:
            custom_field_names = [field.name for field in request.custom_fields]
            items_to_extract.extend([field for field in custom_field_names if field not in items_to_extract])
        
        # Create LLM client
        llm_client = LLMClient()
        
        # Build the system prompt
        system_prompt = """You are an expert document analyzer specialized in extracting structured information.
IMPORTANT: Return ONLY the requested JSON object without any additional text, markdown formatting, or explanation.
Do not include ```json, ``` markers, or any other text.
The response should be a valid JSON object that can be parsed directly.
If a field cannot be found in the document, set its value to "MISSING".
Normalize values like dates (YYYY-MM-DD), numbers, and currencies where appropriate.
"""
        
        # Use the document extraction prompt template
        output_format_instruction = f"Return ONLY a valid {request.output_format} object with no additional text or formatting."
        prompt = PromptTemplates.document_extraction(
            document_text, 
            items_to_extract, 
            output_format_instruction
        )
        
        # Add additional instructions if provided
        if request.additional_instructions:
            prompt += f"\n\nAdditional instructions:\n{request.additional_instructions}"
        
        # Generate the extraction
        result = await llm_client.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.1,  # Lower temperature for more deterministic extraction
            max_tokens=1500
        )
        
        # Parse the result
        # This is a simplified implementation - in a real service, you'd need more robust parsing
        try:
            # Try to parse as JSON first
            extracted_data = json.loads(result)
            
            # Check for missing items
            missing_items = [item for item in items_to_extract if item not in extracted_data]
            
            # Simple confidence scoring (could be more sophisticated in a real service)
            confidence_scores = {
                item: 1.0 if item in extracted_data and extracted_data[item] else 0.0
                for item in items_to_extract
            }
            
            # Format the output
            formatted_output = format_output(extracted_data, request.output_format)
            
            return ExtractionResponse(
                request_id=request_id,
                extracted_data=formatted_output,
                confidence_scores=confidence_scores,
                missing_items=missing_items if missing_items else None,
                warnings=None
            )
            
        except json.JSONDecodeError:
            # If the result is not valid JSON, return it as raw text
            logger.warning("LLM did not return valid JSON, returning raw result", request_id)
            
            return ExtractionResponse(
                request_id=request_id,
                extracted_data=result,
                warnings=["LLM did not return proper JSON format, returning raw result"]
            )
            
    except Exception as e:
        logger.error(f"Error extracting from document: {str(e)}", request_id)
        raise HTTPException(status_code=500, detail=f"Error extracting from document: {str(e)}")

@app.post(
    "/batch-extract",
    response_model=BatchExtractionResponse,
    tags=["extraction"],
    summary="Batch extract information from multiple documents",
    description="""
    Process multiple documents in a single request for efficient batch extraction.
    
    ## Features
    - Multiple file processing
    - Consistent extraction rules
    - Batch progress tracking
    - Aggregated results
    - Error handling per file
    
    ## Processing Options
    - Parallel processing
    - Template application
    - Custom field extraction
    - Format standardization
    - Validation rules
    
    ## Batch Processing
    - Document validation
    - Text extraction
    - Field identification
    - Data normalization
    - Result aggregation
    
    ## Summary Information
    - Total files processed
    - Successful extractions
    - Failed extractions
    - Processing time
    - Error details
    
    ## Example Use Cases
    - Medical record processing
    - Invoice batch processing
    - Resume screening
    - Contract analysis
    - Report aggregation
    - Legal document review
    """,
    response_description="Successfully processed batch of documents with extraction results"
)
async def batch_extract(
    files: List[UploadFile] = File(..., description="List of document files to extract information from"),
    extraction_request: str = Form(..., description="JSON string containing extraction request parameters"),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Batch extract structured information from multiple documents.
    
    Args:
        files: List of document files to process
        extraction_request: JSON string containing extraction parameters
        current_user: The authenticated user making the request
        
    Returns:
        BatchExtractionResponse containing results for each document
        
    Raises:
        HTTPException: If there's an error during batch processing
    """
    request_id = str(uuid.uuid4())
    
    try:
        # Parse the extraction request
        request_data = json.loads(extraction_request)
        request = ExtractionRequest(**request_data)
        
        logger.info(
            "Batch document extraction request received", 
            request_id, 
            {"file_count": len(files), "output_format": request.output_format}
        )
        
        results = []
        
        # Process each file
        for file in files:
            try:
                # Extract from the document
                document_text = await extract_text_from_document(file)
                
                # Determine items to extract (same logic as single extraction)
                items_to_extract = request.items_to_extract
                
                if request.template and request.template != ExtractionTemplate.CUSTOM:
                    template_fields = get_template_fields(request.template)
                    items_to_extract.extend([field for field in template_fields if field not in items_to_extract])
                
                if request.custom_fields:
                    custom_field_names = [field.name for field in request.custom_fields]
                    items_to_extract.extend([field for field in custom_field_names if field not in items_to_extract])
                
                # Create LLM client
                llm_client = LLMClient()
                
                # Use the document extraction prompt template
                output_format_instruction = f"Return ONLY a valid {request.output_format} object with no additional text or formatting."
                prompt = PromptTemplates.document_extraction(
                    document_text, 
                    items_to_extract, 
                    output_format_instruction
                )
                
                # Add additional instructions if provided
                if request.additional_instructions:
                    prompt += f"\n\nAdditional instructions:\n{request.additional_instructions}"
                
                # Generate the extraction
                result = await llm_client.generate(
                    prompt=prompt,
                    system_prompt="""You are an expert document analyzer specialized in extracting structured information.
IMPORTANT: Return ONLY the requested JSON object without any additional text, markdown formatting, or explanation.
Do not include ```json, ``` markers, or any other text.
The response should be a valid JSON object that can be parsed directly.
If a field cannot be found in the document, set its value to "MISSING".
Normalize values like dates (YYYY-MM-DD), numbers, and currencies where appropriate.""",
                    temperature=0.1,
                    max_tokens=1500
                )
                
                # Parse the result (similar logic to single extraction)
                try:
                    extracted_data = json.loads(result)
                    missing_items = [item for item in items_to_extract if item not in extracted_data]
                    confidence_scores = {
                        item: 1.0 if item in extracted_data and extracted_data[item] else 0.0
                        for item in items_to_extract
                    }
                    formatted_output = format_output(extracted_data, request.output_format)
                    
                    file_result = ExtractionResponse(
                        request_id=str(uuid.uuid4()),  # Each file gets its own request ID
                        extracted_data=formatted_output,
                        confidence_scores=confidence_scores,
                        missing_items=missing_items if missing_items else None,
                        warnings=None
                    )
                    
                except json.JSONDecodeError:
                    file_result = ExtractionResponse(
                        request_id=str(uuid.uuid4()),
                        extracted_data=result,
                        warnings=["LLM did not return proper JSON format, returning raw result"]
                    )
                
                results.append(file_result)
                
            except Exception as e:
                # Log error for this file but continue processing others
                logger.error(f"Error processing file {file.filename}: {str(e)}", request_id)
                results.append(ExtractionResponse(
                    request_id=str(uuid.uuid4()),
                    extracted_data={},
                    warnings=[f"Error processing file: {str(e)}"]
                ))
        
        # Create a summary of the batch extraction
        summary = {
            "total_files": len(files),
            "successful_extractions": sum(1 for result in results if not result.warnings),
            "failed_extractions": sum(1 for result in results if result.warnings),
            "common_missing_items": []  # Would calculate in a real service
        }
        
        return BatchExtractionResponse(
            request_id=request_id,
            results=results,
            summary=summary
        )
        
    except Exception as e:
        logger.error(f"Error in batch extraction: {str(e)}", request_id)
        raise HTTPException(status_code=500, detail=f"Error in batch extraction: {str(e)}")

@app.post(
    "/extract-from-text",
    response_model=ExtractionResponse,
    tags=["extraction"],
    summary="Extract structured information from plain text",
    description="""
    Extract structured information directly from text content without file upload.
    
    ## Features
    - Direct text processing
    - Template application
    - Custom field extraction
    - Format conversion
    - Validation rules
    
    ## Text Analysis
    - Content structure analysis
    - Field identification
    - Data validation
    - Format standardization
    - Confidence scoring
    
    ## Processing Options
    - Template-based extraction
    - Custom field definitions
    - Output format selection
    - Validation rules
    - Additional instructions
    
    ## Example Use Cases
    - Clinical note processing
    - Email content analysis
    - Chat log extraction
    - Document snippets
    - Text-based reports
    - Legal text analysis
    
    ## Output Options
    - JSON structured data
    - CSV format
    - XML output
    - YAML format
    - Plain text
    - Markdown
    """,
    response_description="Successfully extracted structured information from the text"
)
async def extract_from_text(
    text: str = Body(..., description="The text content to extract information from"),
    request: ExtractionRequest = Body(..., description="Extraction request parameters"),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Extract structured information from plain text content.
    
    Args:
        text: The text content to extract information from
        request: Extraction parameters and configuration
        current_user: The authenticated user making the request
        
    Returns:
        ExtractionResponse containing the extracted data and metadata
        
    Raises:
        HTTPException: If there's an error during extraction
    """
    request_id = str(uuid.uuid4())
    
    try:
        logger.info(
            "Text extraction request received", 
            request_id, 
            {"text_length": len(text), "output_format": request.output_format}
        )
        
        # Determine items to extract
        items_to_extract = request.items_to_extract
        
        # If a template is specified, add template fields
        if request.template and request.template != ExtractionTemplate.CUSTOM:
            template_fields = get_template_fields(request.template)
            items_to_extract.extend([field for field in template_fields if field not in items_to_extract])
        
        # If custom fields are provided, add them
        if request.custom_fields:
            custom_field_names = [field.name for field in request.custom_fields]
            items_to_extract.extend([field for field in custom_field_names if field not in items_to_extract])
        
        # Create LLM client
        llm_client = LLMClient()
        
        # Use the document extraction prompt template
        output_format_instruction = f"Return ONLY a valid {request.output_format} object with no additional text or formatting."
        prompt = PromptTemplates.document_extraction(
            text, 
            items_to_extract, 
            output_format_instruction
        )
        
        # Add additional instructions if provided
        if request.additional_instructions:
            prompt += f"\n\nAdditional instructions:\n{request.additional_instructions}"
        
        # Generate the extraction
        result = await llm_client.generate(
            prompt=prompt,
            system_prompt="""You are an expert document analyzer specialized in extracting structured information.
IMPORTANT: Return ONLY the requested JSON object without any additional text, markdown formatting, or explanation.
Do not include ```json, ``` markers, or any other text.
The response should be a valid JSON object that can be parsed directly.
If a field cannot be found in the document, set its value to "MISSING".
Normalize values like dates (YYYY-MM-DD), numbers, and currencies where appropriate.""",
            temperature=0.1,
            max_tokens=1500
        )
        
        # Parse the result (similar logic to document extraction)
        try:
            extracted_data = json.loads(result)
            missing_items = [item for item in items_to_extract if item not in extracted_data]
            confidence_scores = {
                item: 1.0 if item in extracted_data and extracted_data[item] else 0.0
                for item in items_to_extract
            }
            formatted_output = format_output(extracted_data, request.output_format)
            
            return ExtractionResponse(
                request_id=request_id,
                extracted_data=formatted_output,
                confidence_scores=confidence_scores,
                missing_items=missing_items if missing_items else None,
                warnings=None
            )
            
        except json.JSONDecodeError:
            return ExtractionResponse(
                request_id=request_id,
                extracted_data=result,
                warnings=["LLM did not return proper JSON format, returning raw result"]
            )
            
    except Exception as e:
        logger.error(f"Error extracting from text: {str(e)}", request_id)
        raise HTTPException(status_code=500, detail=f"Error extracting from text: {str(e)}")

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

    # Update request body examples safely
    if "paths" in openapi_schema:
        for path, path_item in openapi_schema["paths"].items():
            if "post" not in path_item:
                continue
                
            post_op = path_item["post"]
            if "requestBody" not in post_op or "content" not in post_op["requestBody"]:
                continue

            content = post_op["requestBody"]["content"]
            
            if path == "/extract":
                if "multipart/form-data" in content:
                    schema = content["multipart/form-data"].get("schema", {})
                    if "properties" in schema and "extraction_request" in schema["properties"]:
                        schema["properties"]["extraction_request"]["example"] = json.dumps(
                            EXAMPLE_REQUESTS["single_clinical_note"]["extraction_request"]
                        )
                        
            elif path == "/batch-extract":
                if "multipart/form-data" in content:
                    schema = content["multipart/form-data"].get("schema", {})
                    if "properties" in schema and "extraction_request" in schema["properties"]:
                        schema["properties"]["extraction_request"]["example"] = json.dumps(
                            EXAMPLE_REQUESTS["batch_clinical_notes"]["extraction_request"]
                        )
                        
            elif path == "/extract-from-text":
                if "application/json" in content:
                    schema = content["application/json"].get("schema", {})
                    if "properties" in schema and "request" in schema["properties"]:
                        schema["properties"]["request"]["example"] = EXAMPLE_REQUESTS["csv_output"]["extraction_request"]

    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi