# services/document_extraction/main.py
from fastapi import FastAPI, Depends, HTTPException, File, UploadFile, Form, Body
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
import uuid
import json
import io
from enum import Enum

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

# Create app
app = FastAPI(
    title="Document Extraction Service",
    description="API for extracting structured information from various document types"
)

# Setup logger
logger = StructuredLogger("document_extraction_service")

# Models
class DocumentType(str, Enum):
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
    
class OutputFormat(str, Enum):
    JSON = "json"
    CSV = "csv"
    XML = "xml"
    YAML = "yaml"
    TEXT = "text"
    MARKDOWN = "markdown"
    
class ExtractionTemplate(str, Enum):
    INVOICE = "invoice"
    RESUME = "resume"
    RECEIPT = "receipt"
    CONTRACT = "contract"
    REPORT = "report"
    CUSTOM = "custom"
    
class CustomField(BaseModel):
    name: str = Field(..., description="Name of the field to extract")
    description: Optional[str] = Field(None, description="Description of the field")
    type: str = Field("string", description="Data type of the field (string, number, date, etc.)")
    required: Optional[bool] = Field(False, description="Whether the field is required")
    validation_regex: Optional[str] = Field(None, description="Regular expression for validation")
    
class ExtractionRequest(BaseModel):
    items_to_extract: List[str] = Field(..., description="List of items to extract from the document")
    output_format: OutputFormat = Field(OutputFormat.JSON, description="Format of the output")
    template: Optional[ExtractionTemplate] = Field(None, description="Predefined extraction template")
    custom_fields: Optional[List[CustomField]] = Field(None, description="Custom fields to extract (when template is CUSTOM)")
    additional_instructions: Optional[str] = Field(None, description="Additional instructions for extraction")
    
class ExtractionResponse(BaseModel):
    request_id: str = Field(..., description="Unique identifier for the request")
    extracted_data: Union[Dict[str, Any], str] = Field(..., description="Extracted data in the requested format")
    confidence_scores: Optional[Dict[str, float]] = Field(None, description="Confidence scores for each extracted item")
    missing_items: Optional[List[str]] = Field(None, description="Items that could not be extracted")
    warnings: Optional[List[str]] = Field(None, description="Warnings or issues encountered during extraction")

class BatchExtractionResponse(BaseModel):
    request_id: str = Field(..., description="Unique identifier for the request")
    results: List[ExtractionResponse] = Field(..., description="Extraction results for each document")
    summary: Optional[Dict[str, Any]] = Field(None, description="Summary of the batch extraction")

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
@app.post("/extract", response_model=ExtractionResponse)
async def extract_from_document(
    file: UploadFile = File(...),
    extraction_request: str = Form(...),  # JSON string of ExtractionRequest
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Extract structured information from a document."""
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
        system_prompt = f"""
        You are an expert document analyzer specialized in extracting structured information.
        Extract the requested information from the document accurately and precisely.
        Format the output exactly as specified.
        Maintain the original text where appropriate, but normalize values like dates, numbers, and currencies.
        If an item cannot be found, indicate that it's missing.
        """
        
        # Use the document extraction prompt template
        output_format_instruction = f"Return the extracted information as a valid {request.output_format} object."
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

@app.post("/batch-extract", response_model=BatchExtractionResponse)
async def batch_extract(
    files: List[UploadFile] = File(...),
    extraction_request: str = Form(...),  # JSON string of ExtractionRequest
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Batch extract structured information from multiple documents."""
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
                output_format_instruction = f"Return the extracted information as a valid {request.output_format} object."
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
                    system_prompt=f"You are an expert document analyzer extracting information from {file.filename}.",
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

@app.post("/extract-from-text", response_model=ExtractionResponse)
async def extract_from_text(
    text: str = Body(..., embed=True),
    request: ExtractionRequest = Body(...),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Extract structured information from plain text."""
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
        output_format_instruction = f"Return the extracted information as a valid {request.output_format} object."
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
            system_prompt="You are an expert text analyzer extracting structured information.",
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