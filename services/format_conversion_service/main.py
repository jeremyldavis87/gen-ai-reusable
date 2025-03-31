# services/format_conversion_service/main.py

import os
import json
import yaml
import csv
import io
import uuid
import logging
import asyncio
import re
from typing import Dict, List, Optional, Union, Any
from enum import Enum
from datetime import datetime
from fastapi import FastAPI, Depends, HTTPException, Body, File, UploadFile, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

# Import shared utilities
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utilities.llm_client import LLMClient, LLMProvider
from utilities.text_chunking import TextChunker

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("format_conversion_service")

# Initialize FastAPI app
app = FastAPI(title="Format Conversion Service", 
              description="Service for converting unstructured text into various structured formats")

# Add CORS middleware for development
from fastapi.middleware.cors import CORSMiddleware

# Get CORS settings from environment variables
cors_origins = os.getenv("CORS_ALLOW_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define models
class OutputFormat(str, Enum):
    JSON = "json"
    YAML = "yaml"
    CSV = "csv"
    XML = "xml"
    MARKDOWN = "markdown"
    HTML = "html"
    SQL = "sql"

class ConversionRequest(BaseModel):
    text: str = Field(..., description="The unstructured text to convert")
    instructions: str = Field(..., description="Instructions for how to structure the output")
    output_format: OutputFormat = Field(..., description="The desired output format")
    schema: Optional[Dict[str, Any]] = Field(None, description="Optional schema definition for the output")
    max_tokens: Optional[int] = Field(8000, description="Maximum tokens for LLM response")
    temperature: Optional[float] = Field(0.2, description="Temperature for LLM generation")

class FileConversionRequest(BaseModel):
    instructions: str = Field(..., description="Instructions for how to structure the output")
    output_format: OutputFormat = Field(..., description="The desired output format")
    schema: Optional[Dict[str, Any]] = Field(None, description="Optional schema definition for the output")
    max_tokens: Optional[int] = Field(8000, description="Maximum tokens for LLM response")
    temperature: Optional[float] = Field(0.2, description="Temperature for LLM generation")

class ConversionResponse(BaseModel):
    conversion_id: str = Field(..., description="Unique identifier for this conversion")
    output: str = Field(..., description="The formatted output")
    format: OutputFormat = Field(..., description="The format of the output")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp of the conversion")

class ConversionStatus(BaseModel):
    conversion_id: str
    status: str
    progress: float
    message: Optional[str] = None

class SelfHealRequest(BaseModel):
    text: str = Field(..., description="The text to heal")
    format: OutputFormat = Field(..., description="The format of the text to heal")
    schema: Optional[Dict[str, Any]] = Field(None, description="Optional schema definition for the output")
    max_tokens: Optional[int] = Field(8000, description="Maximum tokens for LLM response")
    temperature: Optional[float] = Field(0.1, description="Temperature for LLM generation")

class SelfHealResponse(BaseModel):
    healed_text: str = Field(..., description="The healed text")
    format: OutputFormat = Field(..., description="The format of the healed text")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp of the healing")

# Authentication dependency (placeholder - implement actual auth)
async def get_current_user():
    # In a real implementation, this would validate the token and return the user
    return {"id": "user123", "name": "Test User"}

# Helper functions
async def format_output(text: str, format_type: OutputFormat, instructions: str, schema: Optional[Dict[str, Any]] = None) -> str:
    """Format the text according to the specified output format."""
    llm_client = LLMClient()
    
    # Check if mock responses are enabled
    if os.getenv("MOCK_LLM_RESPONSES", "false").lower() == "true":
        logger.info("Using mock LLM responses for testing")
        # Construct system prompt based on format type
        system_prompts = {
            OutputFormat.JSON: "You are a precise JSON formatter.",
            OutputFormat.YAML: "You are a precise YAML formatter.",
            OutputFormat.CSV: "You are a precise CSV formatter.",
            OutputFormat.XML: "You are a precise XML formatter.",
            OutputFormat.MARKDOWN: "You are a precise Markdown formatter.",
            OutputFormat.HTML: "You are a precise HTML formatter.",
            OutputFormat.SQL: "You are a precise SQL formatter."
        }
        system_prompt = system_prompts.get(format_type, "You are a precise formatter.")
        return llm_client._generate_mock_response(text, system_prompt)
    
    # If not using mock responses, proceed with normal API calls
    # Construct system prompt based on format type
    system_prompts = {
        OutputFormat.JSON: """You are a precise JSON formatter. Convert the provided text into valid JSON format according to the instructions. Ensure the output is properly structured and valid JSON.

Important JSON formatting rules:
1. Use double quotes (") for all strings, including URLs
2. Do not escape quotes in URLs (e.g., use "https://example.com" not \"https://example.com\")
3. Use proper JSON syntax for all values (null, true, false, numbers, strings, arrays, objects)
4. Ensure all objects and arrays are properly closed
5. Do not include trailing commas
6. Use proper nesting and indentation
7. Ensure all property names are in double quotes
8. Do not include any markdown formatting or code blocks in the output""",
        OutputFormat.YAML: "You are a precise YAML formatter. Convert the provided text into valid YAML format according to the instructions. Ensure the output is properly structured and valid YAML.",
        OutputFormat.CSV: "You are a precise CSV formatter. Convert the provided text into valid CSV format according to the instructions. The first row should contain headers.",
        OutputFormat.XML: "You are a precise XML formatter. Convert the provided text into valid XML format according to the instructions. Ensure the output is properly structured and valid XML.",
        OutputFormat.MARKDOWN: "You are a precise Markdown formatter. Convert the provided text into well-structured Markdown format according to the instructions.",
        OutputFormat.HTML: "You are a precise HTML formatter. Convert the provided text into valid HTML format according to the instructions. Ensure the output is properly structured and valid HTML.",
        OutputFormat.SQL: "You are a precise SQL formatter. Convert the provided text into valid SQL statements according to the instructions. Ensure the output is properly structured and valid SQL."
    }
    
    system_prompt = system_prompts.get(format_type, "You are a precise formatter. Convert the provided text into the specified format according to the instructions.")
    
    # Add schema information if provided
    if schema:
        schema_str = json.dumps(schema, indent=2)
        system_prompt += f"\n\nUse the following schema for the output:\n{schema_str}"
    
    # Construct user prompt
    user_prompt = f"""Convert the following text into {format_type.value} format.

Instructions: {instructions}

Text to convert:
{text}

IMPORTANT: Return ONLY the formatted output without any additional text, explanations, or markdown formatting. The output should be a valid {format_type.value} object that can be parsed directly. Do not include any other text, comments, or explanations in the output."""
    
    # Combine system and user prompts
    combined_prompt = f"{system_prompt}\n\n{user_prompt}"
    
    # Generate the formatted output
    result = await llm_client.generate(
        prompt=combined_prompt,
        temperature=0.2,  # Low temperature for more deterministic output
        max_tokens=8000   # Allow for larger outputs
    )
    
    logger.info(f"LLM Response for {format_type.value}:\n{result}")
    
    # Clean up the result based on format type
    if format_type == OutputFormat.JSON:
        # Extract JSON if it's wrapped in code blocks
        if "```json" in result:
            result = result.split("```json")[1].split("```")[0].strip()
        elif "```" in result:
            result = result.split("```")[1].split("```")[0].strip()
        
        # Validate JSON
        try:
            # First try to parse the JSON
            json_obj = json.loads(result)
            
            # If we have a schema, ensure the output matches the schema structure
            if schema:
                # Ensure the output has the required top-level keys from the schema
                if isinstance(json_obj, dict):
                    for key in schema.keys():
                        if key not in json_obj:
                            json_obj[key] = {}
            
            # Convert back to string with proper formatting
            result = json.dumps(json_obj, indent=2)
        except json.JSONDecodeError as e:
            logger.warning(f"LLM generated invalid JSON: {str(e)}")
            logger.warning(f"Raw output: {result}")
            
            # Try to fix common JSON issues first
            try:
                # Replace single quotes with double quotes, but preserve apostrophes in contractions
                result = re.sub(r"(?<!\w)'", '"', result)
                # Fix missing quotes around property names
                result = re.sub(r'(\w+):', r'"\1":', result)
                # Fix trailing commas
                result = result.replace(',]', ']').replace(',}', '}')
                # Fix double-escaped quotes in URLs
                result = result.replace('""https"', '"https')
                # Fix escaped quotes in URLs
                result = result.replace('\\"https', '"https')
                
                # Try parsing again after regex fixes
                json_obj = json.loads(result)
                result = json.dumps(json_obj, indent=2)
            except json.JSONDecodeError:
                # If regex fixes didn't work, try self-healing with LLM
                try:
                    logger.info("Attempting to self-heal JSON using LLM")
                    result = await self_heal_json(result, schema)
                except Exception as e:
                    logger.error(f"Could not fix invalid JSON: {str(e)}")
                    logger.error(f"Attempted to fix output: {result}")
                    # Return a more informative error
                    raise HTTPException(
                        status_code=500,
                        detail={
                            "error": "Failed to generate valid JSON output",
                            "original_error": str(e),
                            "raw_output": result
                        }
                    )
    
    elif format_type == OutputFormat.YAML:
        # Extract YAML if it's wrapped in code blocks
        if "```yaml" in result:
            result = result.split("```yaml")[1].split("```")[0].strip()
        elif "```" in result:
            result = result.split("```")[1].split("```")[0].strip()
        
        # Validate YAML
        try:
            yaml_obj = yaml.safe_load(result)
            result = yaml.dump(yaml_obj, sort_keys=False, default_flow_style=False)
        except yaml.YAMLError:
            logger.warning("LLM generated invalid YAML, attempting to self-heal")
            try:
                result = await self_heal_yaml(result, schema)
            except Exception as e:
                logger.error(f"Could not fix invalid YAML: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail={
                        "error": "Failed to generate valid YAML output",
                        "original_error": str(e),
                        "raw_output": result
                    }
                )
    
    elif format_type == OutputFormat.CSV:
        # Extract CSV if it's wrapped in code blocks
        if "```csv" in result:
            result = result.split("```csv")[1].split("```")[0].strip()
        elif "```" in result:
            result = result.split("```")[1].split("```")[0].strip()
        
        # Validate CSV
        try:
            csv_reader = csv.reader(io.StringIO(result))
            rows = list(csv_reader)
            if not rows:
                raise ValueError("Empty CSV")
            output = io.StringIO()
            csv_writer = csv.writer(output)
            csv_writer.writerows(rows)
            result = output.getvalue()
        except Exception:
            logger.warning("LLM generated invalid CSV, attempting to self-heal")
            try:
                result = await self_heal_csv(result, schema)
            except Exception as e:
                logger.error(f"Could not fix invalid CSV: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail={
                        "error": "Failed to generate valid CSV output",
                        "original_error": str(e),
                        "raw_output": result
                    }
                )
    
    elif format_type == OutputFormat.XML:
        # Extract XML if it's wrapped in code blocks
        if "```xml" in result:
            result = result.split("```xml")[1].split("```")[0].strip()
        elif "```" in result:
            result = result.split("```")[1].split("```")[0].strip()
        
        # Validate XML
        try:
            if not result.strip().startswith('<?xml'):
                result = '<?xml version="1.0" encoding="UTF-8"?>\n' + result
        except Exception:
            logger.warning("LLM generated invalid XML, attempting to self-heal")
            try:
                result = await self_heal_xml(result, schema)
            except Exception as e:
                logger.error(f"Could not fix invalid XML: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail={
                        "error": "Failed to generate valid XML output",
                        "original_error": str(e),
                        "raw_output": result
                    }
                )
    
    return result

async def self_heal_json(invalid_json: str, schema: Optional[Dict[str, Any]] = None) -> str:
    """Use LLM to fix invalid JSON output."""
    llm_client = LLMClient()
    
    # Construct healing prompt
    healing_prompt = f"""Fix the following invalid JSON to make it valid. The JSON appears to be truncated or contains formatting errors.
Ensure all strings are properly quoted, all objects and arrays are properly closed, and the structure is valid.
If a schema is provided, ensure the output matches the schema structure.

Invalid JSON:
{invalid_json}

{f'Required Schema:\n{json.dumps(schema, indent=2)}' if schema else ''}

Return ONLY the fixed JSON without any explanations or markdown formatting. The output should be a valid JSON object that can be parsed directly."""

    # Generate fixed JSON
    try:
        fixed_json = await llm_client.generate(
            prompt=healing_prompt,
            temperature=0.1,  # Very low temperature for more deterministic output
            max_tokens=8000
        )
        
        logger.info(f"LLM Response for JSON healing:\n{fixed_json}")
        
        # Extract JSON if it's wrapped in code blocks
        if "```json" in fixed_json:
            fixed_json = fixed_json.split("```json")[1].split("```")[0].strip()
        elif "```" in fixed_json:
            fixed_json = fixed_json.split("```")[1].split("```")[0].strip()
        
        # Validate the fixed JSON
        json_obj = json.loads(fixed_json)
        
        # If we have a schema, ensure the output matches the schema structure
        if schema and isinstance(json_obj, dict):
            for key in schema.keys():
                if key not in json_obj:
                    json_obj[key] = {}
        
        return json.dumps(json_obj, indent=2)
    except Exception as e:
        logger.error(f"Failed to self-heal JSON: {str(e)}")
        raise

async def self_heal_yaml(invalid_yaml: str, schema: Optional[Dict[str, Any]] = None) -> str:
    """Use LLM to fix invalid YAML output."""
    llm_client = LLMClient()
    
    # Construct healing prompt
    healing_prompt = f"""Fix the following invalid YAML to make it valid. The YAML appears to be malformed or contains formatting errors.
Ensure proper indentation, all values are properly formatted, and the structure is valid.
If a schema is provided, ensure the output matches the schema structure.

Invalid YAML:
{invalid_yaml}

{f'Required Schema:\n{json.dumps(schema, indent=2)}' if schema else ''}

Return ONLY the fixed YAML without any explanations or markdown formatting. The output should be a valid YAML object that can be parsed directly."""

    # Generate fixed YAML
    try:
        fixed_yaml = await llm_client.generate(
            prompt=healing_prompt,
            temperature=0.1,
            max_tokens=8000
        )
        
        logger.info(f"LLM Response for YAML healing:\n{fixed_yaml}")
        
        # Extract YAML if it's wrapped in code blocks
        if "```yaml" in fixed_yaml:
            fixed_yaml = fixed_yaml.split("```yaml")[1].split("```")[0].strip()
        elif "```" in fixed_yaml:
            fixed_yaml = fixed_yaml.split("```")[1].split("```")[0].strip()
        
        # Validate the fixed YAML
        yaml_obj = yaml.safe_load(fixed_yaml)
        
        # If we have a schema, ensure the output matches the schema structure
        if schema and isinstance(yaml_obj, dict):
            for key in schema.keys():
                if key not in yaml_obj:
                    yaml_obj[key] = {}
        
        return yaml.dump(yaml_obj, sort_keys=False, default_flow_style=False)
    except Exception as e:
        logger.error(f"Failed to self-heal YAML: {str(e)}")
        raise

async def self_heal_csv(invalid_csv: str, schema: Optional[Dict[str, Any]] = None) -> str:
    """Use LLM to fix invalid CSV output."""
    llm_client = LLMClient()
    
    # Construct healing prompt
    healing_prompt = f"""Fix the following invalid CSV to make it valid. The CSV appears to be malformed or contains formatting errors.
Ensure proper comma separation, all values are properly quoted if needed, and the structure is valid.
If a schema is provided, ensure the output matches the schema structure.

Invalid CSV:
{invalid_csv}

{f'Required Schema:\n{json.dumps(schema, indent=2)}' if schema else ''}

Return ONLY the fixed CSV without any explanations or markdown formatting. The output should be a valid CSV that can be parsed directly."""

    # Generate fixed CSV
    try:
        fixed_csv = await llm_client.generate(
            prompt=healing_prompt,
            temperature=0.1,
            max_tokens=8000
        )
        
        logger.info(f"LLM Response for CSV healing:\n{fixed_csv}")
        
        # Extract CSV if it's wrapped in code blocks
        if "```csv" in fixed_csv:
            fixed_csv = fixed_csv.split("```csv")[1].split("```")[0].strip()
        elif "```" in fixed_csv:
            fixed_csv = fixed_csv.split("```")[1].split("```")[0].strip()
        
        # Validate the fixed CSV
        csv_reader = csv.reader(io.StringIO(fixed_csv))
        rows = list(csv_reader)
        if not rows:
            raise ValueError("Empty CSV")
        
        # Write back to CSV to ensure proper formatting
        output = io.StringIO()
        csv_writer = csv.writer(output)
        csv_writer.writerows(rows)
        return output.getvalue()
    except Exception as e:
        logger.error(f"Failed to self-heal CSV: {str(e)}")
        raise

async def self_heal_xml(invalid_xml: str, schema: Optional[Dict[str, Any]] = None) -> str:
    """Use LLM to fix invalid XML output."""
    llm_client = LLMClient()
    
    # Construct healing prompt
    healing_prompt = f"""Fix the following invalid XML to make it valid. The XML appears to be malformed or contains formatting errors.
Ensure proper XML syntax, all tags are properly closed, and the structure is valid.
If a schema is provided, ensure the output matches the schema structure.

Invalid XML:
{invalid_xml}

{f'Required Schema:\n{json.dumps(schema, indent=2)}' if schema else ''}

Return ONLY the fixed XML without any explanations or markdown formatting. The output should be a valid XML document that can be parsed directly."""

    # Generate fixed XML
    try:
        fixed_xml = await llm_client.generate(
            prompt=healing_prompt,
            temperature=0.1,
            max_tokens=8000
        )
        
        logger.info(f"LLM Response for XML healing:\n{fixed_xml}")
        
        # Extract XML if it's wrapped in code blocks
        if "```xml" in fixed_xml:
            fixed_xml = fixed_xml.split("```xml")[1].split("```")[0].strip()
        elif "```" in fixed_xml:
            fixed_xml = fixed_xml.split("```")[1].split("```")[0].strip()
        
        # Basic XML validation (ensure it starts with <?xml and has a root element)
        if not fixed_xml.strip().startswith('<?xml'):
            fixed_xml = '<?xml version="1.0" encoding="UTF-8"?>\n' + fixed_xml
        
        return fixed_xml
    except Exception as e:
        logger.error(f"Failed to self-heal XML: {str(e)}")
        raise

async def process_large_text(text: str, instructions: str, output_format: OutputFormat, schema: Optional[Dict[str, Any]] = None, max_tokens: int = 4000, temperature: float = 0.2) -> str:
    """Process large text by chunking it and then combining the results."""
    # Initialize text chunker
    chunker = TextChunker(max_chunk_size=6000, chunking_method="semantic")
    
    # Chunk the text
    chunks = chunker.chunk_text(text)
    
    # Process each chunk
    formatted_chunks = []
    for chunk_data in chunks:
        chunk_text = chunk_data["text"]
        chunk_index = chunk_data["chunk_index"]
        total_chunks = chunk_data["total_chunks"]
        
        # Adjust instructions for chunk context
        chunk_instructions = f"{instructions}\n\nThis is chunk {chunk_index + 1} of {total_chunks}."
        if chunk_index > 0:
            chunk_instructions += " Continue from the previous chunk."
        
        # Format the chunk
        formatted_chunk = await format_output(chunk_text, output_format, chunk_instructions, schema)
        formatted_chunks.append(formatted_chunk)
    
    # Combine the formatted chunks based on the output format
    if output_format == OutputFormat.JSON:
        # For JSON, we need to merge the JSON objects
        combined_result = {}
        for chunk in formatted_chunks:
            try:
                chunk_json = json.loads(chunk)
                if isinstance(chunk_json, dict):
                    combined_result.update(chunk_json)
                elif isinstance(chunk_json, list):
                    if not combined_result:
                        combined_result = []
                    if isinstance(combined_result, list):
                        combined_result.extend(chunk_json)
            except json.JSONDecodeError:
                logger.warning(f"Could not parse chunk as JSON: {chunk[:100]}...")
        
        return json.dumps(combined_result, indent=2)
    
    elif output_format == OutputFormat.YAML:
        # For YAML, we need to merge the YAML objects
        combined_result = {}
        for chunk in formatted_chunks:
            try:
                chunk_yaml = yaml.safe_load(chunk)
                if isinstance(chunk_yaml, dict):
                    combined_result.update(chunk_yaml)
                elif isinstance(chunk_yaml, list):
                    if not combined_result:
                        combined_result = []
                    if isinstance(combined_result, list):
                        combined_result.extend(chunk_yaml)
            except yaml.YAMLError:
                logger.warning(f"Could not parse chunk as YAML: {chunk[:100]}...")
        
        return yaml.dump(combined_result, sort_keys=False, default_flow_style=False)
    
    elif output_format == OutputFormat.CSV:
        # For CSV, we need to keep the header from the first chunk and combine the data rows
        if not formatted_chunks:
            return ""
        
        combined_rows = []
        header = None
        
        for i, chunk in enumerate(formatted_chunks):
            # Parse CSV
            csv_reader = csv.reader(io.StringIO(chunk))
            rows = list(csv_reader)
            
            if not rows:
                continue
            
            if i == 0:
                # Keep the header from the first chunk
                header = rows[0]
                combined_rows.extend(rows)
            else:
                # Skip the header for subsequent chunks
                combined_rows.extend(rows[1:] if len(rows) > 1 else [])
        
        # Write back to CSV
        output = io.StringIO()
        csv_writer = csv.writer(output)
        csv_writer.writerows(combined_rows)
        return output.getvalue()
    
    else:
        # For other formats, simple concatenation with appropriate separators
        if output_format == OutputFormat.MARKDOWN or output_format == OutputFormat.HTML:
            return "\n\n".join(formatted_chunks)
        elif output_format == OutputFormat.SQL:
            return "\n\n".join(formatted_chunks)
        elif output_format == OutputFormat.XML:
            # This is a simplification - proper XML merging would be more complex
            # For now, we'll just wrap the chunks in a root element
            return f"<root>\n{''.join(formatted_chunks)}\n</root>"
        else:
            return "\n".join(formatted_chunks)

# Endpoints
@app.post("/self-heal", response_model=SelfHealResponse)
async def self_heal_text(request: SelfHealRequest, current_user: Dict[str, Any] = Depends(get_current_user)):
    """Attempt to fix formatting issues in text using LLM."""
    try:
        logger.info(f"Starting self-healing for {request.format} format")
        
        # Select the appropriate healing function based on format
        healing_functions = {
            OutputFormat.JSON: self_heal_json,
            OutputFormat.YAML: self_heal_yaml,
            OutputFormat.CSV: self_heal_csv,
            OutputFormat.XML: self_heal_xml
        }
        
        healing_function = healing_functions.get(request.format)
        if not healing_function:
            raise HTTPException(
                status_code=400,
                detail=f"Self-healing is not supported for {request.format} format"
            )
        
        # Attempt to heal the text
        healed_text = await healing_function(request.text, request.schema)
        
        logger.info(f"Completed self-healing for {request.format} format")
        
        return SelfHealResponse(
            healed_text=healed_text,
            format=request.format,
            timestamp=datetime.now()
        )
    except Exception as e:
        logger.error(f"Error in self-healing: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to heal the text",
                "original_error": str(e),
                "raw_input": request.text
            }
        )

@app.post("/convert", response_model=ConversionResponse)
async def convert_text(request: ConversionRequest, current_user: Dict[str, Any] = Depends(get_current_user)):
    """Convert unstructured text to a specified format."""
    try:
        conversion_id = str(uuid.uuid4())
        logger.info(f"Starting conversion {conversion_id} to {request.output_format}")
        
        # Check if text is large and needs chunking
        if len(request.text) > 4000:  # Approximate threshold
            output = await process_large_text(
                request.text, 
                request.instructions, 
                request.output_format, 
                request.schema,
                request.max_tokens,
                request.temperature
            )
        else:
            output = await format_output(
                request.text, 
                request.output_format, 
                request.instructions, 
                request.schema
            )
        
        # Validate the output and attempt self-healing if needed
        try:
            if request.output_format == OutputFormat.JSON:
                json.loads(output)
            elif request.output_format == OutputFormat.YAML:
                yaml.safe_load(output)
            elif request.output_format == OutputFormat.CSV:
                csv.reader(io.StringIO(output))
            elif request.output_format == OutputFormat.XML:
                if not output.strip().startswith('<?xml'):
                    output = '<?xml version="1.0" encoding="UTF-8"?>\n' + output
        except Exception as e:
            logger.warning(f"Output validation failed, attempting self-healing: {str(e)}")
            healing_request = SelfHealRequest(
                text=output,
                format=request.output_format,
                schema=request.schema,
                max_tokens=request.max_tokens,
                temperature=request.temperature
            )
            healed_response = await self_heal_text(healing_request, current_user)
            output = healed_response.healed_text
        
        logger.info(f"Completed conversion {conversion_id}")
        
        return ConversionResponse(
            conversion_id=conversion_id,
            output=output,
            format=request.output_format,
            timestamp=datetime.now()
        )
    except Exception as e:
        logger.error(f"Error in conversion: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/convert/file", response_model=ConversionResponse)
async def convert_file(
    instructions: str = Body(...),
    output_format: OutputFormat = Body(...),
    schema: Optional[Dict[str, Any]] = Body(None),
    max_tokens: Optional[int] = Body(8000),
    temperature: Optional[float] = Body(0.2),
    file: UploadFile = File(...),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Convert an uploaded file to a specified format."""
    try:
        conversion_id = str(uuid.uuid4())
        logger.info(f"Starting file conversion {conversion_id} to {output_format}")
        
        # Read file content
        content = await file.read()
        text = content.decode("utf-8")
        
        # Process the text
        if len(text) > 4000:  # Approximate threshold
            output = await process_large_text(
                text, 
                instructions, 
                output_format, 
                schema,
                max_tokens,
                temperature
            )
        else:
            output = await format_output(
                text, 
                output_format, 
                instructions, 
                schema
            )
        
        # Validate the output and attempt self-healing if needed
        try:
            if output_format == OutputFormat.JSON:
                json.loads(output)
            elif output_format == OutputFormat.YAML:
                yaml.safe_load(output)
            elif output_format == OutputFormat.CSV:
                csv.reader(io.StringIO(output))
            elif output_format == OutputFormat.XML:
                if not output.strip().startswith('<?xml'):
                    output = '<?xml version="1.0" encoding="UTF-8"?>\n' + output
        except Exception as e:
            logger.warning(f"Output validation failed, attempting self-healing: {str(e)}")
            healing_request = SelfHealRequest(
                text=output,
                format=output_format,
                schema=schema,
                max_tokens=max_tokens,
                temperature=temperature
            )
            healed_response = await self_heal_text(healing_request, current_user)
            output = healed_response.healed_text
        
        logger.info(f"Completed file conversion {conversion_id}")
        
        return ConversionResponse(
            conversion_id=conversion_id,
            output=output,
            format=output_format,
            timestamp=datetime.now()
        )
    except Exception as e:
        logger.error(f"Error in file conversion: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/convert/stream")
async def convert_text_streaming(
    request: ConversionRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Stream the conversion process for large texts."""
    conversion_id = str(uuid.uuid4())
    
    async def conversion_stream():
        try:
            # Start the stream with metadata
            yield json.dumps({
                "conversion_id": conversion_id,
                "status": "started",
                "message": f"Starting conversion to {request.output_format}"
            }) + "\n"
            
            # Initialize text chunker
            chunker = TextChunker(max_chunk_size=6000, chunking_method="semantic")
            
            # Chunk the text
            chunks = chunker.chunk_text(request.text)
            total_chunks = len(chunks)
            
            # Process each chunk
            formatted_chunks = []
            for i, chunk_data in enumerate(chunks):
                chunk_text = chunk_data["text"]
                
                # Update progress
                progress = (i / total_chunks) * 100
                yield json.dumps({
                    "conversion_id": conversion_id,
                    "status": "processing",
                    "progress": progress,
                    "message": f"Processing chunk {i+1} of {total_chunks}"
                }) + "\n"
                
                # Adjust instructions for chunk context
                chunk_instructions = f"{request.instructions}\n\nThis is chunk {i + 1} of {total_chunks}."
                if i > 0:
                    chunk_instructions += " Continue from the previous chunk."
                
                # Format the chunk
                formatted_chunk = await format_output(
                    chunk_text, 
                    request.output_format, 
                    chunk_instructions, 
                    request.schema
                )
                
                # Validate the chunk and attempt self-healing if needed
                try:
                    if request.output_format == OutputFormat.JSON:
                        json.loads(formatted_chunk)
                    elif request.output_format == OutputFormat.YAML:
                        yaml.safe_load(formatted_chunk)
                    elif request.output_format == OutputFormat.CSV:
                        csv.reader(io.StringIO(formatted_chunk))
                    elif request.output_format == OutputFormat.XML:
                        if not formatted_chunk.strip().startswith('<?xml'):
                            formatted_chunk = '<?xml version="1.0" encoding="UTF-8"?>\n' + formatted_chunk
                except Exception as e:
                    logger.warning(f"Chunk validation failed, attempting self-healing: {str(e)}")
                    healing_request = SelfHealRequest(
                        text=formatted_chunk,
                        format=request.output_format,
                        schema=request.schema,
                        max_tokens=request.max_tokens,
                        temperature=request.temperature
                    )
                    healed_response = await self_heal_text(healing_request, current_user)
                    formatted_chunk = healed_response.healed_text
                
                formatted_chunks.append(formatted_chunk)
                
                # Send the chunk
                yield json.dumps({
                    "conversion_id": conversion_id,
                    "status": "chunk_complete",
                    "chunk_index": i,
                    "total_chunks": total_chunks,
                    "chunk_output": formatted_chunk
                }) + "\n"
            
            # Combine the chunks
            final_output = await process_large_text(
                request.text, 
                request.instructions, 
                request.output_format, 
                request.schema,
                request.max_tokens,
                request.temperature
            )
            
            # Validate the final output and attempt self-healing if needed
            try:
                if request.output_format == OutputFormat.JSON:
                    json.loads(final_output)
                elif request.output_format == OutputFormat.YAML:
                    yaml.safe_load(final_output)
                elif request.output_format == OutputFormat.CSV:
                    csv.reader(io.StringIO(final_output))
                elif request.output_format == OutputFormat.XML:
                    if not final_output.strip().startswith('<?xml'):
                        final_output = '<?xml version="1.0" encoding="UTF-8"?>\n' + final_output
            except Exception as e:
                logger.warning(f"Final output validation failed, attempting self-healing: {str(e)}")
                healing_request = SelfHealRequest(
                    text=final_output,
                    format=request.output_format,
                    schema=request.schema,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature
                )
                healed_response = await self_heal_text(healing_request, current_user)
                final_output = healed_response.healed_text
            
            # Complete the stream
            yield json.dumps({
                "conversion_id": conversion_id,
                "status": "completed",
                "progress": 100,
                "output": final_output,
                "format": request.output_format.value
            }) + "\n"
            
        except Exception as e:
            logger.error(f"Error in streaming conversion: {str(e)}")
            yield json.dumps({
                "conversion_id": conversion_id,
                "status": "error",
                "message": str(e)
            }) + "\n"
    
    return StreamingResponse(conversion_stream(), media_type="application/x-ndjson")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "format_conversion_service"}

# Environment info endpoint
@app.get("/environment")
async def environment_info():
    """Return information about the current runtime environment"""
    return {
        "service": "format_conversion_service",
        "version": "1.0.0",
        "environment": "container" if os.getenv("CONTAINER_ENV", "false").lower() == "true" else "local",
        "python_version": sys.version,
        "timestamp": datetime.now().isoformat()
    }

# Run the application
if __name__ == "__main__":
    import uvicorn
    
    # Determine if running in container or locally
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    
    print(f"Starting Format Conversion Service on http://{host}:{port}")
    print(f"Environment: {'Container' if os.getenv('CONTAINER_ENV', 'false').lower() == 'true' else 'Local Development'}")
    
    uvicorn.run("main:app", host=host, port=port, reload=True)
