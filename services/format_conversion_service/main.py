# services/format_conversion_service/main.py

import os
import json
import yaml
import csv
import io
import uuid
import logging
import asyncio
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
    max_tokens: Optional[int] = Field(4000, description="Maximum tokens for LLM response")
    temperature: Optional[float] = Field(0.2, description="Temperature for LLM generation")

class FileConversionRequest(BaseModel):
    instructions: str = Field(..., description="Instructions for how to structure the output")
    output_format: OutputFormat = Field(..., description="The desired output format")
    schema: Optional[Dict[str, Any]] = Field(None, description="Optional schema definition for the output")
    max_tokens: Optional[int] = Field(4000, description="Maximum tokens for LLM response")
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

# Authentication dependency (placeholder - implement actual auth)
async def get_current_user():
    # In a real implementation, this would validate the token and return the user
    return {"id": "user123", "name": "Test User"}

# Helper functions
async def format_output(text: str, format_type: OutputFormat, instructions: str, schema: Optional[Dict[str, Any]] = None) -> str:
    """Format the text according to the specified output format."""
    llm_client = LLMClient()
    
    # Construct system prompt based on format type
    system_prompts = {
        OutputFormat.JSON: "You are a precise JSON formatter. Convert the provided text into valid JSON format according to the instructions. Ensure the output is properly structured and valid JSON.",
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

Provide ONLY the formatted output without any explanations or markdown code blocks."""
    
    # Generate the formatted output
    result = await llm_client.generate(
        prompt=user_prompt,
        system_prompt=system_prompt,
        temperature=0.2,  # Low temperature for more deterministic output
        max_tokens=4000   # Allow for larger outputs
    )
    
    # Clean up the result based on format type
    if format_type == OutputFormat.JSON:
        # Extract JSON if it's wrapped in code blocks
        if "```json" in result:
            result = result.split("```json")[1].split("```")[0].strip()
        elif "```" in result:
            result = result.split("```")[1].split("```")[0].strip()
        
        # Validate JSON
        try:
            json_obj = json.loads(result)
            result = json.dumps(json_obj, indent=2)  # Pretty print
        except json.JSONDecodeError:
            logger.warning("LLM generated invalid JSON, returning as-is")
    
    elif format_type == OutputFormat.YAML:
        # Extract YAML if it's wrapped in code blocks
        if "```yaml" in result:
            result = result.split("```yaml")[1].split("```")[0].strip()
        elif "```" in result:
            result = result.split("```")[1].split("```")[0].strip()
        
        # Validate YAML
        try:
            yaml_obj = yaml.safe_load(result)
            result = yaml.dump(yaml_obj, sort_keys=False, default_flow_style=False)  # Pretty print
        except yaml.YAMLError:
            logger.warning("LLM generated invalid YAML, returning as-is")
    
    elif format_type == OutputFormat.CSV:
        # Extract CSV if it's wrapped in code blocks
        if "```csv" in result:
            result = result.split("```csv")[1].split("```")[0].strip()
        elif "```" in result:
            result = result.split("```")[1].split("```")[0].strip()
    
    return result

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
    max_tokens: Optional[int] = Body(4000),
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

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
