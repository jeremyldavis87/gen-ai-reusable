#!/usr/bin/env python3

"""
Simplified Format Conversion Service for local testing

This script provides a simplified version of the Format Conversion Service
that doesn't rely on external API calls for testing purposes.
"""

import os
import json
import yaml
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Union, Any
from enum import Enum

from fastapi import FastAPI, HTTPException, Body, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Setup FastAPI app
app = FastAPI(title="Format Conversion Service (Test)", 
              description="Test service for converting unstructured text into various structured formats")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    max_tokens: Optional[int] = Field(4000, description="Maximum tokens for LLM response")
    temperature: Optional[float] = Field(0.2, description="Temperature for LLM generation")

class ConversionResponse(BaseModel):
    conversion_id: str = Field(..., description="Unique identifier for this conversion")
    output: str = Field(..., description="The formatted output")
    format: OutputFormat = Field(..., description="The format of the output")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp of the conversion")

# Helper function to generate mock responses
def generate_mock_response(text: str, format_type: OutputFormat) -> str:
    """Generate a mock response for testing without API keys."""
    # Generate mock responses based on format type
    if format_type == OutputFormat.JSON:
        return '{"name": "John Smith", "age": 35, "occupation": "Software Engineer", "skills": ["Python", "JavaScript", "Docker"]}'
    elif format_type == OutputFormat.YAML:
        return 'name: John Smith\nage: 35\noccupation: Software Engineer\nskills:\n  - Python\n  - JavaScript\n  - Docker'
    elif format_type == OutputFormat.CSV:
        return 'name,age,occupation,skills\nJohn Smith,35,Software Engineer,"Python, JavaScript, Docker"'
    elif format_type == OutputFormat.XML:
        return '<person>\n  <name>John Smith</name>\n  <age>35</age>\n  <occupation>Software Engineer</occupation>\n  <skills>\n    <skill>Python</skill>\n    <skill>JavaScript</skill>\n    <skill>Docker</skill>\n  </skills>\n</person>'
    elif format_type == OutputFormat.MARKDOWN:
        return '# John Smith\n\n**Age:** 35\n\n**Occupation:** Software Engineer\n\n## Skills\n- Python\n- JavaScript\n- Docker'
    elif format_type == OutputFormat.HTML:
        return '<div class="person">\n  <h1>John Smith</h1>\n  <p><strong>Age:</strong> 35</p>\n  <p><strong>Occupation:</strong> Software Engineer</p>\n  <h2>Skills</h2>\n  <ul>\n    <li>Python</li>\n    <li>JavaScript</li>\n    <li>Docker</li>\n  </ul>\n</div>'
    elif format_type == OutputFormat.SQL:
        return 'CREATE TABLE person (\n  name VARCHAR(100),\n  age INT,\n  occupation VARCHAR(100)\n);\n\nINSERT INTO person (name, age, occupation) VALUES ("John Smith", 35, "Software Engineer");\n\nCREATE TABLE skills (\n  person_name VARCHAR(100),\n  skill VARCHAR(100)\n);\n\nINSERT INTO skills (person_name, skill) VALUES ("John Smith", "Python");\nINSERT INTO skills (person_name, skill) VALUES ("John Smith", "JavaScript");\nINSERT INTO skills (person_name, skill) VALUES ("John Smith", "Docker");'
    else:
        return "Mock response for testing purposes."

# Endpoints
@app.post("/convert", response_model=ConversionResponse)
async def convert_text(request: ConversionRequest):
    """Convert unstructured text to a specified format."""
    try:
        conversion_id = str(uuid.uuid4())
        print(f"Starting conversion {conversion_id} to {request.output_format}")
        
        # Generate mock response
        output = generate_mock_response(request.text, request.output_format)
        
        print(f"Completed conversion {conversion_id}")
        
        return ConversionResponse(
            conversion_id=conversion_id,
            output=output,
            format=request.output_format,
            timestamp=datetime.now()
        )
    except Exception as e:
        print(f"Error in conversion: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
