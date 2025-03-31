"""Utility functions for document processing and text extraction."""
import io
import json
import pandas as pd
from typing import Optional
from fastapi import UploadFile
from utilities.logging_utils import StructuredLogger

# Document parsers
import PyPDF2
import docx
from pdfminer.high_level import extract_text as extract_text_pdfminer

# Setup logger
logger = StructuredLogger("document_utils")

async def extract_text_from_file(file: UploadFile) -> str:
    """Extract text from different document types.
    
    Args:
        file (UploadFile): The uploaded file to extract text from.
        
    Returns:
        str: The extracted text content.
        
    Raises:
        ValueError: If the file type is unsupported or if there's an error processing the file.
    """
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