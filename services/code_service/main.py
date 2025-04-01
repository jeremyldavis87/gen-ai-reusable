# services/code_service/main.py
from fastapi import FastAPI, Depends, HTTPException, Body
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum
import uuid
import asyncio
import json
import os

from utilities.llm_client import LLMClient, LLMProvider
from utilities.logging_utils import StructuredLogger
from utilities.auth import get_current_user
from utilities.prompts import PromptTemplates

# Load OpenAPI documentation
def load_openapi_docs():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    docs_path = os.path.join(current_dir, "api_docs.json")
    try:
        with open(docs_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load OpenAPI documentation: {e}")
        return None

# Create app with OpenAPI documentation
openapi_docs = load_openapi_docs()
app = FastAPI(
    title="Code Generation and Transformation Service",
    description="""
    API for code generation, documentation, refactoring, and language translation.
    
    ## Features
    - Natural language to code conversion
    - Support for multiple programming languages
    - Code documentation generation
    - Code refactoring and optimization
    - Cross-language translation
    - Test case generation
    
    ## Supported Languages
    - Python
    - JavaScript/TypeScript
    - Java
    - C#
    - Go
    - Rust
    - C++
    - Ruby
    - PHP
    - Swift
    - Kotlin
    
    ## Code Quality Features
    - Error handling
    - Input validation
    - Performance optimization
    - Clean code principles
    - Documentation generation
    
    ## Example Use Cases
    - Rapid prototyping
    - Code scaffolding
    - Algorithm implementation
    - API endpoint generation
    - Data structure implementation
    """,
    version="1.0.0",
    openapi_url="/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_schema=openapi_docs,
    openapi_tags=[
        {
            "name": "code",
            "description": "Operations for code generation and transformation"
        }
    ]
)

# Setup logger
logger = StructuredLogger("code_service")

# Models
class ProgrammingLanguage(str, Enum):
    """Supported programming languages."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    CSHARP = "csharp"
    GO = "go"
    RUST = "rust"
    CPP = "cpp"
    RUBY = "ruby"
    PHP = "php"
    SWIFT = "swift"
    KOTLIN = "kotlin"
    OTHER = "other"

    class Config:
        schema_extra = {
            "example": "python"
        }

class CodeGenerationRequest(BaseModel):
    """Request model for code generation."""
    specifications: str = Field(
        ..., 
        description="Natural language specifications for the code to generate",
        example="Create a function that calculates the Fibonacci sequence up to n terms"
    )
    language: ProgrammingLanguage = Field(
        ..., 
        description="Target programming language",
        example="python"
    )
    context: Optional[str] = Field(
        None, 
        description="Additional context for the generation task",
        example="The function should be optimized for performance and include error handling"
    )
    comments_level: Optional[str] = Field(
        "moderate", 
        description="Level of comments in the generated code: 'minimal', 'moderate', or 'extensive'",
        example="moderate"
    )
    style_guide: Optional[str] = Field(
        None, 
        description="Specific style guidelines to follow",
        example="PEP 8"
    )

    class Config:
        schema_extra = {
            "example": {
                "specifications": "Create a function that calculates the Fibonacci sequence up to n terms",
                "language": "python",
                "context": "The function should be optimized for performance and include error handling",
                "comments_level": "moderate",
                "style_guide": "PEP 8"
            }
        }

class CodeDocumentationRequest(BaseModel):
    """Request model for code documentation."""
    code: str = Field(..., description="Code to document")
    language: ProgrammingLanguage = Field(..., description="Programming language of the code")
    documentation_style: Optional[str] = Field("standard", description="Style of documentation: 'standard', 'docstring', 'jsdoc', 'javadoc', etc.")
    include_examples: Optional[bool] = Field(True, description="Whether to include examples in the documentation")
    
class CodeRefactoringRequest(BaseModel):
    """Request model for code refactoring."""
    code: str = Field(..., description="Code to refactor")
    language: ProgrammingLanguage = Field(..., description="Programming language of the code")
    refactoring_goals: List[str] = Field(..., description="Goals of the refactoring (e.g., 'improve readability', 'optimize performance')")
    
class CodeTranslationRequest(BaseModel):
    """Request model for code translation."""
    code: str = Field(..., description="Code to translate")
    source_language: ProgrammingLanguage = Field(..., description="Source programming language")
    target_language: ProgrammingLanguage = Field(..., description="Target programming language")
    maintain_comments: Optional[bool] = Field(True, description="Whether to maintain comments in the translated code")
    
class TestGenerationRequest(BaseModel):
    """Request model for test generation."""
    code: str = Field(..., description="Code to generate tests for")
    language: ProgrammingLanguage = Field(..., description="Programming language of the code")
    test_framework: Optional[str] = Field(None, description="Testing framework to use (e.g., 'pytest', 'jest', 'junit')")
    coverage_level: Optional[str] = Field("medium", description="Level of test coverage: 'basic', 'medium', or 'comprehensive'")

class CodeResponse(BaseModel):
    """Response model for code-related operations."""
    request_id: str = Field(..., description="Unique identifier for the request")
    code: str = Field(..., description="Generated, documented, refactored, or translated code")
    explanation: Optional[str] = Field(None, description="Explanation of the code or the changes made")
    suggestions: Optional[List[str]] = Field(None, description="Suggestions for further improvements")

    class Config:
        schema_extra = {
            "example": {
                "request_id": "a534a8c5-3309-4790-935b-6dc1a2e59ba0",
                "code": "def fibonacci(n):\n    if n <= 0:\n        return []\n    elif n == 1:\n        return [0]\n    \n    sequence = [0, 1]\n    for i in range(2, n):\n        sequence.append(sequence[i-1] + sequence[i-2])\n    return sequence",
                "explanation": "The function generates a Fibonacci sequence up to n terms using an efficient iterative approach.",
                "suggestions": [
                    "Consider adding type hints for better code clarity",
                    "Add input validation for negative numbers",
                    "Consider using a generator for memory efficiency with large n"
                ]
            }
        }

# Routes
@app.post(
    "/generate", 
    response_model=CodeResponse,
    tags=["code"],
    summary="Generate code from natural language specifications",
    description="""
    Generate code based on natural language specifications using advanced LLM techniques.
    
    ## Features
    - Natural language to code conversion
    - Support for multiple programming languages
    - Configurable code style and comments
    - Context-aware code generation
    - Performance optimization suggestions
    
    ## Example Use Cases
    - Rapid prototyping
    - Algorithm implementation
    - Data structure implementation
    - API endpoint generation
    - Utility function creation
    
    ## Code Quality
    - Error handling
    - Input validation
    - Performance optimization
    - Clean code principles
    - Documentation
    """,
    response_description="Successfully generated code with explanations and suggestions"
)
async def generate_code(
    request: CodeGenerationRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Generate code from natural language specifications.
    
    Args:
        request: The code generation request containing specifications and preferences
        current_user: The authenticated user making the request
        
    Returns:
        CodeResponse containing the generated code, explanations, and suggestions
        
    Raises:
        HTTPException: If there's an error during code generation
    """
    request_id = str(uuid.uuid4())
    logger.info("Code generation request received", request_id, {"language": request.language})
    
    # Create LLM client
    llm_client = LLMClient()
    
    # Build the system prompt
    system_prompt = f"""
    You are an expert programmer with deep knowledge of {request.language.value}.
    Generate clean, efficient, and well-structured code based on the specifications provided.
    Follow these guidelines:
    - Comments level: {request.comments_level}
    - Use idiomatic {request.language.value} code
    {f'- Follow these style guidelines: {request.style_guide}' if request.style_guide else ''}
    """
    
    # Build the user prompt
    prompt = f"""
    Please generate {request.language.value} code based on these specifications:
    
    {request.specifications}
    
    {f'Additional context: {request.context}' if request.context else ''}
    """
    
    try:
        # Generate the code
        result = await llm_client.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.2,  # Lower temperature for more deterministic code generation
            max_tokens=2000
        )
        
        # Extract code and explanations
        # This is a simplified parsing - in a real service, you would need more robust parsing
        code_parts = result.split("```")
        
        if len(code_parts) >= 3:
            # Extract the code inside the code block
            code = code_parts[1]
            if code.startswith(request.language.value):
                code = code[len(request.language.value):].strip()
                
            # Extract explanations outside the code block
            explanation = (code_parts[0] + code_parts[2]).strip()
            
            # Extract suggestions if any
            suggestions = None
            if "suggestion" in explanation.lower() or "recommend" in explanation.lower():
                # Simple heuristic to extract suggestions
                suggestion_parts = explanation.split("Suggestions:")
                if len(suggestion_parts) > 1:
                    suggestions_text = suggestion_parts[1].strip()
                    suggestions = [s.strip() for s in suggestions_text.split("\n") if s.strip()]
        else:
            # If there are no code blocks, the entire result is the code
            code = result
            explanation = None
            suggestions = None
        
        logger.info("Code generated successfully", request_id)
        
        return CodeResponse(
            request_id=request_id,
            code=code,
            explanation=explanation,
            suggestions=suggestions
        )
        
    except Exception as e:
        logger.error(f"Error generating code: {str(e)}", request_id)
        raise HTTPException(status_code=500, detail=f"Error generating code: {str(e)}")

@app.post(
    "/document",
    response_model=CodeResponse,
    tags=["code"],
    summary="Generate documentation for existing code",
    description="""
    Generate comprehensive documentation for existing code using AI-powered analysis.
    
    ## Features
    - Multiple documentation styles
    - Code analysis and explanation
    - Example generation
    - Best practices adherence
    - Language-specific documentation
    
    ## Documentation Styles
    - Standard: General documentation format
    - Docstring: Python-specific docstring format
    - JSDoc: JavaScript documentation format
    - JavaDoc: Java documentation format
    - Custom: User-defined documentation format
    
    ## Documentation Components
    - Function/class description
    - Parameter documentation
    - Return value documentation
    - Usage examples
    - Edge cases and limitations
    - Performance considerations
    
    ## Example Use Cases
    - Legacy code documentation
    - API documentation
    - Library documentation
    - Code review documentation
    """,
    response_description="Successfully generated documentation with explanations"
)
async def document_code(
    request: CodeDocumentationRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Generate documentation for existing code.
    
    Args:
        request: The documentation request containing code and preferences
        current_user: The authenticated user making the request
        
    Returns:
        CodeResponse containing the documented code and explanations
        
    Raises:
        HTTPException: If there's an error during documentation generation
    """
    request_id = str(uuid.uuid4())
    logger.info("Code documentation request received", request_id, {"language": request.language})
    
    # Create LLM client
    llm_client = LLMClient()
    
    # Build the system prompt
    system_prompt = f"""
    You are an expert technical writer with deep knowledge of {request.language.value}.
    Generate comprehensive documentation for the provided code following {request.documentation_style} style.
    {'Include practical usage examples.' if request.include_examples else ''}
    """
    
    # Use the documentation prompt template
    prompt = PromptTemplates.code_documentation(request.code)
    
    try:
        # Generate the documentation
        result = await llm_client.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.3,
            max_tokens=2000
        )
        
        # In a real service, you would parse and format the response more carefully
        
        return CodeResponse(
            request_id=request_id,
            code=result,  # The documented code or documentation to be applied
            explanation="Documentation generated based on the provided code."
        )
        
    except Exception as e:
        logger.error(f"Error documenting code: {str(e)}", request_id)
        raise HTTPException(status_code=500, detail=f"Error documenting code: {str(e)}")

@app.post(
    "/refactor",
    response_model=CodeResponse,
    tags=["code"],
    summary="Refactor existing code according to specified goals",
    description="""
    Refactor code to improve quality, maintainability, and performance.
    
    ## Features
    - Code quality improvement
    - Performance optimization
    - Readability enhancement
    - Best practices implementation
    - Design pattern application
    
    ## Refactoring Goals
    - Improve readability
    - Optimize performance
    - Enhance maintainability
    - Implement design patterns
    - Add error handling
    - Improve type safety
    - Reduce complexity
    - Enhance modularity
    
    ## Code Analysis
    - Static code analysis
    - Complexity metrics
    - Code smell detection
    - Performance bottlenecks
    - Security vulnerabilities
    
    ## Example Use Cases
    - Legacy code modernization
    - Performance optimization
    - Code quality improvement
    - Design pattern implementation
    - Security enhancement
    """,
    response_description="Successfully refactored code with explanations of changes"
)
async def refactor_code(
    request: CodeRefactoringRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Refactor code according to specified goals.
    
    Args:
        request: The refactoring request containing code and goals
        current_user: The authenticated user making the request
        
    Returns:
        CodeResponse containing the refactored code and explanations
        
    Raises:
        HTTPException: If there's an error during code refactoring
    """
    request_id = str(uuid.uuid4())
    logger.info("Code refactoring request received", request_id, {"language": request.language})
    
    # Create LLM client
    llm_client = LLMClient()
    
    # Build the system prompt
    system_prompt = f"""
    You are an expert programmer with deep knowledge of {request.language.value} and software design principles.
    Refactor the provided code to achieve these goals:
    {', '.join(request.refactoring_goals)}
    
    Maintain the original functionality while improving the code according to best practices.
    """
    
    # Build the user prompt
    prompt = f"""
    Please refactor this {request.language.value} code:
    
    ```{request.language.value}
    {request.code}
    ```
    
    Refactoring goals:
    {', '.join(request.refactoring_goals)}
    
    Provide the refactored code and explain the changes made.
    """
    
    try:
        # Generate the refactored code
        result = await llm_client.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.2,
            max_tokens=2000
        )
        
        # Similar parsing logic to extract code and explanations
        code_parts = result.split("```")
        
        if len(code_parts) >= 3:
            code = code_parts[1]
            if code.startswith(request.language.value):
                code = code[len(request.language.value):].strip()
            explanation = (code_parts[0] + code_parts[2]).strip()
        else:
            code = result
            explanation = None
        
        return CodeResponse(
            request_id=request_id,
            code=code,
            explanation=explanation
        )
        
    except Exception as e:
        logger.error(f"Error refactoring code: {str(e)}", request_id)
        raise HTTPException(status_code=500, detail=f"Error refactoring code: {str(e)}")

@app.post(
    "/translate",
    response_model=CodeResponse,
    tags=["code"],
    summary="Translate code from one programming language to another",
    description="""
    Translate code between different programming languages while preserving functionality.
    
    ## Features
    - Cross-language translation
    - Idiomatic code generation
    - Comment preservation
    - Error handling translation
    - Library mapping
    
    ## Language Support
    - Python ↔ JavaScript/TypeScript
    - Java ↔ C#
    - Python ↔ Go
    - JavaScript ↔ TypeScript
    - And more...
    
    ## Translation Features
    - Idiomatic code style
    - Library equivalents
    - Error handling patterns
    - Type system mapping
    - Memory management
    
    ## Example Use Cases
    - Codebase migration
    - Cross-platform development
    - Language learning
    - Library porting
    - Framework adaptation
    """,
    response_description="Successfully translated code with explanations of adaptations"
)
async def translate_code(
    request: CodeTranslationRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Translate code from one programming language to another.
    
    Args:
        request: The translation request containing source code and target language
        current_user: The authenticated user making the request
        
    Returns:
        CodeResponse containing the translated code and explanations
        
    Raises:
        HTTPException: If there's an error during code translation
    """
    request_id = str(uuid.uuid4())
    logger.info("Code translation request received", request_id, {
        "source_language": request.source_language,
        "target_language": request.target_language
    })
    
    # Create LLM client
    llm_client = LLMClient()
    
    # Build the system prompt
    system_prompt = f"""
    You are an expert programmer with deep knowledge of both {request.source_language.value} and {request.target_language.value}.
    Translate the provided code from {request.source_language.value} to {request.target_language.value}.
    {'Maintain comments and documentation in the translated code.' if request.maintain_comments else ''}
    Ensure the translated code follows idiomatic patterns in the target language.
    """
    
    # Build the user prompt
    prompt = f"""
    Please translate this {request.source_language.value} code to {request.target_language.value}:
    
    ```{request.source_language.value}
    {request.code}
    ```
    
    Provide the translated code and explain any significant changes or adaptations made.
    """
    
    try:
        # Generate the translated code
        result = await llm_client.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.2,
            max_tokens=2000
        )
        
        # Similar parsing logic to extract code and explanations
        code_parts = result.split("```")
        
        if len(code_parts) >= 3:
            code = code_parts[1]
            if code.startswith(request.target_language.value):
                code = code[len(request.target_language.value):].strip()
            explanation = (code_parts[0] + code_parts[2]).strip()
        else:
            code = result
            explanation = None
        
        return CodeResponse(
            request_id=request_id,
            code=code,
            explanation=explanation
        )
        
    except Exception as e:
        logger.error(f"Error translating code: {str(e)}", request_id)
        raise HTTPException(status_code=500, detail=f"Error translating code: {str(e)}")

@app.post(
    "/generate-tests",
    response_model=CodeResponse,
    tags=["code"],
    summary="Generate test cases for existing code",
    description="""
    Generate comprehensive test cases for existing code using AI-powered analysis.
    
    ## Features
    - Multiple test frameworks
    - Various coverage levels
    - Edge case detection
    - Performance testing
    - Integration testing
    
    ## Test Frameworks
    - pytest (Python)
    - Jest (JavaScript)
    - JUnit (Java)
    - NUnit (C#)
    - Go testing
    - And more...
    
    ## Coverage Levels
    - Basic: Essential test cases
    - Medium: Comprehensive test cases
    - Comprehensive: Full test coverage
    
    ## Test Types
    - Unit tests
    - Integration tests
    - Edge cases
    - Error cases
    - Performance tests
    
    ## Example Use Cases
    - Legacy code testing
    - API testing
    - Library testing
    - Framework testing
    - Performance testing
    """,
    response_description="Successfully generated test cases with explanations"
)
async def generate_tests(
    request: TestGenerationRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Generate test cases for existing code.
    
    Args:
        request: The test generation request containing code and preferences
        current_user: The authenticated user making the request
        
    Returns:
        CodeResponse containing the generated tests and explanations
        
    Raises:
        HTTPException: If there's an error during test generation
    """
    request_id = str(uuid.uuid4())
    logger.info("Test generation request received", request_id, {
        "language": request.language,
        "test_framework": request.test_framework,
        "coverage_level": request.coverage_level
    })
    
    # Create LLM client
    llm_client = LLMClient()
    
    # Build the system prompt
    system_prompt = f"""
    You are an expert in software testing with deep knowledge of {request.language.value}.
    Generate comprehensive tests for the provided code using {request.test_framework or 'the most appropriate testing framework'}.
    Test coverage level: {request.coverage_level}
    Include tests for both normal operation and edge cases.
    """
    
    # Build the user prompt
    prompt = f"""
    Please generate tests for this {request.language.value} code:
    
    ```{request.language.value}
    {request.code}
    ```
    
    Test requirements:
    - Framework: {request.test_framework or 'Choose the most appropriate framework'}
    - Coverage level: {request.coverage_level}
    - Include both positive and negative test cases
    - Add comments explaining the test cases
    
    Provide the test code and explain the test coverage strategy.
    """
    
    try:
        # Generate the tests
        result = await llm_client.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.2,
            max_tokens=2000
        )
        
        # Similar parsing logic to extract code and explanations
        code_parts = result.split("```")
        
        if len(code_parts) >= 3:
            code = code_parts[1]
            if code.startswith(request.language.value):
                code = code[len(request.language.value):].strip()
            explanation = (code_parts[0] + code_parts[2]).strip()
        else:
            code = result
            explanation = None
        
        return CodeResponse(
            request_id=request_id,
            code=code,
            explanation=explanation
        )
        
    except Exception as e:
        logger.error(f"Error generating tests: {str(e)}", request_id)
        raise HTTPException(status_code=500, detail=f"Error generating tests: {str(e)}")