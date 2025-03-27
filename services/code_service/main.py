# services/code_service/main.py
from fastapi import FastAPI, Depends, HTTPException, Body
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum
import uuid
import asyncio

from utilities.llm_client import LLMClient, LLMProvider
from utilities.logging_utils import StructuredLogger
from utilities.auth import get_current_user
from utilities.prompts import PromptTemplates

# Create app
app = FastAPI(
    title="Code Generation and Transformation Service",
    description="API for code generation, documentation, refactoring, and language translation"
)

# Setup logger
logger = StructuredLogger("code_service")

# Models
class ProgrammingLanguage(str, Enum):
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

class CodeGenerationRequest(BaseModel):
    specifications: str = Field(..., description="Natural language specifications for the code to generate")
    language: ProgrammingLanguage = Field(..., description="Target programming language")
    context: Optional[str] = Field(None, description="Additional context for the generation task")
    comments_level: Optional[str] = Field("moderate", description="Level of comments in the generated code: 'minimal', 'moderate', or 'extensive'")
    style_guide: Optional[str] = Field(None, description="Specific style guidelines to follow")
    
class CodeDocumentationRequest(BaseModel):
    code: str = Field(..., description="Code to document")
    language: ProgrammingLanguage = Field(..., description="Programming language of the code")
    documentation_style: Optional[str] = Field("standard", description="Style of documentation: 'standard', 'docstring', 'jsdoc', 'javadoc', etc.")
    include_examples: Optional[bool] = Field(True, description="Whether to include examples in the documentation")
    
class CodeRefactoringRequest(BaseModel):
    code: str = Field(..., description="Code to refactor")
    language: ProgrammingLanguage = Field(..., description="Programming language of the code")
    refactoring_goals: List[str] = Field(..., description="Goals of the refactoring (e.g., 'improve readability', 'optimize performance')")
    
class CodeTranslationRequest(BaseModel):
    code: str = Field(..., description="Code to translate")
    source_language: ProgrammingLanguage = Field(..., description="Source programming language")
    target_language: ProgrammingLanguage = Field(..., description="Target programming language")
    maintain_comments: Optional[bool] = Field(True, description="Whether to maintain comments in the translated code")
    
class TestGenerationRequest(BaseModel):
    code: str = Field(..., description="Code to generate tests for")
    language: ProgrammingLanguage = Field(..., description="Programming language of the code")
    test_framework: Optional[str] = Field(None, description="Testing framework to use (e.g., 'pytest', 'jest', 'junit')")
    coverage_level: Optional[str] = Field("medium", description="Level of test coverage: 'basic', 'medium', or 'comprehensive'")

class CodeResponse(BaseModel):
    request_id: str = Field(..., description="Unique identifier for the request")
    code: str = Field(..., description="Generated, documented, refactored, or translated code")
    explanation: Optional[str] = Field(None, description="Explanation of the code or the changes made")
    suggestions: Optional[List[str]] = Field(None, description="Suggestions for further improvements")

# Routes
@app.post("/generate", response_model=CodeResponse)
async def generate_code(
    request: CodeGenerationRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Generate code from natural language specifications."""
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

@app.post("/document", response_model=CodeResponse)
async def document_code(
    request: CodeDocumentationRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Generate documentation for existing code."""
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

@app.post("/refactor", response_model=CodeResponse)
async def refactor_code(
    request: CodeRefactoringRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Refactor existing code according to specified goals."""
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
            
            # Extract suggestions if any
            suggestions = None
            if "suggestion" in explanation.lower() or "recommend" in explanation.lower():
                suggestion_parts = explanation.split("Suggestions:")
                if len(suggestion_parts) > 1:
                    suggestions_text = suggestion_parts[1].strip()
                    suggestions = [s.strip() for s in suggestions_text.split("\n") if s.strip()]
        else:
            code = result
            explanation = None
            suggestions = None
        
        logger.info("Test cases generated successfully", request_id)
        
        return CodeResponse(
            request_id=request_id,
            code=code,
            explanation=explanation,
            suggestions=suggestions
        )
        
    except Exception as e:
        logger.error(f"Error generating test cases: {str(e)}", request_id)
        raise HTTPException(status_code=500, detail=f"Error generating test cases: {str(e)}")
 3:
            code = code_parts[1]
            if code.startswith(request.language.value):
                code = code[len(request.language.value):].strip()
                
            explanation = (code_parts[0] + code_parts[2]).strip()
            
            # Extract suggestions if any
            suggestions = None
            if "suggestion" in explanation.lower() or "recommend" in explanation.lower():
                suggestion_parts = explanation.split("Suggestions:")
                if len(suggestion_parts) > 1:
                    suggestions_text = suggestion_parts[1].strip()
                    suggestions = [s.strip() for s in suggestions_text.split("\n") if s.strip()]
        else:
            code = result
            explanation = None
            suggestions = None
        
        logger.info("Code refactored successfully", request_id)
        
        return CodeResponse(
            request_id=request_id,
            code=code,
            explanation=explanation,
            suggestions=suggestions
        )
        
    except Exception as e:
        logger.error(f"Error refactoring code: {str(e)}", request_id)
        raise HTTPException(status_code=500, detail=f"Error refactoring code: {str(e)}")

@app.post("/translate", response_model=CodeResponse)
async def translate_code(
    request: CodeTranslationRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Translate code from one programming language to another."""
    request_id = str(uuid.uuid4())
    logger.info(
        "Code translation request received", 
        request_id, 
        {"source_language": request.source_language, "target_language": request.target_language}
    )
    
    # Create LLM client
    llm_client = LLMClient()
    
    # Build the system prompt
    system_prompt = f"""
    You are an expert programmer fluent in both {request.source_language.value} and {request.target_language.value}.
    Translate the provided code from {request.source_language.value} to {request.target_language.value}.
    Maintain the original functionality and logic.
    {'Preserve comments and documentation in the translated code.' if request.maintain_comments else ''}
    Use idiomatic patterns and best practices in the target language.
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
            
            # Extract suggestions if any
            suggestions = None
            if "suggestion" in explanation.lower() or "recommend" in explanation.lower():
                suggestion_parts = explanation.split("Suggestions:")
                if len(suggestion_parts) > 1:
                    suggestions_text = suggestion_parts[1].strip()
                    suggestions = [s.strip() for s in suggestions_text.split("\n") if s.strip()]
        else:
            code = result
            explanation = None
            suggestions = None
        
        logger.info("Code translated successfully", request_id)
        
        return CodeResponse(
            request_id=request_id,
            code=code,
            explanation=explanation,
            suggestions=suggestions
        )
        
    except Exception as e:
        logger.error(f"Error translating code: {str(e)}", request_id)
        raise HTTPException(status_code=500, detail=f"Error translating code: {str(e)}")

@app.post("/generate-tests", response_model=CodeResponse)
async def generate_tests(
    request: TestGenerationRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Generate test cases for existing code."""
    request_id = str(uuid.uuid4())
    logger.info("Test generation request received", request_id, {"language": request.language})
    
    # Create LLM client
    llm_client = LLMClient()
    
    # Determine test framework if not provided
    if not request.test_framework:
        # Default test frameworks by language
        framework_map = {
            ProgrammingLanguage.PYTHON: "pytest",
            ProgrammingLanguage.JAVASCRIPT: "jest",
            ProgrammingLanguage.TYPESCRIPT: "jest",
            ProgrammingLanguage.JAVA: "junit",
            ProgrammingLanguage.CSHARP: "nunit",
            ProgrammingLanguage.GO: "testing",
            ProgrammingLanguage.RUST: "cargo test",
            ProgrammingLanguage.CPP: "googletest",
            ProgrammingLanguage.RUBY: "rspec",
            ProgrammingLanguage.PHP: "phpunit",
            ProgrammingLanguage.SWIFT: "xctest",
            ProgrammingLanguage.KOTLIN: "junit",
        }
        test_framework = framework_map.get(request.language, "standard testing library")
    else:
        test_framework = request.test_framework
    
    # Build the system prompt
    system_prompt = f"""
    You are an expert in software testing with deep knowledge of {request.language.value} and {test_framework}.
    Generate {request.coverage_level} test cases for the provided code.
    Focus on testing functionality, edge cases, and error handling.
    Use best practices for {test_framework}.
    """
    
    # Build the user prompt
    prompt = f"""
    Please generate test cases for this {request.language.value} code using {test_framework}:
    
    ```{request.language.value}
    {request.code}
    ```
    
    Coverage level: {request.coverage_level}
    
    Provide the test code and explain the testing strategy.
    """
    
    try:
        # Generate the test cases
        result = await llm_client.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.2,
            max_tokens=2000
        )
        
        # Similar parsing logic to extract code and explanations
        code_parts = result.split("```")
        
        if len(code_parts) >=