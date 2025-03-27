# services/quality_service/main.py
from fastapi import FastAPI, Depends, HTTPException, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
import uuid
from datetime import datetime
import json
import asyncio
from enum import Enum

from utilities.llm_client import LLMClient, LLMProvider
from utilities.logging_utils import StructuredLogger
from utilities.auth import get_current_user
from utilities.prompts import PromptTemplates

# Create app
app = FastAPI(
    title="Quality Assurance Service",
    description="API for code review, bug detection, security scanning, and documentation verification"
)

# Setup logger
logger = StructuredLogger("quality_service")

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
    HTML = "html"
    CSS = "css"
    SQL = "sql"
    OTHER = "other"
    
class CodeReviewRequest(BaseModel):
    code: str = Field(..., description="Code to review")
    language: ProgrammingLanguage = Field(..., description="Programming language of the code")
    review_focus: Optional[List[str]] = Field(None, description="Specific aspects to focus on in the review")
    style_guide: Optional[str] = Field(None, description="Style guide to follow")
    context: Optional[str] = Field(None, description="Additional context about the code")
    
class IssueLevel(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"
    
class CodeIssue(BaseModel):
    level: IssueLevel = Field(..., description="Severity level of the issue")
    type: str = Field(..., description="Type of issue")
    line_number: Optional[int] = Field(None, description="Line number where the issue occurs")
    description: str = Field(..., description="Description of the issue")
    suggestion: str = Field(..., description="Suggestion for fixing the issue")
    code_snippet: Optional[str] = Field(None, description="Relevant code snippet")
    
class CodeReviewResponse(BaseModel):
    request_id: str = Field(..., description="Unique identifier for the request")
    issues: List[CodeIssue] = Field(..., description="Issues found in the code")
    summary: str = Field(..., description="Summary of the review")
    quality_score: Optional[float] = Field(None, description="Overall code quality score (0-100)")
    best_practices: Optional[List[str]] = Field(None, description="Best practices followed")
    improvement_areas: Optional[List[str]] = Field(None, description="Areas for improvement")

class SecurityScanRequest(BaseModel):
    code: str = Field(..., description="Code to scan for security issues")
    language: ProgrammingLanguage = Field(..., description="Programming language of the code")
    security_level: Optional[str] = Field("standard", description="Level of security scanning: 'basic', 'standard', or 'comprehensive'")
    application_context: Optional[str] = Field(None, description="Context about the application")
    dependencies: Optional[List[Dict[str, str]]] = Field(None, description="Dependencies used by the code")
    
class SecurityVulnerability(BaseModel):
    level: IssueLevel = Field(..., description="Severity level of the vulnerability")
    type: str = Field(..., description="Type of vulnerability")
    line_number: Optional[int] = Field(None, description="Line number where the vulnerability occurs")
    description: str = Field(..., description="Description of the vulnerability")
    cwe_id: Optional[str] = Field(None, description="Common Weakness Enumeration ID")
    remediation: str = Field(..., description="Remediation steps")
    code_snippet: Optional[str] = Field(None, description="Relevant code snippet")
    
class SecurityScanResponse(BaseModel):
    request_id: str = Field(..., description="Unique identifier for the request")
    vulnerabilities: List[SecurityVulnerability] = Field(..., description="Vulnerabilities found in the code")
    summary: str = Field(..., description="Summary of the security scan")
    risk_level: IssueLevel = Field(..., description="Overall risk level")
    security_score: Optional[float] = Field(None, description="Security score (0-100)")
    best_practices: Optional[List[str]] = Field(None, description="Security best practices followed")
    improvement_areas: Optional[List[str]] = Field(None, description="Security areas for improvement")

class DocumentationVerificationRequest(BaseModel):
    code: str = Field(..., description="Code to verify documentation for")
    documentation: str = Field(..., description="Documentation to verify")
    language: ProgrammingLanguage = Field(..., description="Programming language of the code")
    verification_focus: Optional[List[str]] = Field(None, description="Specific aspects to focus on in the verification")
    
class DocumentationIssue(BaseModel):
    level: IssueLevel = Field(..., description="Severity level of the issue")
    type: str = Field(..., description="Type of documentation issue")
    element: str = Field(..., description="The code element with documentation issues")
    description: str = Field(..., description="Description of the issue")
    suggestion: str = Field(..., description="Suggestion for fixing the issue")
    
class DocumentationVerificationResponse(BaseModel):
    request_id: str = Field(..., description="Unique identifier for the request")
    issues: List[DocumentationIssue] = Field(..., description="Issues found in the documentation")
    summary: str = Field(..., description="Summary of the verification")
    completeness_score: Optional[float] = Field(None, description="Documentation completeness score (0-100)")
    accuracy_score: Optional[float] = Field(None, description="Documentation accuracy score (0-100)")
    missing_documentation: Optional[List[str]] = Field(None, description="Code elements missing documentation")
    outdated_documentation: Optional[List[str]] = Field(None, description="Outdated documentation elements")

# Routes
@app.post("/code-review", response_model=CodeReviewResponse)
async def review_code(
    request: CodeReviewRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Perform code review to identify issues and suggest improvements."""
    request_id = str(uuid.uuid4())
    
    try:
        logger.info(
            "Code review request received", 
            request_id, 
            {"language": request.language}
        )
        
        # Create LLM client
        llm_client = LLMClient()
        
        # Build the system prompt
        system_prompt = f"""
        You are an expert code reviewer specializing in {request.language.value}.
        Perform a thorough code review to identify issues and suggest improvements.
        Focus on code quality, readability, maintainability, performance, and best practices.
        Provide specific, actionable feedback for each issue identified.
        {f'Pay special attention to these aspects: {", ".join(request.review_focus)}' if request.review_focus else ''}
        {f'Follow this style guide: {request.style_guide}' if request.style_guide else ''}
        """
        
        # Build the user prompt using the code quality template
        quality_criteria = "code quality, readability, maintainability, performance, and best practices"
        if request.review_focus:
            quality_criteria = ", ".join(request.review_focus)
            
        prompt = PromptTemplates.code_quality(request.code, quality_criteria)
        
        # Add context if provided
        if request.context:
            prompt += f"\n\nAdditional context about the code:\n{request.context}"
        
        # Append format instructions
        prompt += """
        
        Format your response as a valid JSON object with this structure:
        {
            "issues": [
                {
                    "level": "critical/high/medium/low/info",
                    "type": "Type of issue",
                    "line_number": 42,
                    "description": "Description of the issue",
                    "suggestion": "Suggestion for fixing the issue",
                    "code_snippet": "Relevant code snippet"
                },
                ...
            ],
            "summary": "Summary of the review",
            "quality_score": 75,
            "best_practices": [
                "Best practice 1",
                ...
            ],
            "improvement_areas": [
                "Area for improvement 1",
                ...
            ]
        }
        """
        
        # Generate the code review
        result = await llm_client.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.2,
            max_tokens=2000
        )
        
        # Parse the result
        try:
            # Find JSON in the response
            json_start = result.find("{")
            json_end = result.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = result[json_start:json_end]
                review_data = json.loads(json_str)
            else:
                # If no JSON found, try to parse the entire response
                review_data = json.loads(result)
            
            return CodeReviewResponse(
                request_id=request_id,
                issues=review_data.get("issues", []),
                summary=review_data.get("summary", ""),
                quality_score=review_data.get("quality_score"),
                best_practices=review_data.get("best_practices"),
                improvement_areas=review_data.get("improvement_areas")
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing code review result: {str(e)}", request_id)
            raise HTTPException(status_code=500, detail=f"Error parsing code review result: {str(e)}")
            
    except Exception as e:
        logger.error(f"Error reviewing code: {str(e)}", request_id)
        raise HTTPException(status_code=500, detail=f"Error reviewing code: {str(e)}")

@app.post("/security-scan", response_model=SecurityScanResponse)
async def scan_for_security_issues(
    request: SecurityScanRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Scan code for security vulnerabilities and suggest remediations."""
    request_id = str(uuid.uuid4())
    
    try:
        logger.info(
            "Security scan request received", 
            request_id, 
            {"language": request.language, "security_level": request.security_level}
        )
        
        # Create LLM client
        llm_client = LLMClient()
        
        # Build the system prompt
        system_prompt = f"""
        You are an expert security analyst specializing in {request.language.value} code security.
        Perform a {request.security_level} security scan to identify vulnerabilities and weaknesses.
        Focus on common security issues like injection flaws, authentication problems, sensitive data exposure, etc.
        For each vulnerability, provide a clear description and specific remediation steps.
        Reference CWE IDs where applicable.
        """
        
        # Format dependencies if provided
        dependencies_str = ""
        if request.dependencies:
            dependencies_str = "Dependencies:\n"
            for dep in request.dependencies:
                name = dep.get("name", "")
                version = dep.get("version", "")
                dependencies_str += f"- {name}@{version}\n"
        
        # Build the user prompt
        prompt = f"""
        Please scan this {request.language.value} code for security vulnerabilities:
        
        ```{request.language.value}
        {request.code}
        ```
        
        Security scan level: {request.security_level}
        
        {dependencies_str}
        
        {f'Application context: {request.application_context}' if request.application_context else ''}
        
        Format your response as a valid JSON object with this structure:
        {{
            "vulnerabilities": [
                {{
                    "level": "critical/high/medium/low/info",
                    "type": "Type of vulnerability",
                    "line_number": 42,
                    "description": "Description of the vulnerability",
                    "cwe_id": "CWE-123",
                    "remediation": "Remediation steps",
                    "code_snippet": "Relevant code snippet"
                }},
                ...
            ],
            "summary": "Summary of the security scan",
            "risk_level": "critical/high/medium/low/info",
            "security_score": 75,
            "best_practices": [
                "Security best practice 1",
                ...
            ],
            "improvement_areas": [
                "Security area for improvement 1",
                ...
            ]
        }}
        """
        
        # Generate the security scan
        result = await llm_client.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.2,
            max_tokens=2000
        )
        
        # Parse the result
        try:
            # Find JSON in the response
            json_start = result.find("{")
            json_end = result.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = result[json_start:json_end]
                scan_data = json.loads(json_str)
            else:
                # If no JSON found, try to parse the entire response
                scan_data = json.loads(result)
            
            return SecurityScanResponse(
                request_id=request_id,
                vulnerabilities=scan_data.get("vulnerabilities", []),
                summary=scan_data.get("summary", ""),
                risk_level=scan_data.get("risk_level", IssueLevel.LOW),
                security_score=scan_data.get("security_score"),
                best_practices=scan_data.get("best_practices"),
                improvement_areas=scan_data.get("improvement_areas")
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing security scan result: {str(e)}", request_id)
            raise HTTPException(status_code=500, detail=f"Error parsing security scan result: {str(e)}")
            
    except Exception as e:
        logger.error(f"Error scanning for security issues: {str(e)}", request_id)
        raise HTTPException(status_code=500, detail=f"Error scanning for security issues: {str(e)}")

@app.post("/verify-documentation", response_model=DocumentationVerificationResponse)
async def verify_documentation(
    request: DocumentationVerificationRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Verify documentation for completeness, accuracy, and consistency with code."""
    request_id = str(uuid.uuid4())
    
    try:
        logger.info(
            "Documentation verification request received", 
            request_id, 
            {"language": request.language}
        )
        
        # Create LLM client
        llm_client = LLMClient()
        
        # Build the system prompt
        system_prompt = f"""
        You are an expert documentation analyst specializing in {request.language.value} code documentation.
        Verify the provided documentation for completeness, accuracy, and consistency with the code.
        Check if all code elements (functions, classes, methods, etc.) are properly documented.
        Identify missing, outdated, or incorrect documentation.
        {f'Focus on these aspects: {", ".join(request.verification_focus)}' if request.verification_focus else ''}
        """
        
        # Build the user prompt
        prompt = f"""
        Please verify this {request.language.value} code documentation:
        
        Code:
        ```{request.language.value}
        {request.code}
        ```
        
        Documentation:
        ```
        {request.documentation}
        ```
        
        {f'Verification focus: {", ".join(request.verification_focus)}' if request.verification_focus else ''}
        
        Format your response as a valid JSON object with this structure:
        {{
            "issues": [
                {{
                    "level": "critical/high/medium/low/info",
                    "type": "Type of documentation issue",
                    "element": "Code element with documentation issues",
                    "description": "Description of the issue",
                    "suggestion": "Suggestion for fixing the issue"
                }},
                ...
            ],
            "summary": "Summary of the verification",
            "completeness_score": 75,
            "accuracy_score": 80,
            "missing_documentation": [
                "Code element 1",
                ...
            ],
            "outdated_documentation": [
                "Documentation element 1",
                ...
            ]
        }}
        """
        
        # Generate the documentation verification
        result = await llm_client.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.2,
            max_tokens=2000
        )
        
        # Parse the result
        try:
            # Find JSON in the response
            json_start = result.find("{")
            json_end = result.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = result[json_start:json_end]
                verification_data = json.loads(json_str)
            else:
                # If no JSON found, try to parse the entire response
                verification_data = json.loads(result)
            
            return DocumentationVerificationResponse(
                request_id=request_id,
                issues=verification_data.get("issues", []),
                summary=verification_data.get("summary", ""),
                completeness_score=verification_data.get("completeness_score"),
                accuracy_score=verification_data.get("accuracy_score"),
                missing_documentation=verification_data.get("missing_documentation"),
                outdated_documentation=verification_data.get("outdated_documentation")
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing documentation verification result: {str(e)}", request_id)
            raise HTTPException(status_code=500, detail=f"Error parsing documentation verification result: {str(e)}")
            
    except Exception as e:
        logger.error(f"Error verifying documentation: {str(e)}", request_id)
        raise HTTPException(status_code=500, detail=f"Error verifying documentation: {str(e)}")

@app.post("/check-compliance")
async def check_compliance(
    code: str = Body(..., embed=True),
    language: ProgrammingLanguage = Body(..., embed=True),
    standards: List[str] = Body(..., embed=True),
    context: Optional[str] = Body(None, embed=True),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Check code for compliance with specified standards or regulations."""
    request_id = str(uuid.uuid4())
    
    try:
        logger.info(
            "Compliance check request received", 
            request_id, 
            {"language": language, "standards": standards}
        )
        
        # Create LLM client
        llm_client = LLMClient()
        
        # Build the system prompt
        system_prompt = f"""
        You are an expert compliance analyst specializing in {language.value} code.
        Check the provided code for compliance with the specified standards or regulations.
        Identify compliance issues and suggest remediations.
        Be thorough and specific in your analysis.
        """
        
        # Format standards for the prompt
        standards_str = "Standards to check for compliance:\n"
        for std in standards:
            standards_str += f"- {std}\n"
        
        # Build the user prompt
        prompt = f"""
        Please check this {language.value} code for compliance with the specified standards:
        
        ```{language.value}
        {code}
        ```
        
        {standards_str}
        
        {f'Context: {context}' if context else ''}
        
        Format your response as a valid JSON object with this structure:
        {{
            "compliance_issues": [
                {{
                    "standard": "Standard name",
                    "requirement": "Specific requirement",
                    "level": "critical/high/medium/low/info",
                    "description": "Description of the compliance issue",
                    "location": "Location in the code",
                    "remediation": "Suggested remediation"
                }},
                ...
            ],
            "summary": "Summary of the compliance check",
            "overall_compliance": true/false,
            "compliance_score": 75,
            "standard_specific_results": [
                {{
                    "standard": "Standard name",
                    "compliant": true/false,
                    "issues_count": 3,
                    "notes": "Notes about compliance with this standard"
                }},
                ...
            ]
        }}
        """
        
        # Generate the compliance check
        result = await llm_client.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.2,
            max_tokens=2000
        )
        
        # Parse the result
        try:
            # Find JSON in the response
            json_start = result.find("{")
            json_end = result.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = result[json_start:json_end]
                compliance_data = json.loads(json_str)
            else:
                # If no JSON found, try to parse the entire response
                compliance_data = json.loads(result)
            
            return JSONResponse(
                content={
                    "request_id": request_id,
                    "compliance_check_results": compliance_data
                }
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing compliance check result: {str(e)}", request_id)
            raise HTTPException(status_code=500, detail=f"Error parsing compliance check result: {str(e)}")
            
    except Exception as e:
        logger.error(f"Error checking compliance: {str(e)}", request_id)
        raise HTTPException(status_code=500, detail=f"Error checking compliance: {str(e)}")

@app.post("/bug-prediction")
async def predict_bugs(
    code: str = Body(..., embed=True),
    language: ProgrammingLanguage = Body(..., embed=True),
    code_history: Optional[List[Dict[str, Any]]] = Body(None, embed=True),
    context: Optional[str] = Body(None, embed=True),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Predict potential bugs in code based on static analysis and patterns."""
    request_id = str(uuid.uuid4())
    
    try:
        logger.info(
            "Bug prediction request received", 
            request_id, 
            {"language": language}
        )
         
        # Create LLM client
        llm_client = LLMClient()
        
        # Build the system prompt
        system_prompt = f"""
        You are an expert bug predictor specializing in {language.value} code.
        Analyze the provided code to predict potential bugs that may occur.
        Focus on edge cases, error handling, resource management, concurrency issues, etc.
        Consider both obvious bugs and more subtle issues that might arise.
        """
        
        # Format code history if provided
        history_str = ""
        if code_history:
            history_str = "Code History (for context):\n"
            for i, entry in enumerate(code_history):
                history_str += f"Change {i+1}:\n"
                if 'description' in entry:
                    history_str += f"Description: {entry['description']}\n"
                if 'date' in entry:
                    history_str += f"Date: {entry['date']}\n"
                if 'author' in entry:
                    history_str += f"Author: {entry['author']}\n"
                if 'changes' in entry:
                    history_str += f"Changes: {entry['changes']}\n"
                history_str += "\n"
        
        # Build the user prompt
        prompt = f"""
        Please predict potential bugs in this {language.value} code:
        
        ```{language.value}
        {code}
        ```
        
        {history_str}
        
        {f'Context: {context}' if context else ''}
        
        Format your response as a valid JSON object with this structure:
        {{
            "predicted_bugs": [
                {{
                    "type": "Type of bug",
                    "location": "Location in the code",
                    "probability": "high/medium/low",
                    "description": "Description of the potential bug",
                    "scenario": "Scenario in which the bug might occur",
                    "prevention": "How to prevent this bug"
                }},
                ...
            ],
            "summary": "Summary of the bug prediction analysis",
            "high_risk_areas": [
                "High risk area 1",
                ...
            ],
            "recommended_tests": [
                "Test scenario 1",
                ...
            ]
        }}
        """
        
        # Generate the bug prediction
        result = await llm_client.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.3,
            max_tokens=2000
        )
        
        # Parse the result
        try:
            # Find JSON in the response
            json_start = result.find("{")
            json_end = result.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = result[json_start:json_end]
                prediction_data = json.loads(json_str)
            else:
                # If no JSON found, try to parse the entire response
                prediction_data = json.loads(result)
            
            return JSONResponse(
                content={
                    "request_id": request_id,
                    "bug_prediction_results": prediction_data
                }
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing bug prediction result: {str(e)}", request_id)
            raise HTTPException(status_code=500, detail=f"Error parsing bug prediction result: {str(e)}")
            
    except Exception as e:
        logger.error(f"Error predicting bugs: {str(e)}", request_id)
        raise HTTPException(status_code=500, detail=f"Error predicting bugs: {str(e)}")
