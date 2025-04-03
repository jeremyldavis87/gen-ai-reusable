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

# Create app with detailed documentation
app = FastAPI(
    title="Quality Assurance Service",
    description="""
    The Quality Assurance Service provides AI-powered code quality analysis and improvement capabilities through a RESTful API.
    
    ## Key Features
    
    * **Code Review**: Automated code review with detailed feedback
    * **Security Scanning**: Vulnerability detection and security analysis
    * **Documentation Verification**: Documentation completeness and accuracy checks
    * **Compliance Checking**: Standards and regulation compliance verification
    * **Bug Prediction**: Potential bug detection and prevention
    
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
    - HTML/CSS
    - SQL
    
    ## Quality Aspects
    - Code Style and Standards
    - Security Vulnerabilities
    - Documentation Quality
    - Performance Issues
    - Error Handling
    - Test Coverage
    - Maintainability
    - Best Practices
    
    ## Authentication
    
    All endpoints require authentication using a valid API key or token.
    
    ## Rate Limiting
    
    The API is rate-limited to ensure fair usage. Please check the response headers for rate limit information.
    
    ## Best Practices
    
    1. Provide comprehensive code context for better analysis
    2. Use specific review focus areas for targeted feedback
    3. Include relevant style guides and standards
    4. Consider security implications in all code reviews
    5. Maintain up-to-date documentation
    6. Follow language-specific best practices
    7. Implement proper error handling
    8. Write comprehensive tests
    9. Consider performance implications
    10. Follow accessibility guidelines
    
    ## Example Use Cases
    - Automated Code Review
    - Security Vulnerability Detection
    - Documentation Quality Assurance
    - Compliance Verification
    - Bug Prevention
    """,
    version="1.0.0",
    contact={
        "name": "AI Services Team",
        "email": "ai-services@example.com",
    },
    license_info={
        "name": "Proprietary",
        "url": "https://example.com/license",
    },
    openapi_tags=[
        {
            "name": "code-review",
            "description": "Endpoints for automated code review and quality analysis"
        },
        {
            "name": "security",
            "description": "Endpoints for security scanning and vulnerability detection"
        },
        {
            "name": "documentation",
            "description": "Endpoints for documentation verification and improvement"
        },
        {
            "name": "compliance",
            "description": "Endpoints for compliance checking and standards verification"
        },
        {
            "name": "bug-prediction",
            "description": "Endpoints for bug prediction and prevention"
        },
        {
            "name": "utilities",
            "description": "Endpoints for utility functions"
        }
    ]
)

# Setup logger
logger = StructuredLogger("quality_service")

# Models
class ProgrammingLanguage(str, Enum):
    """
    Supported programming languages for quality analysis.
    
    Attributes:
        PYTHON: Python programming language
        JAVASCRIPT: JavaScript programming language
        TYPESCRIPT: TypeScript programming language
        JAVA: Java programming language
        CSHARP: C# programming language
        GO: Go programming language
        RUST: Rust programming language
        CPP: C++ programming language
        RUBY: Ruby programming language
        PHP: PHP programming language
        SWIFT: Swift programming language
        KOTLIN: Kotlin programming language
        HTML: HTML markup language
        CSS: CSS styling language
        SQL: SQL query language
        OTHER: Other programming languages
    """
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
    
    class Config:
        schema_extra = {
            "description": "Programming language of the code to analyze",
            "example": "python"
        }
    
class CodeReviewRequest(BaseModel):
    """
    Request model for code review analysis.
    
    Attributes:
        code: The code to review
        language: Programming language of the code
        review_focus: Specific aspects to focus on in the review
        style_guide: Style guide to follow
        context: Additional context about the code
    """
    code: str = Field(..., description="Code to review")
    language: ProgrammingLanguage = Field(..., description="Programming language of the code")
    review_focus: Optional[List[str]] = Field(None, description="Specific aspects to focus on in the review")
    style_guide: Optional[str] = Field(None, description="Style guide to follow")
    context: Optional[str] = Field(None, description="Additional context about the code")
    
    class Config:
        schema_extra = {
            "example": {
                "code": "def calculate_sum(a, b):\n    return a + b",
                "language": "python",
                "review_focus": ["code style", "error handling", "documentation"],
                "style_guide": "PEP 8",
                "context": "This is a utility function for basic arithmetic operations"
            }
        }
    
class IssueLevel(str, Enum):
    """
    Severity levels for code issues.
    
    Attributes:
        CRITICAL: Critical issues that must be fixed immediately
        HIGH: High priority issues that should be fixed soon
        MEDIUM: Medium priority issues that should be addressed
        LOW: Low priority issues that can be fixed later
        INFO: Informational issues or suggestions
    """
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"
    
    class Config:
        schema_extra = {
            "description": "Severity level of the issue",
            "example": "high"
        }
    
class CodeIssue(BaseModel):
    """
    Model for code issues found during review.
    
    Attributes:
        level: Severity level of the issue
        type: Type of issue
        line_number: Line number where the issue occurs
        description: Description of the issue
        suggestion: Suggestion for fixing the issue
        code_snippet: Relevant code snippet
    """
    level: IssueLevel = Field(..., description="Severity level of the issue")
    type: str = Field(..., description="Type of issue")
    line_number: Optional[int] = Field(None, description="Line number where the issue occurs")
    description: str = Field(..., description="Description of the issue")
    suggestion: str = Field(..., description="Suggestion for fixing the issue")
    code_snippet: Optional[str] = Field(None, description="Relevant code snippet")
    
    class Config:
        schema_extra = {
            "example": {
                "level": "high",
                "type": "security",
                "line_number": 42,
                "description": "Potential SQL injection vulnerability",
                "suggestion": "Use parameterized queries instead of string concatenation",
                "code_snippet": "query = f\"SELECT * FROM users WHERE id = {user_id}\""
            }
        }
    
class CodeReviewResponse(BaseModel):
    """
    Response model for code review results.
    
    Attributes:
        request_id: Unique identifier for the request
        issues: List of issues found in the code
        summary: Summary of the review
        quality_score: Overall code quality score (0-100)
        best_practices: Best practices followed
        improvement_areas: Areas for improvement
    """
    request_id: str = Field(..., description="Unique identifier for the request")
    issues: List[CodeIssue] = Field(..., description="Issues found in the code")
    summary: str = Field(..., description="Summary of the review")
    quality_score: Optional[float] = Field(None, description="Overall code quality score (0-100)")
    best_practices: Optional[List[str]] = Field(None, description="Best practices followed")
    improvement_areas: Optional[List[str]] = Field(None, description="Areas for improvement")
    
    class Config:
        schema_extra = {
            "example": {
                "request_id": "123e4567-e89b-12d3-a456-426614174000",
                "issues": [
                    {
                        "level": "high",
                        "type": "security",
                        "line_number": 42,
                        "description": "Potential SQL injection vulnerability",
                        "suggestion": "Use parameterized queries",
                        "code_snippet": "query = f\"SELECT * FROM users WHERE id = {user_id}\""
                    }
                ],
                "summary": "Code review completed with 3 issues found",
                "quality_score": 85.5,
                "best_practices": ["Proper error handling", "Clean code structure"],
                "improvement_areas": ["Add input validation", "Improve documentation"]
            }
        }

class SecurityScanRequest(BaseModel):
    """
    Request model for security scanning.
    
    Attributes:
        code: The code to scan for security issues
        language: Programming language of the code
        security_level: Level of security scanning
        application_context: Context about the application
        dependencies: Dependencies used by the code
    """
    code: str = Field(..., description="Code to scan for security issues")
    language: ProgrammingLanguage = Field(..., description="Programming language of the code")
    security_level: Optional[str] = Field("standard", description="Level of security scanning: 'basic', 'standard', or 'comprehensive'")
    application_context: Optional[str] = Field(None, description="Context about the application")
    dependencies: Optional[List[Dict[str, str]]] = Field(None, description="Dependencies used by the code")
    
    class Config:
        schema_extra = {
            "example": {
                "code": "def process_user_input(user_input):\n    return eval(user_input)",
                "language": "python",
                "security_level": "comprehensive",
                "application_context": "Web application",
                "dependencies": [
                    {"name": "flask", "version": "2.0.1"},
                    {"name": "sqlalchemy", "version": "1.4.23"}
                ]
            }
        }
    
class SecurityVulnerability(BaseModel):
    """
    Model for security vulnerabilities found during scanning.
    
    Attributes:
        level: Severity level of the vulnerability
        type: Type of vulnerability
        line_number: Line number where the vulnerability occurs
        description: Description of the vulnerability
        cwe_id: Common Weakness Enumeration ID
        remediation: Remediation steps
        code_snippet: Relevant code snippet
    """
    level: IssueLevel = Field(..., description="Severity level of the vulnerability")
    type: str = Field(..., description="Type of vulnerability")
    line_number: Optional[int] = Field(None, description="Line number where the vulnerability occurs")
    description: str = Field(..., description="Description of the vulnerability")
    cwe_id: Optional[str] = Field(None, description="Common Weakness Enumeration ID")
    remediation: str = Field(..., description="Remediation steps")
    code_snippet: Optional[str] = Field(None, description="Relevant code snippet")
    
    class Config:
        schema_extra = {
            "example": {
                "level": "critical",
                "type": "code_injection",
                "line_number": 42,
                "description": "Use of eval() with user input",
                "cwe_id": "CWE-94",
                "remediation": "Use safe alternatives like ast.literal_eval() or proper input validation",
                "code_snippet": "result = eval(user_input)"
            }
        }
    
class SecurityScanResponse(BaseModel):
    """
    Response model for security scanning results.
    
    Attributes:
        request_id: Unique identifier for the request
        vulnerabilities: List of vulnerabilities found
        summary: Summary of the security scan
        risk_level: Overall risk level
        security_score: Security score (0-100)
        best_practices: Security best practices followed
        improvement_areas: Security areas for improvement
    """
    request_id: str = Field(..., description="Unique identifier for the request")
    vulnerabilities: List[SecurityVulnerability] = Field(..., description="Vulnerabilities found in the code")
    summary: str = Field(..., description="Summary of the security scan")
    risk_level: IssueLevel = Field(..., description="Overall risk level")
    security_score: Optional[float] = Field(None, description="Security score (0-100)")
    best_practices: Optional[List[str]] = Field(None, description="Security best practices followed")
    improvement_areas: Optional[List[str]] = Field(None, description="Security areas for improvement")
    
    class Config:
        schema_extra = {
            "example": {
                "request_id": "123e4567-e89b-12d3-a456-426614174000",
                "vulnerabilities": [
                    {
                        "level": "critical",
                        "type": "code_injection",
                        "line_number": 42,
                        "description": "Use of eval() with user input",
                        "cwe_id": "CWE-94",
                        "remediation": "Use safe alternatives",
                        "code_snippet": "result = eval(user_input)"
                    }
                ],
                "summary": "Security scan completed with 2 critical vulnerabilities found",
                "risk_level": "high",
                "security_score": 65.0,
                "best_practices": ["Input validation", "Secure dependencies"],
                "improvement_areas": ["Remove eval() usage", "Add input sanitization"]
            }
        }

class DocumentationVerificationRequest(BaseModel):
    """
    Request model for documentation verification.
    
    Attributes:
        code: The code to verify documentation for
        documentation: Documentation to verify
        language: Programming language of the code
        verification_focus: Specific aspects to focus on in the verification
    """
    code: str = Field(..., description="Code to verify documentation for")
    documentation: str = Field(..., description="Documentation to verify")
    language: ProgrammingLanguage = Field(..., description="Programming language of the code")
    verification_focus: Optional[List[str]] = Field(None, description="Specific aspects to focus on in the verification")
    
    class Config:
        schema_extra = {
            "example": {
                "code": "def calculate_sum(a, b):\n    return a + b",
                "documentation": "This function adds two numbers together.",
                "language": "python",
                "verification_focus": ["completeness", "accuracy", "examples"]
            }
        }
    
class DocumentationIssue(BaseModel):
    """
    Model for documentation issues found during verification.
    
    Attributes:
        level: Severity level of the issue
        type: Type of documentation issue
        element: The code element with documentation issues
        description: Description of the issue
        suggestion: Suggestion for fixing the issue
    """
    level: IssueLevel = Field(..., description="Severity level of the issue")
    type: str = Field(..., description="Type of documentation issue")
    element: str = Field(..., description="The code element with documentation issues")
    description: str = Field(..., description="Description of the issue")
    suggestion: str = Field(..., description="Suggestion for fixing the issue")
    
    class Config:
        schema_extra = {
            "example": {
                "level": "medium",
                "type": "incomplete",
                "element": "calculate_sum",
                "description": "Missing parameter descriptions and return value documentation",
                "suggestion": "Add docstring with parameter and return value descriptions"
            }
        }
    
class DocumentationVerificationResponse(BaseModel):
    """
    Response model for documentation verification results.
    
    Attributes:
        request_id: Unique identifier for the request
        issues: List of documentation issues found
        summary: Summary of the verification
        completeness_score: Documentation completeness score (0-100)
        accuracy_score: Documentation accuracy score (0-100)
        missing_documentation: Code elements missing documentation
        outdated_documentation: Outdated documentation elements
    """
    request_id: str = Field(..., description="Unique identifier for the request")
    issues: List[DocumentationIssue] = Field(..., description="Issues found in the documentation")
    summary: str = Field(..., description="Summary of the verification")
    completeness_score: Optional[float] = Field(None, description="Documentation completeness score (0-100)")
    accuracy_score: Optional[float] = Field(None, description="Documentation accuracy score (0-100)")
    missing_documentation: Optional[List[str]] = Field(None, description="Code elements missing documentation")
    outdated_documentation: Optional[List[str]] = Field(None, description="Outdated documentation elements")
    
    class Config:
        schema_extra = {
            "example": {
                "request_id": "123e4567-e89b-12d3-a456-426614174000",
                "issues": [
                    {
                        "level": "medium",
                        "type": "incomplete",
                        "element": "calculate_sum",
                        "description": "Missing parameter descriptions",
                        "suggestion": "Add parameter documentation"
                    }
                ],
                "summary": "Documentation verification completed with 2 issues found",
                "completeness_score": 75.0,
                "accuracy_score": 90.0,
                "missing_documentation": ["Element 1", "Element 2"],
                "outdated_documentation": ["Element 3", "Element 4"]
            }
        }

class ComplianceIssue(BaseModel):
    """
    Model for compliance issues found during verification.
    
    Attributes:
        standard: Name of the standard or regulation
        requirement: Specific requirement that was violated
        level: Severity level of the issue
        description: Description of the compliance issue
        location: Location in the code where the issue occurs
        remediation: Steps to achieve compliance
    """
    standard: str = Field(..., description="Name of the standard or regulation")
    requirement: str = Field(..., description="Specific requirement that was violated")
    level: IssueLevel = Field(..., description="Severity level of the issue")
    description: str = Field(..., description="Description of the compliance issue")
    location: Optional[str] = Field(None, description="Location in the code where the issue occurs")
    remediation: str = Field(..., description="Steps to achieve compliance")
    
    class Config:
        schema_extra = {
            "example": {
                "standard": "PEP 8",
                "requirement": "Line length",
                "level": "low",
                "description": "Line exceeds 79 characters",
                "location": "Line 42",
                "remediation": "Break line into multiple lines"
            }
        }

class StandardSpecificResult(BaseModel):
    """
    Model for standard-specific compliance results.
    
    Attributes:
        standard: Name of the standard or regulation
        compliant: Whether the code is compliant with this standard
        issues_count: Number of compliance issues found
        notes: Additional notes about compliance with this standard
    """
    standard: str = Field(..., description="Name of the standard or regulation")
    compliant: bool = Field(..., description="Whether the code is compliant with this standard")
    issues_count: int = Field(..., description="Number of compliance issues found")
    notes: Optional[str] = Field(None, description="Additional notes about compliance with this standard")
    
    class Config:
        schema_extra = {
            "example": {
                "standard": "PEP 8",
                "compliant": True,
                "issues_count": 2,
                "notes": "Minor style issues found"
            }
        }

class ComplianceRequest(BaseModel):
    """
    Request model for compliance checking.
    
    Attributes:
        code: The code to check for compliance
        language: Programming language of the code
        standards: List of standards to check against
        context: Additional context about the code
    """
    code: str = Field(..., description="Code to check for compliance")
    language: ProgrammingLanguage = Field(..., description="Programming language of the code")
    standards: List[str] = Field(..., description="Standards to check against")
    context: Optional[str] = Field(None, description="Additional context about the code")
    
    class Config:
        schema_extra = {
            "example": {
                "code": "def process_data(data):\n    return data",
                "language": "python",
                "standards": ["PEP 8", "ISO 27001"],
                "context": "Data processing function"
            }
        }

class ComplianceResponse(BaseModel):
    """
    Response model for compliance check results.
    
    Attributes:
        request_id: Unique identifier for the request
        compliance_issues: List of compliance issues found
        summary: Summary of the compliance check
        overall_compliance: Whether the code is compliant overall
        compliance_score: Overall compliance score (0-100)
        standard_specific_results: Results specific to each standard
    """
    request_id: str = Field(..., description="Unique identifier for the request")
    compliance_issues: List[ComplianceIssue] = Field(..., description="List of compliance issues found")
    summary: str = Field(..., description="Summary of the compliance check")
    overall_compliance: bool = Field(..., description="Whether the code is compliant overall")
    compliance_score: float = Field(..., description="Overall compliance score (0-100)")
    standard_specific_results: List[StandardSpecificResult] = Field(..., description="Results specific to each standard")
    
    class Config:
        schema_extra = {
            "example": {
                "request_id": "123e4567-e89b-12d3-a456-426614174000",
                "compliance_issues": [
                    {
                        "standard": "PEP 8",
                        "requirement": "Line length",
                        "level": "low",
                        "description": "Line exceeds 79 characters",
                        "location": "Line 42",
                        "remediation": "Break line into multiple lines"
                    }
                ],
                "summary": "Code is mostly compliant with minor style issues",
                "overall_compliance": True,
                "compliance_score": 95.0,
                "standard_specific_results": [
                    {
                        "standard": "PEP 8",
                        "compliant": True,
                        "issues_count": 2,
                        "notes": "Minor style issues found"
                    }
                ]
            }
        }

class BugPredictionRequest(BaseModel):
    """
    Request model for bug prediction.
    
    Attributes:
        code: The code to analyze for potential bugs
        language: Programming language of the code
        code_history: Optional history of code changes for context
        context: Additional context about the code
    """
    code: str = Field(..., description="Code to analyze for potential bugs")
    language: ProgrammingLanguage = Field(..., description="Programming language of the code")
    code_history: Optional[List[Dict[str, str]]] = Field(None, description="History of code changes for context")
    context: Optional[str] = Field(None, description="Additional context about the code")
    
    class Config:
        schema_extra = {
            "example": {
                "code": "def process_data(data):\n    return data",
                "language": "python",
                "code_history": [
                    {
                        "description": "Initial implementation",
                        "date": "2023-01-01",
                        "author": "developer1"
                    }
                ],
                "context": "Data processing function"
            }
        }

class PredictedBug(BaseModel):
    """
    Model for predicted bugs in code.
    
    Attributes:
        type: Type of potential bug
        location: Location in the code where the bug might occur
        probability: Likelihood of the bug occurring
        description: Description of the potential bug
        scenario: Scenario in which the bug might occur
        prevention: Steps to prevent this bug
    """
    type: str = Field(..., description="Type of potential bug")
    location: str = Field(..., description="Location in the code where the bug might occur")
    probability: str = Field(..., description="Likelihood of the bug occurring (high/medium/low)")
    description: str = Field(..., description="Description of the potential bug")
    scenario: str = Field(..., description="Scenario in which the bug might occur")
    prevention: str = Field(..., description="Steps to prevent this bug")
    
    class Config:
        schema_extra = {
            "example": {
                "type": "null_pointer",
                "location": "Line 42",
                "probability": "high",
                "description": "Potential null pointer exception when input is None",
                "scenario": "When input is None",
                "prevention": "Add null check before processing data"
            }
        }

class BugPredictionResponse(BaseModel):
    """
    Response model for bug prediction results.
    
    Attributes:
        request_id: Unique identifier for the request
        predicted_bugs: List of potential bugs found
        summary: Summary of the bug prediction analysis
        high_risk_areas: Areas of code with high risk of bugs
        recommended_tests: Recommended test scenarios
    """
    request_id: str = Field(..., description="Unique identifier for the request")
    predicted_bugs: List[PredictedBug] = Field(..., description="List of potential bugs found")
    summary: str = Field(..., description="Summary of the bug prediction analysis")
    high_risk_areas: List[str] = Field(..., description="Areas of code with high risk of bugs")
    recommended_tests: List[str] = Field(..., description="Recommended test scenarios")
    
    class Config:
        schema_extra = {
            "example": {
                "request_id": "123e4567-e89b-12d3-a456-426614174000",
                "predicted_bugs": [
                    {
                        "type": "null_pointer",
                        "location": "Line 42",
                        "probability": "high",
                        "description": "Potential null pointer exception",
                        "scenario": "When input is None",
                        "prevention": "Add null check"
                    }
                ],
                "summary": "Found 1 potential high-risk bug",
                "high_risk_areas": ["Input validation", "Error handling"],
                "recommended_tests": ["Test with None input", "Test with empty data"]
            }
        }

class SecurityFixRequest(BaseModel):
    """
    Request model for security fix generation.
    
    Attributes:
        code: The code to fix security issues in
        language: Programming language of the code
        security_level: Level of security fixes to apply
        application_context: Context about the application
        dependencies: Dependencies used by the code
    """
    code: str = Field(..., description="Code to fix security issues in")
    language: ProgrammingLanguage = Field(..., description="Programming language of the code")
    security_level: Optional[str] = Field("standard", description="Level of security fixes: 'basic', 'standard', or 'comprehensive'")
    application_context: Optional[str] = Field(None, description="Context about the application")
    dependencies: Optional[List[Dict[str, str]]] = Field(None, description="Dependencies used by the code")
    
    class Config:
        schema_extra = {
            "example": {
                "code": "def process_user_input(user_input):\n    return eval(user_input)",
                "language": "python",
                "security_level": "comprehensive",
                "application_context": "Web application handling user input",
                "dependencies": [
                    {"name": "flask", "version": "2.0.1"},
                    {"name": "sqlalchemy", "version": "1.4.23"}
                ]
            }
        }

class SecurityFix(BaseModel):
    """
    Model for security fixes applied to the code.
    
    Attributes:
        original_code: The original vulnerable code
        fixed_code: The fixed secure code
        vulnerability_type: Type of vulnerability that was fixed
        description: Description of the security fix
        cwe_id: Common Weakness Enumeration ID
        security_improvement: Description of how security was improved
    """
    original_code: str = Field(..., description="The original vulnerable code")
    fixed_code: str = Field(..., description="The fixed secure code")
    vulnerability_type: str = Field(..., description="Type of vulnerability that was fixed")
    description: str = Field(..., description="Description of the security fix")
    cwe_id: Optional[str] = Field(None, description="Common Weakness Enumeration ID")
    security_improvement: str = Field(..., description="Description of how security was improved")
    
    class Config:
        schema_extra = {
            "example": {
                "original_code": "result = eval(user_input)",
                "fixed_code": "result = ast.literal_eval(user_input)",
                "vulnerability_type": "code_injection",
                "description": "Replaced unsafe eval() with ast.literal_eval()",
                "cwe_id": "CWE-94",
                "security_improvement": "Prevents arbitrary code execution by restricting evaluation to literals only"
            }
        }

class SecurityFixResponse(BaseModel):
    """
    Response model for security fix results.
    
    Attributes:
        request_id: Unique identifier for the request
        fixed_code: Complete fixed code with all security issues resolved
        fixes: List of security fixes applied
        summary: Summary of the security improvements
        security_score_before: Security score before fixes (0-100)
        security_score_after: Security score after fixes (0-100)
        best_practices_applied: Security best practices applied in fixes
    """
    request_id: str = Field(..., description="Unique identifier for the request")
    fixed_code: str = Field(..., description="Complete fixed code with all security issues resolved")
    fixes: List[SecurityFix] = Field(..., description="List of security fixes applied")
    summary: str = Field(..., description="Summary of the security improvements")
    security_score_before: float = Field(..., description="Security score before fixes (0-100)")
    security_score_after: float = Field(..., description="Security score after fixes (0-100)")
    best_practices_applied: List[str] = Field(..., description="Security best practices applied in fixes")
    
    class Config:
        schema_extra = {
            "example": {
                "request_id": "123e4567-e89b-12d3-a456-426614174000",
                "fixed_code": "def process_user_input(user_input):\n    return ast.literal_eval(user_input)",
                "fixes": [
                    {
                        "original_code": "result = eval(user_input)",
                        "fixed_code": "result = ast.literal_eval(user_input)",
                        "vulnerability_type": "code_injection",
                        "description": "Replaced unsafe eval() with ast.literal_eval()",
                        "cwe_id": "CWE-94",
                        "security_improvement": "Prevents arbitrary code execution"
                    }
                ],
                "summary": "Fixed 1 critical security vulnerability",
                "security_score_before": 65.0,
                "security_score_after": 95.0,
                "best_practices_applied": ["Input validation", "Safe evaluation"]
            }
        }

class PayloadFormatRequest(BaseModel):
    """
    Request model for formatting code and dependencies into request payloads.
    
    Attributes:
        code: The multi-line code to format
        dependencies_file: Content of the dependencies file (e.g. requirements.txt, package.json)
        language: Programming language of the code
        file_type: Type of dependencies file (e.g. 'requirements.txt', 'package.json', 'Cargo.toml')
    """
    code: str = Field(..., description="Multi-line code to format")
    dependencies_file: str = Field(..., description="Content of the dependencies file")
    language: ProgrammingLanguage = Field(..., description="Programming language of the code")
    file_type: str = Field(..., description="Type of dependencies file")
    
    class Config:
        schema_extra = {
            "example": {
                "code": "def process_data(data):\n    return eval(data)",
                "dependencies_file": "flask==2.0.1\nsqlalchemy==1.4.23\nrequests>=2.25.0",
                "language": "python",
                "file_type": "requirements.txt"
            }
        }

class PayloadFormatResponse(BaseModel):
    """
    Response model for formatted request payload components.
    
    Attributes:
        request_id: Unique identifier for the request
        formatted_code: Code formatted for request payloads
        formatted_dependencies: Dependencies formatted for request payloads
        detected_packages: List of detected package names
        package_versions: Mapping of package names to versions
    """
    request_id: str = Field(..., description="Unique identifier for the request")
    formatted_code: str = Field(..., description="Code formatted for request payloads")
    formatted_dependencies: List[Dict[str, str]] = Field(..., description="Dependencies formatted for request payloads")
    detected_packages: List[str] = Field(..., description="List of detected package names")
    package_versions: Dict[str, str] = Field(..., description="Mapping of package names to versions")
    
    class Config:
        schema_extra = {
            "example": {
                "request_id": "123e4567-e89b-12d3-a456-426614174000",
                "formatted_code": "def process_data(data):\n    return eval(data)",
                "formatted_dependencies": [
                    {"name": "flask", "version": "2.0.1"},
                    {"name": "sqlalchemy", "version": "1.4.23"},
                    {"name": "requests", "version": "2.25.0"}
                ],
                "detected_packages": ["flask", "sqlalchemy", "requests"],
                "package_versions": {
                    "flask": "2.0.1",
                    "sqlalchemy": "1.4.23",
                    "requests": "2.25.0"
                }
            }
        }

class DocumentationFixRequest(BaseModel):
    """
    Request model for documentation fix generation.
    
    Attributes:
        code: The code to fix documentation for
        language: Programming language of the code
        documentation_level: Level of documentation detail ('basic', 'standard', 'comprehensive')
        context: Additional context about the code
        style_guide: Optional documentation style guide to follow
    """
    code: str = Field(..., description="Code to fix documentation for")
    language: ProgrammingLanguage = Field(..., description="Programming language of the code")
    documentation_level: Optional[str] = Field("standard", description="Level of documentation detail: 'basic', 'standard', or 'comprehensive'")
    context: Optional[str] = Field(None, description="Additional context about the code")
    style_guide: Optional[str] = Field(None, description="Documentation style guide to follow")
    
    class Config:
        schema_extra = {
            "example": {
                "code": "def calculate_sum(a, b):\n    return a + b",
                "language": "python",
                "documentation_level": "comprehensive",
                "context": "Utility function for basic arithmetic operations",
                "style_guide": "Google Python Style Guide"
            }
        }

class DocumentationFix(BaseModel):
    """
    Model for documentation fixes applied to the code.
    
    Attributes:
        original_code: The original code with documentation issues
        fixed_code: The fixed code with proper documentation
        element: The code element that was fixed
        description: Description of the documentation fix
        improvement: Description of how documentation was improved
    """
    original_code: str = Field(..., description="The original code with documentation issues")
    fixed_code: str = Field(..., description="The fixed code with proper documentation")
    element: str = Field(..., description="The code element that was fixed")
    description: str = Field(..., description="Description of the documentation fix")
    improvement: str = Field(..., description="Description of how documentation was improved")
    
    class Config:
        schema_extra = {
            "example": {
                "original_code": "def calculate_sum(a, b):\n    return a + b",
                "fixed_code": "def calculate_sum(a: float, b: float) -> float:\n    \"\"\"Calculate the sum of two numbers.\n\n    Args:\n        a: First number to add\n        b: Second number to add\n\n    Returns:\n        The sum of a and b\n    \"\"\"\n    return a + b",
                "element": "calculate_sum function",
                "description": "Added function docstring with parameters and return type",
                "improvement": "Added type hints, parameter descriptions, and return value documentation"
            }
        }

class DocumentationFixResponse(BaseModel):
    """
    Response model for documentation fix results.
    
    Attributes:
        request_id: Unique identifier for the request
        fixed_code: Complete fixed code with proper documentation
        fixes: List of documentation fixes applied
        summary: Summary of the documentation improvements
        doc_score_before: Documentation score before fixes (0-100)
        doc_score_after: Documentation score after fixes (0-100)
        best_practices_applied: Documentation best practices applied
    """
    request_id: str = Field(..., description="Unique identifier for the request")
    fixed_code: str = Field(..., description="Complete fixed code with proper documentation")
    fixes: List[DocumentationFix] = Field(..., description="List of documentation fixes applied")
    summary: str = Field(..., description="Summary of the documentation improvements")
    doc_score_before: float = Field(..., description="Documentation score before fixes (0-100)")
    doc_score_after: float = Field(..., description="Documentation score after fixes (0-100)")
    best_practices_applied: List[str] = Field(..., description="Documentation best practices applied")
    
    class Config:
        schema_extra = {
            "example": {
                "request_id": "123e4567-e89b-12d3-a456-426614174000",
                "fixed_code": "def calculate_sum(a: float, b: float) -> float:\n    \"\"\"Calculate the sum of two numbers.\n\n    Args:\n        a: First number to add\n        b: Second number to add\n\n    Returns:\n        The sum of a and b\n    \"\"\"\n    return a + b",
                "fixes": [
                    {
                        "original_code": "def calculate_sum(a, b):\n    return a + b",
                        "fixed_code": "def calculate_sum(a: float, b: float) -> float:\n    \"\"\"Calculate the sum of two numbers.\n\n    Args:\n        a: First number to add\n        b: Second number to add\n\n    Returns:\n        The sum of a and b\n    \"\"\"\n    return a + b",
                        "element": "calculate_sum function",
                        "description": "Added function docstring with parameters and return type",
                        "improvement": "Added type hints, parameter descriptions, and return value documentation"
                    }
                ],
                "summary": "Added comprehensive documentation with type hints and docstrings",
                "doc_score_before": 45.0,
                "doc_score_after": 95.0,
                "best_practices_applied": ["Type hints", "Function docstrings", "Parameter descriptions", "Return value documentation"]
            }
        }

class ComplianceFixRequest(BaseModel):
    """
    Request model for compliance fix generation.
    
    Attributes:
        code: The code to fix compliance issues in
        language: Programming language of the code
        standards: List of standards to check against
        context: Additional context about the code
        fix_level: Level of fixes to apply ('basic', 'standard', 'comprehensive')
    """
    code: str = Field(..., description="Code to fix compliance issues in")
    language: ProgrammingLanguage = Field(..., description="Programming language of the code")
    standards: List[str] = Field(..., description="Standards to check against")
    context: Optional[str] = Field(None, description="Additional context about the code")
    fix_level: Optional[str] = Field("standard", description="Level of fixes to apply: 'basic', 'standard', or 'comprehensive'")
    
    class Config:
        schema_extra = {
            "example": {
                "code": "def process_data(data):\n    return data",
                "language": "python",
                "standards": ["PEP 8", "ISO 27001"],
                "context": "Data processing function",
                "fix_level": "comprehensive"
            }
        }

class ComplianceFix(BaseModel):
    """
    Model for compliance fixes applied to the code.
    
    Attributes:
        original_code: The original non-compliant code
        fixed_code: The fixed compliant code
        standard: The standard that was addressed
        requirement: The specific requirement that was fixed
        description: Description of the compliance fix
        improvement: Description of how compliance was improved
    """
    original_code: str = Field(..., description="The original non-compliant code")
    fixed_code: str = Field(..., description="The fixed compliant code")
    standard: str = Field(..., description="The standard that was addressed")
    requirement: str = Field(..., description="The specific requirement that was fixed")
    description: str = Field(..., description="Description of the compliance fix")
    improvement: str = Field(..., description="Description of how compliance was improved")
    
    class Config:
        schema_extra = {
            "example": {
                "original_code": "def process_data(data):\n    return data",
                "fixed_code": "def process_data(data: Any) -> Any:\n    \"\"\"Process the input data.\n\n    Args:\n        data: Input data to process\n\n    Returns:\n        Processed data\n    \"\"\"\n    return data",
                "standard": "PEP 8",
                "requirement": "Function documentation",
                "description": "Added function docstring and type hints",
                "improvement": "Improved code documentation and type safety"
            }
        }

class ComplianceFixResponse(BaseModel):
    """
    Response model for compliance fix results.
    
    Attributes:
        request_id: Unique identifier for the request
        fixed_code: Complete fixed code with all compliance issues resolved
        fixes: List of compliance fixes applied
        summary: Summary of the compliance improvements
        compliance_score_before: Compliance score before fixes (0-100)
        compliance_score_after: Compliance score after fixes (0-100)
        standard_specific_results: Results specific to each standard
    """
    request_id: str = Field(..., description="Unique identifier for the request")
    fixed_code: str = Field(..., description="Complete fixed code with all compliance issues resolved")
    fixes: List[ComplianceFix] = Field(..., description="List of compliance fixes applied")
    summary: str = Field(..., description="Summary of the compliance improvements")
    compliance_score_before: float = Field(..., description="Compliance score before fixes (0-100)")
    compliance_score_after: float = Field(..., description="Compliance score after fixes (0-100)")
    standard_specific_results: List[StandardSpecificResult] = Field(..., description="Results specific to each standard")
    
    class Config:
        schema_extra = {
            "example": {
                "request_id": "123e4567-e89b-12d3-a456-426614174000",
                "fixed_code": "def process_data(data: Any) -> Any:\n    \"\"\"Process the input data.\n\n    Args:\n        data: Input data to process\n\n    Returns:\n        Processed data\n    \"\"\"\n    return data",
                "fixes": [
                    {
                        "original_code": "def process_data(data):\n    return data",
                        "fixed_code": "def process_data(data: Any) -> Any:\n    \"\"\"Process the input data.\n\n    Args:\n        data: Input data to process\n\n    Returns:\n        Processed data\n    \"\"\"\n    return data",
                        "standard": "PEP 8",
                        "requirement": "Function documentation",
                        "description": "Added function docstring and type hints",
                        "improvement": "Improved code documentation and type safety"
                    }
                ],
                "summary": "Fixed 2 compliance issues across PEP 8 and ISO 27001 standards",
                "compliance_score_before": 65.0,
                "compliance_score_after": 95.0,
                "standard_specific_results": [
                    {
                        "standard": "PEP 8",
                        "compliant": True,
                        "issues_count": 0,
                        "notes": "All PEP 8 requirements met"
                    },
                    {
                        "standard": "ISO 27001",
                        "compliant": True,
                        "issues_count": 0,
                        "notes": "All ISO 27001 requirements met"
                    }
                ]
            }
        }

class BugFixRequest(BaseModel):
    """
    Request model for bug fix generation.
    
    Attributes:
        code: The code to fix bugs in
        language: Programming language of the code
        code_history: Optional history of code changes for context
        context: Additional context about the code
        fix_level: Level of fixes to apply ('basic', 'standard', 'comprehensive')
    """
    code: str = Field(..., description="Code to fix bugs in")
    language: ProgrammingLanguage = Field(..., description="Programming language of the code")
    code_history: Optional[List[Dict[str, str]]] = Field(None, description="History of code changes for context")
    context: Optional[str] = Field(None, description="Additional context about the code")
    fix_level: Optional[str] = Field("standard", description="Level of fixes to apply: 'basic', 'standard', or 'comprehensive'")
    
    class Config:
        schema_extra = {
            "example": {
                "code": "def process_data(data):\n    return data",
                "language": "python",
                "code_history": [
                    {
                        "description": "Initial implementation",
                        "date": "2023-01-01",
                        "author": "developer1"
                    }
                ],
                "context": "Data processing function",
                "fix_level": "comprehensive"
            }
        }

class BugFix(BaseModel):
    """
    Model for bug fixes applied to the code.
    
    Attributes:
        original_code: The original buggy code
        fixed_code: The fixed code
        bug_type: Type of bug that was fixed
        description: Description of the bug fix
        improvement: Description of how the code was improved
    """
    original_code: str = Field(..., description="The original buggy code")
    fixed_code: str = Field(..., description="The fixed code")
    bug_type: str = Field(..., description="Type of bug that was fixed")
    description: str = Field(..., description="Description of the bug fix")
    improvement: str = Field(..., description="Description of how the code was improved")
    
    class Config:
        schema_extra = {
            "example": {
                "original_code": "def process_data(data):\n    return data",
                "fixed_code": "def process_data(data: Any) -> Any:\n    \"\"\"Process the input data.\n\n    Args:\n        data: Input data to process\n\n    Returns:\n        Processed data\n    \"\"\"\n    if data is None:\n        raise ValueError(\"Data cannot be None\")\n    return data",
                "bug_type": "null_pointer",
                "description": "Added null check and type hints",
                "improvement": "Prevents null pointer exceptions and improves type safety"
            }
        }

class BugFixResponse(BaseModel):
    """
    Response model for bug fix results.
    
    Attributes:
        request_id: Unique identifier for the request
        fixed_code: Complete fixed code with all bugs resolved
        fixes: List of bug fixes applied
        summary: Summary of the bug fixes
        bug_score_before: Bug score before fixes (0-100)
        bug_score_after: Bug score after fixes (0-100)
        best_practices_applied: Best practices applied in fixes
    """
    request_id: str = Field(..., description="Unique identifier for the request")
    fixed_code: str = Field(..., description="Complete fixed code with all bugs resolved")
    fixes: List[BugFix] = Field(..., description="List of bug fixes applied")
    summary: str = Field(..., description="Summary of the bug fixes")
    bug_score_before: float = Field(..., description="Bug score before fixes (0-100)")
    bug_score_after: float = Field(..., description="Bug score after fixes (0-100)")
    best_practices_applied: List[str] = Field(..., description="Best practices applied in fixes")
    
    class Config:
        schema_extra = {
            "example": {
                "request_id": "123e4567-e89b-12d3-a456-426614174000",
                "fixed_code": "def process_data(data: Any) -> Any:\n    \"\"\"Process the input data.\n\n    Args:\n        data: Input data to process\n\n    Returns:\n        Processed data\n    \"\"\"\n    if data is None:\n        raise ValueError(\"Data cannot be None\")\n    return data",
                "fixes": [
                    {
                        "original_code": "def process_data(data):\n    return data",
                        "fixed_code": "def process_data(data: Any) -> Any:\n    \"\"\"Process the input data.\n\n    Args:\n        data: Input data to process\n\n    Returns:\n        Processed data\n    \"\"\"\n    if data is None:\n        raise ValueError(\"Data cannot be None\")\n    return data",
                        "bug_type": "null_pointer",
                        "description": "Added null check and type hints",
                        "improvement": "Prevents null pointer exceptions and improves type safety"
                    }
                ],
                "summary": "Fixed 1 critical bug and improved code quality",
                "bug_score_before": 65.0,
                "bug_score_after": 95.0,
                "best_practices_applied": ["Input validation", "Type hints", "Error handling"]
            }
        }

# Routes
@app.post("/code-review", 
    response_model=CodeReviewResponse,
    tags=["code-review"],
    summary="Perform automated code review",
    description="""
    Performs automated code review to identify issues and suggest improvements.
    
    ## Features
    - Code quality analysis
    - Security vulnerability detection
    - Style guide compliance
    - Best practices verification
    - Performance optimization
    
    ## Review Aspects
    - Code structure and organization
    - Error handling
    - Security vulnerabilities
    - Documentation quality
    - Performance considerations
    - Style guide compliance
    - Best practices adherence
    
    ## Example Use Cases
    - Pull request review
    - Code quality assessment
    - Security audit
    - Style guide compliance check
    - Performance optimization
    
    ## Request Format
    ```json
    {
        "code": "def calculate_sum(a, b):\n    return a + b",
        "language": "python",
        "review_focus": ["code style", "error handling"],
        "style_guide": "PEP 8",
        "context": "Utility function for basic arithmetic"
    }
    ```
    
    ## Response Format
    ```json
    {
        "request_id": "uuid",
        "issues": [
            {
                "level": "high",
                "type": "security",
                "line_number": 42,
                "description": "Issue description",
                "suggestion": "Fix suggestion",
                "code_snippet": "Relevant code"
            }
        ],
        "summary": "Review summary",
        "quality_score": 85.5,
        "best_practices": ["Best practice 1", "Best practice 2"],
        "improvement_areas": ["Area 1", "Area 2"]
    }
    ```
    """,
    response_description="Code review results with issues and suggestions"
)
async def review_code(
    request: CodeReviewRequest = Body(
        ...,
        example={
            "code": "def calculate_sum(a, b):\n    return a + b",
            "language": "python",
            "review_focus": ["code style", "error handling", "documentation"],
            "style_guide": "PEP 8",
            "context": "This is a utility function for basic arithmetic operations"
        }
    ),
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

@app.post("/security-scan", 
    response_model=SecurityScanResponse,
    tags=["security"],
    summary="Perform security vulnerability scanning",
    description="""
    Scans code for security vulnerabilities and suggests remediations.
    
    ## Features
    - Vulnerability detection
    - CWE identification
    - Remediation suggestions
    - Risk assessment
    - Dependency analysis
    
    ## Security Aspects
    - Code injection vulnerabilities
    - Authentication issues
    - Authorization problems
    - Data exposure
    - Cryptographic weaknesses
    - Dependency vulnerabilities
    
    ## Example Use Cases
    - Security audit
    - Vulnerability assessment
    - Secure code review
    - Dependency security check
    - Compliance verification
    
    ## Request Format
    ```json
    {
        "code": "def process_user_input(user_input):\n    return eval(user_input)",
        "language": "python",
        "security_level": "comprehensive",
        "application_context": "Web application",
        "dependencies": [
            {"name": "flask", "version": "2.0.1"}
        ]
    }
    ```
    
    ## Response Format
    ```json
    {
        "request_id": "uuid",
        "vulnerabilities": [
            {
                "level": "critical",
                "type": "code_injection",
                "line_number": 42,
                "description": "Vulnerability description",
                "cwe_id": "CWE-123",
                "remediation": "Fix suggestion",
                "code_snippet": "Vulnerable code"
            }
        ],
        "summary": "Scan summary",
        "risk_level": "high",
        "security_score": 75.0,
        "best_practices": ["Best practice 1", "Best practice 2"],
        "improvement_areas": ["Area 1", "Area 2"]
    }
    ```
    """,
    response_description="Security scan results with vulnerabilities and remediations"
)
async def scan_for_security_issues(
    request: SecurityScanRequest = Body(
        ...,
        example={
            "code": "def process_user_input(user_input):\n    return eval(user_input)",
            "language": "python",
            "security_level": "comprehensive",
            "application_context": "Web application handling user input",
            "dependencies": [
                {"name": "flask", "version": "2.0.1"},
                {"name": "sqlalchemy", "version": "1.4.23"}
            ]
        }
    ),
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

@app.post("/verify-documentation", 
    response_model=DocumentationVerificationResponse,
    tags=["documentation"],
    summary="Verify code documentation",
    description="""
    Verifies documentation for completeness, accuracy, and consistency with code.
    
    ## Features
    - Documentation completeness check
    - Accuracy verification
    - Code-documentation consistency
    - Best practices validation
    - Quality scoring
    
    ## Verification Aspects
    - Function documentation
    - Parameter descriptions
    - Return value documentation
    - Examples and usage
    - Code comments
    - API documentation
    
    ## Example Use Cases
    - Documentation quality check
    - API documentation verification
    - Code-documentation sync
    - Documentation standards compliance
    - Technical writing review
    
    ## Request Format
    ```json
    {
        "code": "def calculate_sum(a, b):\n    return a + b",
        "documentation": "This function adds two numbers together.",
        "language": "python",
        "verification_focus": ["completeness", "accuracy"]
    }
    ```
    
    ## Response Format
    ```json
    {
        "request_id": "uuid",
        "issues": [
            {
                "level": "medium",
                "type": "incomplete",
                "element": "function_name",
                "description": "Issue description",
                "suggestion": "Fix suggestion"
            }
        ],
        "summary": "Verification summary",
        "completeness_score": 85.0,
        "accuracy_score": 90.0,
        "missing_documentation": ["Element 1", "Element 2"],
        "outdated_documentation": ["Element 3", "Element 4"]
    }
    ```
    """,
    response_description="Documentation verification results with issues and scores"
)
async def verify_documentation(
    request: DocumentationVerificationRequest = Body(
        ...,
        example={
            "code": "def calculate_sum(a, b):\n    return a + b",
            "documentation": "This function adds two numbers together.",
            "language": "python",
            "verification_focus": ["completeness", "accuracy", "examples"]
        }
    ),
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

@app.post("/check-compliance", 
    response_model=ComplianceResponse,
    tags=["compliance"],
    summary="Check code compliance",
    description="""
    Checks code for compliance with specified standards or regulations.
    
    ## Features
    - Standards compliance verification
    - Regulation adherence check
    - Best practices validation
    - Industry standards compliance
    - Policy enforcement
    
    ## Compliance Aspects
    - Coding standards
    - Security standards
    - Accessibility guidelines
    - Industry regulations
    - Company policies
    
    ## Example Use Cases
    - Standards compliance check
    - Regulation verification
    - Policy enforcement
    - Industry standard compliance
    - Best practices validation
    
    ## Request Format
    ```json
    {
        "code": "def process_data(data):\n    return data",
        "language": "python",
        "standards": ["PEP 8", "ISO 27001"],
        "context": "Data processing function"
    }
    ```
    
    ## Response Format
    ```json
    {
        "request_id": "uuid",
        "compliance_issues": [
            {
                "standard": "PEP 8",
                "requirement": "Line length",
                "level": "low",
                "description": "Line exceeds 79 characters",
                "location": "Line 42",
                "remediation": "Break line into multiple lines"
            }
        ],
        "summary": "Code is mostly compliant with minor style issues",
        "overall_compliance": true,
        "compliance_score": 95.0,
        "standard_specific_results": [
            {
                "standard": "PEP 8",
                "compliant": true,
                "issues_count": 2,
                "notes": "Minor style issues found"
            }
        ]
    }
    ```
    """,
    response_description="Compliance check results with issues and scores"
)
async def check_compliance(
    request: ComplianceRequest = Body(
        ...,
        example={
            "code": "def process_data(data):\n    return data",
            "language": "python",
            "standards": ["PEP 8", "ISO 27001"],
            "context": "Data processing function"
        }
    ),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Check code for compliance with specified standards or regulations."""
    request_id = str(uuid.uuid4())
    
    try:
        logger.info(
            "Compliance check request received", 
            request_id, 
            {"language": request.language, "standards": request.standards}
        )
        
        # Create LLM client
        llm_client = LLMClient()
        
        # Build the system prompt
        system_prompt = f"""
        You are an expert compliance analyst specializing in {request.language.value} code.
        Check the provided code for compliance with the specified standards.
        For each standard, verify all requirements and identify any violations.
        Provide clear explanations and remediation steps for any compliance issues.
        """
        
        # Format standards for the prompt
        standards_str = "Standards to check for compliance:\n"
        for std in request.standards:
            standards_str += f"- {std}\n"
        
        # Build the user prompt
        prompt = f"""
        Please check this {request.language.value} code for compliance with the specified standards:
        
        ```{request.language.value}
        {request.code}
        ```
        
        {standards_str}
        
        {f'Context: {request.context}' if request.context else ''}
        
        Format your response as a valid JSON object with this structure:
        {{
            "compliance_issues": [
                {{
                    "standard": "Standard name",
                    "requirement": "Specific requirement",
                    "level": "critical/high/medium/low/info",
                    "description": "Description of the compliance issue",
                    "location": "Location in the code",
                    "remediation": "Steps to achieve compliance"
                }},
                ...
            ],
            "summary": "Summary of the compliance check",
            "overall_compliance": true/false,
            "compliance_score": 95.0,
            "standard_specific_results": [
                {{
                    "standard": "Standard name",
                    "compliant": true/false,
                    "issues_count": 2,
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
            
            return ComplianceResponse(
                request_id=request_id,
                compliance_issues=compliance_data.get("compliance_issues", []),
                summary=compliance_data.get("summary", ""),
                overall_compliance=compliance_data.get("overall_compliance", False),
                compliance_score=compliance_data.get("compliance_score", 0.0),
                standard_specific_results=compliance_data.get("standard_specific_results", [])
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing compliance check result: {str(e)}", request_id)
            raise HTTPException(status_code=500, detail=f"Error parsing compliance check result: {str(e)}")
            
    except Exception as e:
        logger.error(f"Error checking compliance: {str(e)}", request_id)
        raise HTTPException(status_code=500, detail=f"Error checking compliance: {str(e)}")

@app.post("/bug-prediction", 
    response_model=BugPredictionResponse,
    tags=["bug-prediction"],
    summary="Predict potential bugs",
    description="""
    Predicts potential bugs in code based on static analysis and patterns.
    
    ## Features
    - Bug pattern detection
    - Edge case analysis
    - Error handling verification
    - Resource management check
    - Concurrency issue detection
    
    ## Prediction Aspects
    - Common bug patterns
    - Edge cases
    - Error handling
    - Resource leaks
    - Race conditions
    - Memory issues
    
    ## Example Use Cases
    - Bug prevention
    - Code quality improvement
    - Error handling verification
    - Resource management check
    - Concurrency issue detection
    
    ## Request Format
    ```json
    {
        "code": "def process_data(data):\n    return data",
        "language": "python",
        "code_history": [
            {
                "description": "Initial implementation",
                "date": "2023-01-01",
                "author": "developer1"
            }
        ],
        "context": "Data processing function"
    }
    ```
    
    ## Response Format
    ```json
    {
        "request_id": "uuid",
        "predicted_bugs": [
            {
                "type": "null_pointer",
                "location": "Line 42",
                "probability": "high",
                "description": "Potential null pointer exception",
                "scenario": "When input is None",
                "prevention": "Add null check"
            }
        ],
        "summary": "Bug prediction summary",
        "high_risk_areas": ["Area 1", "Area 2"],
        "recommended_tests": ["Test 1", "Test 2"]
    }
    ```
    """,
    response_description="Bug prediction results with potential issues and recommendations"
)
async def predict_bugs(
    request: BugPredictionRequest = Body(
        ...,
        example={
            "code": "def process_data(data):\n    return data",
            "language": "python",
            "code_history": [
                {
                    "description": "Initial implementation",
                    "date": "2023-01-01",
                    "author": "developer1"
                }
            ],
            "context": "Data processing function"
        }
    ),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Predict potential bugs in code based on static analysis and patterns."""
    request_id = str(uuid.uuid4())
    
    try:
        logger.info(
            "Bug prediction request received", 
            request_id, 
            {"language": request.language}
        )
         
        # Create LLM client
        llm_client = LLMClient()
        
        # Build the system prompt
        system_prompt = f"""
        You are an expert bug predictor specializing in {request.language.value} code.
        Analyze the provided code to predict potential bugs that may occur.
        Focus on edge cases, error handling, resource management, concurrency issues, etc.
        Consider both obvious bugs and more subtle issues that might arise.
        """
        
        # Format code history if provided
        history_str = ""
        if request.code_history:
            history_str = "Code History (for context):\n"
            for i, entry in enumerate(request.code_history):
                history_str += f"Change {i+1}:\n"
                if 'description' in entry:
                    history_str += f"Description: {entry['description']}\n"
                if 'date' in entry:
                    history_str += f"Date: {entry['date']}\n"
                if 'author' in entry:
                    history_str += f"Author: {entry['author']}\n"
                history_str += "\n"
        
        # Build the user prompt
        prompt = f"""
        Please predict potential bugs in this {request.language.value} code:
        
        ```{request.language.value}
        {request.code}
        ```
        
        {history_str}
        
        {f'Context: {request.context}' if request.context else ''}
        
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
            
            return BugPredictionResponse(
                request_id=request_id,
                predicted_bugs=prediction_data.get("predicted_bugs", []),
                summary=prediction_data.get("summary", ""),
                high_risk_areas=prediction_data.get("high_risk_areas", []),
                recommended_tests=prediction_data.get("recommended_tests", [])
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing bug prediction result: {str(e)}", request_id)
            raise HTTPException(status_code=500, detail=f"Error parsing bug prediction result: {str(e)}")
            
    except Exception as e:
        logger.error(f"Error predicting bugs: {str(e)}", request_id)
        raise HTTPException(status_code=500, detail=f"Error predicting bugs: {str(e)}")

@app.post("/fix-security", 
    response_model=SecurityFixResponse,
    tags=["security"],
    summary="Fix security vulnerabilities in code",
    description="""
    Analyzes code for security vulnerabilities and generates a secure version with fixes applied.
    
    ## Features
    - Automatic vulnerability remediation
    - Secure code generation
    - Best practices implementation
    - Security score improvement
    - Code transformation
    
    ## Security Fixes
    - Code injection prevention
    - Input validation
    - Authentication strengthening
    - Authorization enforcement
    - Data protection
    - Secure defaults
    
    ## Example Use Cases
    - Security hardening
    - Vulnerability remediation
    - Secure code generation
    - Security best practices implementation
    - Code security improvement
    
    ## Request Format
    ```json
    {
        "code": "def process_user_input(user_input):\n    return eval(user_input)",
        "language": "python",
        "security_level": "comprehensive",
        "application_context": "Web application",
        "dependencies": [
            {"name": "flask", "version": "2.0.1"}
        ]
    }
    ```
    
    ## Response Format
    ```json
    {
        "request_id": "uuid",
        "fixed_code": "def process_user_input(user_input):\n    return ast.literal_eval(user_input)",
        "fixes": [
            {
                "original_code": "result = eval(user_input)",
                "fixed_code": "result = ast.literal_eval(user_input)",
                "vulnerability_type": "code_injection",
                "description": "Replaced unsafe eval() with ast.literal_eval()",
                "cwe_id": "CWE-94",
                "security_improvement": "Prevents arbitrary code execution"
            }
        ],
        "summary": "Fixed 1 critical security vulnerability",
        "security_score_before": 65.0,
        "security_score_after": 95.0,
        "best_practices_applied": ["Input validation", "Safe evaluation"]
    }
    ```
    """,
    response_description="Security fix results with improved code and applied fixes"
)
async def fix_security_issues(
    request: SecurityFixRequest = Body(
        ...,
        example={
            "code": "def process_user_input(user_input):\n    return eval(user_input)",
            "language": "python",
            "security_level": "comprehensive",
            "application_context": "Web application",
            "dependencies": [
                {"name": "flask", "version": "2.0.1"},
                {"name": "sqlalchemy", "version": "1.4.23"}
            ]
        }
    ),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Fix security vulnerabilities in code and generate a secure version."""
    request_id = str(uuid.uuid4())
    
    try:
        logger.info(
            "Security fix request received", 
            request_id, 
            {"language": request.language, "security_level": request.security_level}
        )
        
        # Create LLM client
        llm_client = LLMClient()
        
        # Build the system prompt
        system_prompt = f"""
        You are an expert security engineer specializing in {request.language.value} code security.
        Analyze the code for security vulnerabilities and generate a secure version with all issues fixed.
        Apply security best practices and ensure the fixes maintain the original functionality.
        For each fix, provide a clear explanation of the vulnerability and how it was remediated.
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
        Please fix security vulnerabilities in this {request.language.value} code:
        
        ```{request.language.value}
        {request.code}
        ```
        
        Security level: {request.security_level}
        
        {dependencies_str}
        
        {f'Application context: {request.application_context}' if request.application_context else ''}
        
        Format your response as a valid JSON object with this structure:
        {{
            "fixed_code": "Complete fixed code with all security issues resolved",
            "fixes": [
                {{
                    "original_code": "Vulnerable code snippet",
                    "fixed_code": "Fixed code snippet",
                    "vulnerability_type": "Type of vulnerability",
                    "description": "Description of the fix",
                    "cwe_id": "CWE-ID",
                    "security_improvement": "How security was improved"
                }},
                ...
            ],
            "summary": "Summary of security improvements",
            "security_score_before": 65.0,
            "security_score_after": 95.0,
            "best_practices_applied": [
                "Security best practice applied",
                ...
            ]
        }}
        """
        
        # Generate the security fixes
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
                fix_data = json.loads(json_str)
            else:
                # If no JSON found, try to parse the entire response
                fix_data = json.loads(result)
            
            return SecurityFixResponse(
                request_id=request_id,
                fixed_code=fix_data.get("fixed_code", request.code),
                fixes=fix_data.get("fixes", []),
                summary=fix_data.get("summary", ""),
                security_score_before=fix_data.get("security_score_before", 0.0),
                security_score_after=fix_data.get("security_score_after", 0.0),
                best_practices_applied=fix_data.get("best_practices_applied", [])
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing security fix result: {str(e)}", request_id)
            raise HTTPException(status_code=500, detail=f"Error parsing security fix result: {str(e)}")
            
    except Exception as e:
        logger.error(f"Error fixing security issues: {str(e)}", request_id)
        raise HTTPException(status_code=500, detail=f"Error fixing security issues: {str(e)}")

@app.post("/format-payload", 
    response_model=PayloadFormatResponse,
    tags=["utilities"],
    summary="Format code and dependencies for request payloads",
    description="""
    Formats raw code and dependencies file content into properly structured request payloads.
    
    ## Features
    - Code formatting
    - Dependencies parsing
    - Package detection
    - Version extraction
    - Request payload generation
    
    ## Supported Dependency Files
    - requirements.txt (Python)
    - package.json (Node.js)
    - Cargo.toml (Rust)
    - pom.xml (Java)
    - build.gradle (Gradle)
    - Gemfile (Ruby)
    - composer.json (PHP)
    
    ## Example Use Cases
    - API request preparation
    - Dependency analysis
    - Package management
    - Version tracking
    - Integration setup
    
    ## Request Format
    ```json
    {
        "code": "def process_data(data):\\n    return eval(data)",
        "dependencies_file": "flask==2.0.1\\nsqlalchemy==1.4.23\\nrequests>=2.25.0",
        "language": "python",
        "file_type": "requirements.txt"
    }
    ```
    
    ## Response Format
    ```json
    {
        "request_id": "uuid",
        "formatted_code": "def process_data(data):\\n    return eval(data)",
        "formatted_dependencies": [
            {"name": "flask", "version": "2.0.1"},
            {"name": "sqlalchemy", "version": "1.4.23"},
            {"name": "requests", "version": "2.25.0"}
        ],
        "detected_packages": ["flask", "sqlalchemy", "requests"],
        "package_versions": {
            "flask": "2.0.1",
            "sqlalchemy": "1.4.23",
            "requests": "2.25.0"
        }
    }
    ```
    """,
    response_description="Formatted code and dependencies for request payloads"
)
async def format_payload(
    request: PayloadFormatRequest = Body(
        ...,
        example={
            "code": """def process_data(data):
    return eval(data)""",
            "dependencies_file": "flask==2.0.1\nsqlalchemy==1.4.23\nrequests>=2.25.0",
            "language": "python",
            "file_type": "requirements.txt"
        }
    ),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Format code and dependencies into request payload format."""
    request_id = str(uuid.uuid4())
    
    try:
        logger.info(
            "Payload format request received", 
            request_id, 
            {"language": request.language, "file_type": request.file_type}
        )
        
        # Format the code by normalizing line endings and handling indentation
        code_lines = request.code.replace('\r\n', '\n').replace('\r', '\n').split('\n')
        
        # Remove leading/trailing empty lines while preserving internal empty lines
        while code_lines and not code_lines[0].strip():
            code_lines.pop(0)
        while code_lines and not code_lines[-1].strip():
            code_lines.pop()
            
        # Detect and normalize indentation
        indent_size = None
        for line in code_lines:
            if line.strip():  # Only check non-empty lines
                spaces = len(line) - len(line.lstrip())
                if spaces > 0:
                    if indent_size is None or spaces < indent_size:
                        indent_size = spaces
        
        # Normalize indentation if detected
        if indent_size:
            normalized_lines = []
            for line in code_lines:
                if line.strip():  # Only normalize non-empty lines
                    # Ensure we don't remove more spaces than we have
                    leading_spaces = len(line) - len(line.lstrip())
                    if leading_spaces >= indent_size:
                        # Remove indentation in multiples of indent_size
                        spaces_to_remove = (leading_spaces // indent_size) * indent_size
                        normalized_lines.append(line[spaces_to_remove:])
                    else:
                        normalized_lines.append(line)
                else:
                    normalized_lines.append(line)
            formatted_code = '\n'.join(normalized_lines)
        else:
            formatted_code = '\n'.join(code_lines)
        
        # Initialize dependency parsing results
        formatted_dependencies = []
        detected_packages = []
        package_versions = {}
        
        # Parse dependencies based on file type
        if request.file_type == "requirements.txt":
            # Normalize line endings in dependencies file
            deps_lines = request.dependencies_file.replace('\r\n', '\n').replace('\r', '\n').split('\n')
            
            # Parse Python requirements.txt
            for line in deps_lines:
                line = line.strip()
                if line and not line.startswith("#"):
                    # Handle different requirement formats
                    if "==" in line:
                        name, version = line.split("==")
                    elif ">=" in line:
                        name, version = line.split(">=")
                    elif "<=" in line:
                        name, version = line.split("<=")
                    elif ">" in line:
                        name, version = line.split(">")
                    elif "<" in line:
                        name, version = line.split("<")
                    else:
                        name = line
                        version = "latest"
                    
                    name = name.strip()
                    version = version.strip()
                    
                    formatted_dependencies.append({
                        "name": name,
                        "version": version
                    })
                    detected_packages.append(name)
                    package_versions[name] = version
                    
        elif request.file_type == "package.json":
            # Parse Node.js package.json
            try:
                package_data = json.loads(request.dependencies_file)
                dependencies = {
                    **package_data.get("dependencies", {}),
                    **package_data.get("devDependencies", {})
                }
                
                for name, version in dependencies.items():
                    # Clean version string (remove ^, ~, etc.)
                    clean_version = version.replace("^", "").replace("~", "")
                    if clean_version.startswith(">="):
                        clean_version = clean_version[2:]
                    elif clean_version.startswith(">"):
                        clean_version = clean_version[1:]
                    
                    formatted_dependencies.append({
                        "name": name,
                        "version": clean_version
                    })
                    detected_packages.append(name)
                    package_versions[name] = clean_version
                    
            except json.JSONDecodeError as e:
                raise HTTPException(status_code=400, detail=f"Invalid package.json format: {str(e)}")
                
        else:
            # Add support for other dependency file types as needed
            logger.warning(f"Unsupported dependency file type: {request.file_type}", request_id)
        
        return PayloadFormatResponse(
            request_id=request_id,
            formatted_code=formatted_code,
            formatted_dependencies=formatted_dependencies,
            detected_packages=detected_packages,
            package_versions=package_versions
        )
        
    except Exception as e:
        logger.error(f"Error formatting payload: {str(e)}", request_id)
        raise HTTPException(status_code=500, detail=f"Error formatting payload: {str(e)}")

@app.post("/fix-documentation", 
    response_model=DocumentationFixResponse,
    tags=["documentation"],
    summary="Fix documentation issues in code",
    description="""
    Analyzes code for documentation issues and generates a properly documented version.
    
    ## Features
    - Automatic documentation generation
    - Type hint addition
    - Docstring creation
    - Best practices implementation
    - Documentation score improvement
    
    ## Documentation Aspects
    - Function docstrings
    - Class docstrings
    - Method docstrings
    - Parameter descriptions
    - Return value documentation
    - Type hints
    - Code comments
    - Usage examples
    
    ## Example Use Cases
    - Documentation improvement
    - Code documentation generation
    - API documentation creation
    - Type hint addition
    - Best practices implementation
    
    ## Request Format
    ```json
    {
        "code": "def calculate_sum(a, b):\\n    return a + b",
        "language": "python",
        "documentation_level": "comprehensive",
        "context": "Utility function",
        "style_guide": "Google Python Style Guide"
    }
    ```
    
    ## Response Format
    ```json
    {
        "request_id": "uuid",
        "fixed_code": "def calculate_sum(a: float, b: float) -> float:\\n    \"\"\"Calculate the sum of two numbers...\"\"\"\\n    return a + b",
        "fixes": [
            {
                "original_code": "def calculate_sum(a, b):\\n    return a + b",
                "fixed_code": "def calculate_sum(a: float, b: float) -> float:\\n    \"\"\"Calculate...\"\"\"\\n    return a + b",
                "element": "calculate_sum function",
                "description": "Added docstring and type hints",
                "improvement": "Added parameter and return documentation"
            }
        ],
        "summary": "Added comprehensive documentation",
        "doc_score_before": 45.0,
        "doc_score_after": 95.0,
        "best_practices_applied": ["Type hints", "Function docstrings"]
    }
    ```
    """,
    response_description="Documentation fix results with improved code and applied fixes"
)
async def fix_documentation_issues(
    request: DocumentationFixRequest = Body(
        ...,
        example={
            "code": "def calculate_sum(a, b):\n    return a + b",
            "language": "python",
            "documentation_level": "comprehensive",
            "context": "Utility function for basic arithmetic operations",
            "style_guide": "Google Python Style Guide"
        }
    ),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Fix documentation issues in code and generate a properly documented version."""
    request_id = str(uuid.uuid4())
    
    try:
        logger.info(
            "Documentation fix request received", 
            request_id, 
            {"language": request.language, "documentation_level": request.documentation_level}
        )
        
        # Create LLM client
        llm_client = LLMClient()
        
        # Build the system prompt
        system_prompt = f"""
        You are an expert documentation engineer specializing in {request.language.value} code.
        Analyze the code for documentation issues and generate a properly documented version.
        Apply documentation best practices and ensure the documentation is clear, complete, and accurate.
        For each fix, provide a clear explanation of what was improved and how.
        {f'Follow the {request.style_guide} documentation style.' if request.style_guide else ''}
        """
        
        # Build the user prompt
        prompt = f"""
        Please fix documentation issues in this {request.language.value} code:
        
        ```{request.language.value}
        {request.code}
        ```
        
        Documentation level: {request.documentation_level}
        
        {f'Style guide: {request.style_guide}' if request.style_guide else ''}
        
        {f'Context: {request.context}' if request.context else ''}
        
        Format your response as a valid JSON object with this structure:
        {{
            "fixed_code": "Complete fixed code with proper documentation",
            "fixes": [
                {{
                    "original_code": "Code snippet before fix",
                    "fixed_code": "Code snippet after fix",
                    "element": "Code element that was fixed",
                    "description": "Description of the fix",
                    "improvement": "How documentation was improved"
                }},
                ...
            ],
            "summary": "Summary of documentation improvements",
            "doc_score_before": 45.0,
            "doc_score_after": 95.0,
            "best_practices_applied": [
                "Documentation best practice applied",
                ...
            ]
        }}
        """
        
        # Generate the documentation fixes
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
                fix_data = json.loads(json_str)
            else:
                # If no JSON found, try to parse the entire response
                fix_data = json.loads(result)
            
            return DocumentationFixResponse(
                request_id=request_id,
                fixed_code=fix_data.get("fixed_code", request.code),
                fixes=fix_data.get("fixes", []),
                summary=fix_data.get("summary", ""),
                doc_score_before=fix_data.get("doc_score_before", 0.0),
                doc_score_after=fix_data.get("doc_score_after", 0.0),
                best_practices_applied=fix_data.get("best_practices_applied", [])
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing documentation fix result: {str(e)}", request_id)
            raise HTTPException(status_code=500, detail=f"Error parsing documentation fix result: {str(e)}")
            
    except Exception as e:
        logger.error(f"Error fixing documentation issues: {str(e)}", request_id)
        raise HTTPException(status_code=500, detail=f"Error fixing documentation issues: {str(e)}")

@app.post("/fix-compliance", 
    response_model=ComplianceFixResponse,
    tags=["compliance"],
    summary="Fix compliance issues in code",
    description="""
    Analyzes code for compliance issues and generates a compliant version.
    
    ## Features
    - Automatic compliance remediation
    - Standards adherence
    - Best practices implementation
    - Compliance score improvement
    - Code transformation
    
    ## Compliance Aspects
    - Coding standards
    - Security standards
    - Accessibility guidelines
    - Industry regulations
    - Company policies
    
    ## Example Use Cases
    - Standards compliance
    - Regulation adherence
    - Policy enforcement
    - Industry standard compliance
    - Best practices implementation
    
    ## Request Format
    ```json
    {
        "code": "def process_data(data):\\n    return data",
        "language": "python",
        "standards": ["PEP 8", "ISO 27001"],
        "context": "Data processing function",
        "fix_level": "comprehensive"
    }
    ```
    
    ## Response Format
    ```json
    {
        "request_id": "uuid",
        "fixed_code": "def process_data(data: Any) -> Any:\\n    \"\"\"Process the input data...\"\"\"\\n    return data",
        "fixes": [
            {
                "original_code": "def process_data(data):\\n    return data",
                "fixed_code": "def process_data(data: Any) -> Any:\\n    \"\"\"Process...\"\"\"\\n    return data",
                "standard": "PEP 8",
                "requirement": "Function documentation",
                "description": "Added docstring and type hints",
                "improvement": "Improved documentation"
            }
        ],
        "summary": "Fixed compliance issues",
        "compliance_score_before": 65.0,
        "compliance_score_after": 95.0,
        "standard_specific_results": [
            {
                "standard": "PEP 8",
                "compliant": true,
                "issues_count": 0,
                "notes": "All requirements met"
            }
        ]
    }
    ```
    """,
    response_description="Compliance fix results with improved code and applied fixes"
)
async def fix_compliance_issues(
    request: ComplianceFixRequest = Body(
        ...,
        example={
            "code": "def process_data(data):\n    return data",
            "language": "python",
            "standards": ["PEP 8", "ISO 27001"],
            "context": "Data processing function",
            "fix_level": "comprehensive"
        }
    ),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Fix compliance issues in code and generate a compliant version."""
    request_id = str(uuid.uuid4())
    
    try:
        logger.info(
            "Compliance fix request received", 
            request_id, 
            {"language": request.language, "standards": request.standards, "fix_level": request.fix_level}
        )
        
        # Create LLM client
        llm_client = LLMClient()
        
        # Build the system prompt
        system_prompt = f"""
        You are an expert compliance engineer specializing in {request.language.value} code.
        Analyze the code for compliance issues and generate a compliant version.
        Apply compliance best practices and ensure the fixes maintain the original functionality.
        For each fix, provide a clear explanation of the compliance issue and how it was remediated.
        Focus on the following standards: {", ".join(request.standards)}
        """
        
        # Build the user prompt
        prompt = f"""
        Please fix compliance issues in this {request.language.value} code:
        
        ```{request.language.value}
        {request.code}
        ```
        
        Standards to comply with: {", ".join(request.standards)}
        Fix level: {request.fix_level}
        
        {f'Context: {request.context}' if request.context else ''}
        
        Format your response as a valid JSON object with this structure:
        {{
            "fixed_code": "Complete fixed code with all compliance issues resolved",
            "fixes": [
                {{
                    "original_code": "Non-compliant code snippet",
                    "fixed_code": "Fixed code snippet",
                    "standard": "Standard name",
                    "requirement": "Specific requirement",
                    "description": "Description of the fix",
                    "improvement": "How compliance was improved"
                }},
                ...
            ],
            "summary": "Summary of compliance improvements",
            "compliance_score_before": 65.0,
            "compliance_score_after": 95.0,
            "standard_specific_results": [
                {{
                    "standard": "Standard name",
                    "compliant": true/false,
                    "issues_count": 0,
                    "notes": "Notes about compliance"
                }},
                ...
            ]
        }}
        """
        
        # Generate the compliance fixes
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
                fix_data = json.loads(json_str)
            else:
                # If no JSON found, try to parse the entire response
                fix_data = json.loads(result)
            
            return ComplianceFixResponse(
                request_id=request_id,
                fixed_code=fix_data.get("fixed_code", request.code),
                fixes=fix_data.get("fixes", []),
                summary=fix_data.get("summary", ""),
                compliance_score_before=fix_data.get("compliance_score_before", 0.0),
                compliance_score_after=fix_data.get("compliance_score_after", 0.0),
                standard_specific_results=fix_data.get("standard_specific_results", [])
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing compliance fix result: {str(e)}", request_id)
            raise HTTPException(status_code=500, detail=f"Error parsing compliance fix result: {str(e)}")
            
    except Exception as e:
        logger.error(f"Error fixing compliance issues: {str(e)}", request_id)
        raise HTTPException(status_code=500, detail=f"Error fixing compliance issues: {str(e)}")

@app.post("/fix-bug", 
    response_model=BugFixResponse,
    tags=["bug-prediction"],
    summary="Fix bugs in code",
    description="""
    Analyzes code for potential bugs and generates a bug-free version.
    
    ## Features
    - Automatic bug remediation
    - Code quality improvement
    - Best practices implementation
    - Bug score improvement
    - Code transformation
    
    ## Bug Fixes
    - Null pointer prevention
    - Input validation
    - Error handling
    - Resource management
    - Concurrency issues
    - Memory leaks
    
    ## Example Use Cases
    - Bug remediation
    - Code quality improvement
    - Error handling implementation
    - Resource management
    - Concurrency fixes
    
    ## Request Format
    ```json
    {
        "code": "def process_data(data):\\n    return data",
        "language": "python",
        "code_history": [
            {
                "description": "Initial implementation",
                "date": "2023-01-01",
                "author": "developer1"
            }
        ],
        "context": "Data processing function",
        "fix_level": "comprehensive"
    }
    ```
    
    ## Response Format
    ```json
    {
        "request_id": "uuid",
        "fixed_code": "def process_data(data: Any) -> Any:\\n    \"\"\"Process the input data...\"\"\"\\n    if data is None:\\n        raise ValueError(\"Data cannot be None\")\\n    return data",
        "fixes": [
            {
                "original_code": "def process_data(data):\\n    return data",
                "fixed_code": "def process_data(data: Any) -> Any:\\n    \"\"\"Process...\"\"\"\\n    if data is None:\\n        raise ValueError(\"Data cannot be None\")\\n    return data",
                "bug_type": "null_pointer",
                "description": "Added null check and type hints",
                "improvement": "Prevents null pointer exceptions"
            }
        ],
        "summary": "Fixed 1 critical bug",
        "bug_score_before": 65.0,
        "bug_score_after": 95.0,
        "best_practices_applied": ["Input validation", "Type hints"]
    }
    ```
    """,
    response_description="Bug fix results with improved code and applied fixes"
)
async def fix_bugs(
    request: BugFixRequest = Body(
        ...,
        example={
            "code": "def process_data(data):\n    return data",
            "language": "python",
            "code_history": [
                {
                    "description": "Initial implementation",
                    "date": "2023-01-01",
                    "author": "developer1"
                }
            ],
            "context": "Data processing function",
            "fix_level": "comprehensive"
        }
    ),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Fix bugs in code and generate a bug-free version."""
    request_id = str(uuid.uuid4())
    
    try:
        logger.info(
            "Bug fix request received", 
            request_id, 
            {"language": request.language, "fix_level": request.fix_level}
        )
        
        # Create LLM client
        llm_client = LLMClient()
        
        # Build the system prompt
        system_prompt = f"""
        You are an expert bug fixer specializing in {request.language.value} code.
        Analyze the code for potential bugs and generate a bug-free version.
        Apply best practices and ensure the fixes maintain the original functionality.
        For each fix, provide a clear explanation of the bug and how it was remediated.
        Focus on common bug patterns like null pointers, input validation, error handling, etc.
        """
        
        # Format code history if provided
        history_str = ""
        if request.code_history:
            history_str = "Code History (for context):\n"
            for i, entry in enumerate(request.code_history):
                history_str += f"Change {i+1}:\n"
                if 'description' in entry:
                    history_str += f"Description: {entry['description']}\n"
                if 'date' in entry:
                    history_str += f"Date: {entry['date']}\n"
                if 'author' in entry:
                    history_str += f"Author: {entry['author']}\n"
                history_str += "\n"
        
        # Build the user prompt
        prompt = f"""
        Please fix bugs in this {request.language.value} code:
        
        ```{request.language.value}
        {request.code}
        ```
        
        Fix level: {request.fix_level}
        
        {history_str}
        
        {f'Context: {request.context}' if request.context else ''}
        
        Format your response as a valid JSON object with this structure:
        {{
            "fixed_code": "Complete fixed code with all bugs resolved",
            "fixes": [
                {{
                    "original_code": "Buggy code snippet",
                    "fixed_code": "Fixed code snippet",
                    "bug_type": "Type of bug",
                    "description": "Description of the fix",
                    "improvement": "How the code was improved"
                }},
                ...
            ],
            "summary": "Summary of bug fixes",
            "bug_score_before": 65.0,
            "bug_score_after": 95.0,
            "best_practices_applied": [
                "Best practice applied",
                ...
            ]
        }}
        """
        
        # Generate the bug fixes
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
                fix_data = json.loads(json_str)
            else:
                # If no JSON found, try to parse the entire response
                fix_data = json.loads(result)
            
            return BugFixResponse(
                request_id=request_id,
                fixed_code=fix_data.get("fixed_code", request.code),
                fixes=fix_data.get("fixes", []),
                summary=fix_data.get("summary", ""),
                bug_score_before=fix_data.get("bug_score_before", 0.0),
                bug_score_after=fix_data.get("bug_score_after", 0.0),
                best_practices_applied=fix_data.get("best_practices_applied", [])
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing bug fix result: {str(e)}", request_id)
            raise HTTPException(status_code=500, detail=f"Error parsing bug fix result: {str(e)}")
            
    except Exception as e:
        logger.error(f"Error fixing bugs: {str(e)}", request_id)
        raise HTTPException(status_code=500, detail=f"Error fixing bugs: {str(e)}")
