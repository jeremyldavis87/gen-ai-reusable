# utilities/prompts.py

"""Prompt templates for LLM interactions."""

class PromptTemplates:
    """Collection of prompt templates for various tasks."""
    
    @staticmethod
    def document_extraction(document_text: str, items_to_extract: list, output_format: str) -> str:
        """Generate prompt for document extraction."""
        return f"""
Extract the following information from the document:
{', '.join(items_to_extract)}

Output the extracted information in {output_format} format.

Document text:
{document_text}
"""
    
    @staticmethod
    def categorization(content: str, categories: list, instructions: str) -> str:
        """Generate prompt for content categorization."""
        return f"""
Categories: {', '.join(categories)}

Instructions: {instructions}

Content to categorize:
{content}
"""
    
    @staticmethod
    def code_generation(specifications: str) -> str:
        """Generate prompt for code generation."""
        return f"""
Generate code based on the following specifications:

{specifications}

Provide clean, well-documented code that follows best practices.
"""
    
    @staticmethod
    def code_documentation(code: str) -> str:
        """Generate a prompt for code documentation.
        
        Args:
            code: Code to document
            
        Returns:
            Documentation prompt
        """
        return f"""
        Please analyze this code and provide comprehensive documentation:
        
        ```
        {code}
        ```
        
        Include:
        1. Overview of the code's purpose and functionality
        2. Detailed documentation for each function/class/method
        3. Parameters, return values, and their types
        4. Usage examples
        5. Any important notes or caveats
        
        Follow best practices for documentation style and clarity.
        """
    
    @staticmethod
    def data_cleaning(data_sample: str, cleaning_instructions: str) -> str:
        """Generate prompt for data cleaning instructions."""
        return f"""
Data sample:
{data_sample}

Generate data cleaning instructions based on the following requirements:
{cleaning_instructions}
"""
    
    @staticmethod
    def summarization(content: str, summary_type: str, length: str) -> str:
        """Generate prompt for content summarization."""
        return f"""
Create a {summary_type} summary of the following content. The summary should be {length}.

Content to summarize:
{content}
"""
    
    @staticmethod
    def conversation_generation(context: str, query: str) -> str:
        """Generate prompt for conversational response generation."""
        return f"""
Context information:
{context}

User query: {query}

Generate a helpful, conversational response that addresses the user's query based on the context information.
"""
    
    @staticmethod
    def semantic_search(query: str, documents: list) -> str:
        """Generate prompt for semantic search."""
        docs_text = "\n\n".join([f"Document {i+1}:\n{doc}" for i, doc in enumerate(documents)])
        return f"""
Query: {query}

Rank the following documents by relevance to the query and explain why each is relevant or not relevant:

{docs_text}
"""
    
    @staticmethod
    def workflow_automation(context: str, task_description: str) -> str:
        """Generate prompt for workflow automation."""
        return f"""
Context:
{context}

Task description:
{task_description}

Analyze the workflow requirements and suggest an automated process with specific steps and tools.
"""
    
    @staticmethod
    def personalization(user_data: str, content_to_personalize: str) -> str:
        """Generate prompt for content personalization."""
        return f"""
User data:
{user_data}

Content to personalize:
{content_to_personalize}

Generate a personalized version of the content based on the user data.
"""
    
    @staticmethod
    def code_quality(code: str, quality_criteria: str) -> str:
        """Generate prompt for code quality analysis."""
        return f"""
Code to analyze:
```
{code}
```

Quality criteria to evaluate:
{quality_criteria}

Provide a detailed analysis of the code quality based on the specified criteria.
"""
    
    @staticmethod
    def code_review(code: str, context: str = "") -> str:
        """Generate a prompt for code review.
        
        Args:
            code: Code to review
            context: Additional context for the review
            
        Returns:
            Code review prompt
        """
        return f"""
        Please review this code for quality, best practices, and potential issues:
        
        {f'Context: {context}' if context else ''}
        
        ```
        {code}
        ```
        
        Consider:
        1. Code style and readability
        2. Performance optimizations
        3. Security concerns
        4. Error handling
        5. Testing considerations
        6. Documentation completeness
        
        Provide specific recommendations for improvements.
        """
    
    @staticmethod
    def error_analysis(error_message: str, code: str, context: str = "") -> str:
        """Generate a prompt for error analysis.
        
        Args:
            error_message: Error message to analyze
            code: Code that produced the error
            context: Additional context about the error
            
        Returns:
            Error analysis prompt
        """
        return f"""
        Please analyze this error and provide debugging suggestions:
        
        Error Message:
        {error_message}
        
        {f'Context: {context}' if context else ''}
        
        Code:
        ```
        {code}
        ```
        
        Please provide:
        1. Root cause analysis
        2. Step-by-step debugging approach
        3. Potential solutions
        4. Prevention strategies
        """
    
    @staticmethod
    def test_generation(code: str, test_framework: str = "pytest") -> str:
        """Generate a prompt for test case generation.
        
        Args:
            code: Code to generate tests for
            test_framework: Testing framework to use
            
        Returns:
            Test generation prompt
        """
        return f"""
        Please generate comprehensive test cases for this code using {test_framework}:
        
        ```
        {code}
        ```
        
        Include tests for:
        1. Normal operation (happy path)
        2. Edge cases and boundary conditions
        3. Error conditions and exception handling
        4. Integration points
        5. Performance considerations
        
        Follow testing best practices and provide clear test documentation.
        """
    
    @staticmethod
    def code_optimization(code: str, optimization_goals: str) -> str:
        """Generate a prompt for code optimization.
        
        Args:
            code: Code to optimize
            optimization_goals: Specific optimization goals
            
        Returns:
            Code optimization prompt
        """
        return f"""
        Please optimize this code according to these goals:
        {optimization_goals}
        
        Original code:
        ```
        {code}
        ```
        
        Consider:
        1. Time complexity
        2. Space complexity
        3. Resource usage
        4. Readability and maintainability
        5. Best practices for the language/framework
        
        Provide the optimized code and explain the improvements made.
        """
    
    @staticmethod
    def security_review(code: str, context: str = "") -> str:
        """Generate a prompt for security review.
        
        Args:
            code: Code to review for security
            context: Additional context about the code
            
        Returns:
            Security review prompt
        """
        return f"""
        Please perform a security review of this code:
        
        {f'Context: {context}' if context else ''}
        
        ```
        {code}
        ```
        
        Check for:
        1. Input validation and sanitization
        2. Authentication and authorization
        3. Data encryption and protection
        4. Common vulnerabilities (XSS, CSRF, SQL injection, etc.)
        5. Secure configuration and environment handling
        6. Dependency security
        
        Provide specific security recommendations and fixes.
        """
