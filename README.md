# Generative AI Services Platform

A comprehensive platform offering reusable AI capabilities as services to help software engineering teams efficiently adopt generative AI into their applications.

## Overview

This platform provides a collection of nine core services, each exposing multiple endpoints that leverage LLMs (Large Language Models) for different tasks. The services are designed to be general-purpose and reusable across various domains and use cases.

## Services

### 1. Code Generation and Transformation

Leverage LLMs to automate various aspects of the software development lifecycle, from initial code creation to maintenance and optimization.

This service utilizes advanced prompt engineering techniques to generate high-quality code, documentation, and tests. It supports multiple programming languages and can be customized for specific coding standards and patterns.

**Use Cases:**
- Rapid prototyping of new features based on natural language specifications
- Automated documentation generation for legacy codebases
- Cross-language code translation for migration projects
- Code refactoring to improve performance or readability
- Test generation to increase code coverage

- **Endpoints**:
  - `/code/generate`: Generate code from natural language specifications
  - `/code/document`: Generate documentation for existing code
  - `/code/refactor`: Refactor code according to specified goals
  - `/code/translate`: Translate code from one programming language to another
  - `/code/generate-tests`: Generate test cases for existing code

### 2. Document Extraction

Transform unstructured documents into structured, machine-readable data by extracting key information using advanced NLP techniques.

This service processes various document formats (PDF, DOCX, images via OCR) and extracts structured information based on specified schemas. It combines LLM capabilities with rule-based extraction for optimal accuracy.

**Use Cases:**
- Automated processing of invoices, receipts, and financial documents
- Extraction of key information from legal contracts and agreements
- Conversion of technical documentation into structured knowledge bases
- Processing of forms and applications for automated workflow systems
- Data extraction from research papers and technical literature

- **Endpoints**:
  - `/document-extraction/extract`: Extract information from a document
  - `/document-extraction/batch-extract`: Extract information from multiple documents
  - `/document-extraction/extract-from-text`: Extract information from plain text

### 3. Classification and Categorization

Automatically classify and categorize content according to customizable taxonomies and classification schemes.

This service uses fine-tuned language models to accurately classify content across multiple dimensions. It supports both single-label and multi-label classification, with confidence scores for each category.

**Use Cases:**
- Content moderation for user-generated content platforms
- Automatic tagging and categorization of articles and documents
- Sentiment analysis for customer feedback and social media monitoring
- Topic classification for knowledge management systems
- Intent recognition for customer support systems

- **Endpoints**:
  - `/classification/classify`: Classify content according to specified categories
  - `/classification/batch-classify`: Classify multiple content items
  - `/classification/classify-tabular`: Classify data from a specific column in a tabular file

### 4. Conversational Interfaces

Build sophisticated conversational experiences with domain-specific knowledge, multi-turn dialogue capabilities, and contextual awareness.

This service provides a complete framework for creating conversational AI applications, from simple chatbots to complex dialogue systems with memory and reasoning capabilities. It includes features for conversation state management, knowledge retrieval, and dialogue analysis.

**Use Cases:**
- Customer support automation with domain-specific knowledge
- Interactive documentation and knowledge base interfaces
- Virtual assistants for internal enterprise applications
- Educational tutoring systems with multi-turn dialogue capabilities
- Conversational interfaces for complex workflows and processes

- **Endpoints**:
  - `/conversation/chat`: Generate responses in a conversation
  - `/conversation/summarize`: Summarize a conversation history
  - `/conversation/analyze-dialogue`: Analyze a conversation for intents, topics, etc.
  - `/conversation/multi-turn-qa`: Multi-turn question answering with knowledge base
  - `/conversation/domain-chat`: Domain-specific chatbot responses
  - `/conversation/contextual-chat`: Context-aware response generation
  - `/conversation/dialogue-management`: Manage complex dialogue flows

### 5. Format Conversion Service

Transform unstructured text into structured, formatted outputs according to specified instructions and schemas.

This service converts raw text and documents into various structured formats using advanced LLM capabilities. It handles texts of any size through intelligent chunking and can process both simple and complex conversion tasks with high accuracy.

**Use Cases:**
- Converting unstructured business documents into structured data formats
- Transforming legacy data formats into modern standards
- Extracting structured information from research papers and technical literature
- Converting between different data serialization formats
- Creating structured datasets from raw text for machine learning applications

- **Endpoints**:
  - `/convert`: Convert unstructured text to a specified format
  - `/convert/file`: Convert an uploaded file to a specified format
  - `/convert/stream`: Stream the conversion process for large texts with progress updates

### 6. Search and Retrieval

Implement advanced semantic search capabilities that understand the meaning and context of queries beyond simple keyword matching.

This service combines vector embeddings, LLM-based relevance scoring, and traditional search techniques to provide highly relevant search results. It includes features for document comparison, query reformulation, and knowledge base querying.

**Use Cases:**
- Enhanced enterprise search for internal knowledge bases and documentation
- Semantic search for customer-facing documentation and support
- Document similarity analysis for legal and compliance applications
- Intelligent query understanding for e-commerce and product search
- Research assistance for scientific and academic applications

- **Endpoints**:
  - `/search/search`: Search documents using semantic or keyword search
  - `/search/semantic-search`: Semantic search using LLM for relevance determination
  - `/search/compare-documents`: Compare two documents and analyze similarities/differences
  - `/search/query-knowledge-base`: Query a knowledge base and generate answers
  - `/search/query-reformulation`: Reformulate search queries to improve results

### 7. Workflow Automation

Streamline and optimize business processes by automating the analysis, classification, and routing of information through workflows.

This service applies AI to common workflow challenges, helping to classify incoming requests, analyze requirements, identify dependencies, and suggest process improvements. It integrates with existing workflow systems through APIs.

**Use Cases:**
- Automated ticket classification and routing for IT support
- Requirements analysis and validation for software development
- Change impact analysis for infrastructure and application changes
- Dependency identification for project planning and risk assessment
- Workflow optimization and bottleneck identification

- **Endpoints**:
  - `/workflow/classify-ticket`: Classify support tickets and determine routing, priority, etc.
  - `/workflow/analyze-requirements`: Analyze software requirements and identify issues
  - `/workflow/identify-dependencies`: Identify dependencies between components
  - `/workflow/suggest-workflow`: Suggest workflow improvements
  - `/workflow/analyze-change-impact`: Analyze impact of proposed changes

### 8. Content Personalization

Deliver tailored content experiences by adapting information based on user preferences, behavior patterns, and contextual factors.

This service enables dynamic content personalization through user preference modeling, A/B testing capabilities, and localization features. It can integrate with existing user management systems to leverage user profile data.

**Use Cases:**
- Personalized documentation and help content based on user expertise level
- Dynamic website content adaptation based on user interests and behavior
- A/B testing of content variations for marketing and product messaging
- Localization and cultural adaptation of content for global audiences
- Personalized learning experiences for educational platforms

- **Endpoints**:
  - `/personalization/personalize`: Personalize content based on user profile
  - `/personalization/generate-ab-test`: Generate variations of content for A/B testing
  - `/personalization/localize`: Localize content for a specific target locale
  - `/personalization/personalize-recommendation`: Generate personalized recommendations

### 9. Quality Assurance

Enhance software quality through AI-powered code review, security scanning, documentation verification, and bug prediction.

This service applies LLMs and specialized models to various aspects of software quality assurance, helping teams identify issues early and maintain high-quality standards. It integrates with existing development workflows and CI/CD pipelines.

**Supported Languages:**
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

**Quality Aspects:**
- Code Style and Standards
- Security Vulnerabilities
- Documentation Quality
- Performance Issues
- Error Handling
- Test Coverage
- Maintainability
- Best Practices

**Use Cases:**
- Automated code review for quality and best practices
- Security vulnerability scanning for early detection of issues
- Documentation completeness and accuracy verification
- Compliance checking against industry or organizational standards
- Predictive bug detection based on code patterns and history
- Automatic security vulnerability remediation
- Code transformation for improved security

**Endpoints:**

**Code Review:**
  - `/quality/code-review`: Review code and identify issues

**Security:**
  - `/quality/security-scan`: Scan code for security vulnerabilities
  - `/quality/security-fix`: Automatically fix security vulnerabilities in code

**Documentation:**
  - `/quality/verify-documentation`: Verify documentation completeness and accuracy
  - `/quality/fix-documentation`: Fix documentation issues in code

**Compliance:**
  - `/quality/check-compliance`: Check code for compliance with standards
  - `/quality/fix-compliance`: Fix compliance issues in code

**Bug Prediction:**
  - `/quality/bug-prediction`: Predict potential bugs in code
  - `/quality/fix-bug`: Fix bugs in code

**Utilities:**
  - `/quality/format-payload`: Format code and dependencies for request payloads

**Best Practices:**
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

## Architecture

The platform is built using:

- **Python 3.12**: Core programming language
- **FastAPI**: Web framework for building APIs
- **Pydantic**: Data validation and settings management
- **SQLAlchemy**: Database ORM (Object-Relational Mapping)
- **PostgreSQL**: Database for persistent storage
- **AWS Managed Services**: For scalable, production-ready deployment
  - **Amazon DynamoDB**: For storing conversation states, configurations, etc.
  - **Amazon S3**: For storing files and large content
  - **Amazon RDS Aurora Serverless**: For PostgreSQL database
- **Claude Sonnet 3.5** or **GPT-4o**: Large Language Models (LLMs)

The architecture follows a modular design where each service is implemented as a separate FastAPI application, which are then combined in the main application. This allows for independent scaling and deployment of services as needed.

## Shared Utilities

The platform includes several shared utilities for common functionality:

- **LLM Client**: Generic client for interacting with different LLM providers
- **Structured Logger**: Consistent logging across services
- **Database**: Common database setup and utilities
- **Authentication**: JWT-based authentication
- **Prompt Templates**: Reusable prompt templates for different tasks

## Setup and Deployment

### Prerequisites

- Python 3.12
- API keys for either OpenAI (GPT-4o) or Anthropic (Claude Sonnet 3.5)
- AWS account with appropriate permissions
- Docker for containerization
- Terraform and Terragrunt for infrastructure management

### Local Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/gen-ai-reusable.git
   cd gen-ai-reusable
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Create a `.env` file with required environment variables:
   ```
   # AWS Configuration
   AWS_REGION=us-east-1
   AWS_ACCESS_KEY_ID=your_access_key
   AWS_SECRET_ACCESS_KEY=your_secret_key

   # Database Configuration
   DATABASE_URL=postgresql://user:password@localhost/db_name

   # LLM API Keys
   OPENAI_API_KEY=your_openai_api_key
   ANTHROPIC_API_KEY=your_anthropic_api_key

   # DynamoDB Tables
   DOMAIN_CONFIG_TABLE=domain_configurations
   KNOWLEDGE_BASE_TABLE=knowledge_base_entries
   CONVERSATIONS_TABLE=conversations
   DIALOGUE_STATES_TABLE=dialogue_states

   # S3 Buckets
   S3_BUCKET=your-s3-bucket-name
   ```

4. Run the application:
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

### Local Development

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up environment variables by copying the sample file:
   ```bash
   cp .env_sample .env
   # Edit .env with your configuration
   ```

3. Run a specific service locally using the run_local.py script:
   ```bash
   # Run the Format Conversion Service
   python run_local.py format
   
   # Run with auto-reload for development
   python run_local.py format --reload
   
   # Specify host and port
   python run_local.py format --host 127.0.0.1 --port 8080
   ```

### Docker Deployment

1. Build the Docker image:
   ```bash
   docker build -t gen-ai-reusable .
   ```

2. Run the Docker container:
   ```bash
   docker run -p 8000:8000 --env-file .env gen-ai-reusable
   ```

### AWS Deployment

1. Set up AWS infrastructure using Terraform:
   ```bash
   cd terraform
   terraform init
   terraform apply
   ```

2. Deploy the application using GitHub Actions:
   - Push your code to GitHub
   - GitHub Actions will automatically build and deploy the application

## Usage Examples

### Conversational Interfaces Service

#### Basic Chat

```python
import requests
import json

response = requests.post(
    "http://localhost:8000/conversation/chat",
    headers={
        "Authorization": "Bearer YOUR_TOKEN",
        "Content-Type": "application/json"
    },
    json={
        "conversation_type": "chat",
        "messages": [
            {"role": "user", "content": "Hello, how can you help me today?"}
        ]
    }
)

print(json.dumps(response.json(), indent=2))
```

#### Domain-Specific Chat

```python
response = requests.post(
    "http://localhost:8000/conversation/domain-chat",
    headers={
        "Authorization": "Bearer YOUR_TOKEN",
        "Content-Type": "application/json"
    },
    json={
        "domain": "healthcare",
        "conversation_type": "domain_specific",
        "messages": [
            {"role": "user", "content": "What are the symptoms of the flu?"}
        ]
    }
)

print(json.dumps(response.json(), indent=2))
```

#### Multi-Turn Question Answering

```python
response = requests.post(
    "http://localhost:8000/conversation/multi-turn-qa",
    headers={
        "Authorization": "Bearer YOUR_TOKEN",
        "Content-Type": "application/json"
    },
    json={
        "query": "What's the capital of France?",
        "conversation_history": [
            {"question": "Tell me about France", "answer": "France is a country in Western Europe..."}
        ],
        "knowledge_base": [
            {
                "id": "kb1",
                "title": "France Facts",
                "content": "Paris is the capital of France. France is known for its cuisine and culture."
            }
        ]
    }
)

print(json.dumps(response.json(), indent=2))
```

#### Conversation Summarization

```python
response = requests.post(
    "http://localhost:8000/conversation/summarize",
    headers={
        "Authorization": "Bearer YOUR_TOKEN",
        "Content-Type": "application/json"
    },
    json={
        "conversation_id": "12345",
        "messages": [
            {"role": "user", "content": "I need help setting up my account."},
            {"role": "assistant", "content": "I'd be happy to help. What specific issue are you having?"},
            {"role": "user", "content": "I can't reset my password. The reset link isn't being sent to my email."},
            {"role": "assistant", "content": "Let me check that for you. Can you confirm your email address?"},
            {"role": "user", "content": "It's user@example.com"},
            {"role": "assistant", "content": "Thank you. I've manually triggered a password reset. Please check your email in the next 5 minutes."}
        ]
    }
)

print(json.dumps(response.json(), indent=2))
```

### Code Generation Service

```python
response = requests.post(
    "http://localhost:8000/code/generate",
    headers={
        "Authorization": "Bearer YOUR_TOKEN",
        "Content-Type": "application/json"
    },
    json={
        "specifications": "Create a Python function that calculates the Fibonacci sequence up to n terms",
        "language": "python",
        "include_tests": True
    }
)

print(json.dumps(response.json(), indent=2))
```

### Document Extraction Service

```python
import base64

# Read PDF file as base64
with open("document.pdf", "rb") as f:
    pdf_base64 = base64.b64encode(f.read()).decode("utf-8")

response = requests.post(
    "http://localhost:8000/document-extraction/extract",
    headers={
        "Authorization": "Bearer YOUR_TOKEN",
        "Content-Type": "application/json"
    },
    json={
        "document": {
            "content": pdf_base64,
            "mime_type": "application/pdf",
            "filename": "document.pdf"
        },
        "extraction_config": {
            "fields": ["invoice_number", "date", "total_amount", "vendor_name"],
            "output_format": "json"
        }
    }
)

print(json.dumps(response.json(), indent=2))
```

## API Documentation

Once the server is running, access the interactive API documentation at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Extending the Platform

### Adding a New Service

1. Create a new directory under `services/`:
   ```bash
   mkdir services/new_service
   ```

2. Create a `main.py` file in the new directory with a FastAPI application:
   ```python
   from fastapi import APIRouter, Depends, HTTPException
   from typing import Dict, Any
   from utilities.auth import get_current_user
   
   router = APIRouter(prefix="/new-service", tags=["new-service"])
   
   @router.post("/endpoint")
   async def new_endpoint(
       request_data: Dict[str, Any],
       current_user: Dict[str, Any] = Depends(get_current_user)
   ):
       # Implement your endpoint logic here
       return {"result": "Success"}
   ```

3. Add your service router to the main `main.py` file:
   ```python
   from services.new_service.main import router as new_service_router
   
   app.include_router(new_service_router)
   ```

4. Update the service list in the README.md

### Adding AWS Resources

1. Define new AWS resources in the Terraform configuration:
   ```hcl
   # terraform/main.tf
   resource "aws_dynamodb_table" "new_service_table" {
     name         = "new-service-table"
     billing_mode = "PAY_PER_REQUEST"
     hash_key     = "id"
     
     attribute {
       name = "id"
       type = "S"
     }
     
     tags = {
       Service = "new-service"
     }
   }
   ```

2. Apply the Terraform changes:
   ```bash
   terraform apply
   ```

## Security Considerations

- **API Keys**: Securely manage your LLM provider API keys using AWS Secrets Manager
- **Authentication**: Use Amazon Cognito for secure user authentication
- **Rate Limiting**: Implement rate limiting using API Gateway
- **Content Filtering**: Implement content filtering for user-generated inputs
- **Data Privacy**: Be mindful of data privacy regulations when processing user data
- **Encryption**: Ensure all data is encrypted at rest and in transit
- **IAM Roles**: Use least privilege IAM roles for AWS services

## Monitoring and Logging

- **CloudWatch**: Monitor application performance and logs
- **X-Ray**: Trace requests through the application
- **Structured Logging**: Use the built-in structured logger for consistent logging

## License

[MIT License](LICENSE)