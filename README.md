# Generative AI Services Platform

A comprehensive platform offering reusable AI capabilities as services to help software engineering teams efficiently adopt generative AI into their applications.

## Overview

This platform provides a collection of eight core services, each exposing multiple endpoints that leverage LLMs (Large Language Models) for different tasks. The services are designed to be general-purpose and reusable across various domains and use cases.

## Services

### 1. Code Generation and Transformation

Generate, document, refactor, translate, and test code using LLMs.

- **Endpoints**:
  - `/code/generate`: Generate code from natural language specifications
  - `/code/document`: Generate documentation for existing code
  - `/code/refactor`: Refactor code according to specified goals
  - `/code/translate`: Translate code from one programming language to another
  - `/code/generate-tests`: Generate test cases for existing code

### 2. Document Extraction

Extract structured information from various document types.

- **Endpoints**:
  - `/document-extraction/extract`: Extract information from a document
  - `/document-extraction/batch-extract`: Extract information from multiple documents
  - `/document-extraction/extract-from-text`: Extract information from plain text

### 3. Classification and Categorization

Classify and categorize content according to specified categories.

- **Endpoints**:
  - `/classification/classify`: Classify content according to specified categories
  - `/classification/batch-classify`: Classify multiple content items
  - `/classification/classify-tabular`: Classify data from a specific column in a tabular file

### 4. Conversational Interfaces

Create domain-specific chatbots, Q&A systems, and manage multi-turn dialogues.

- **Endpoints**:
  - `/conversation/chat`: Generate responses in a conversation
  - `/conversation/summarize`: Summarize a conversation history
  - `/conversation/analyze-dialogue`: Analyze a conversation for intents, topics, etc.
  - `/conversation/multi-turn-qa`: Multi-turn question answering with knowledge base
  - `/conversation/domain-chat`: Domain-specific chatbot responses
  - `/conversation/contextual-chat`: Context-aware response generation
  - `/conversation/dialogue-management`: Manage complex dialogue flows

### 5. Search and Retrieval

Perform semantic search, compare documents, and query knowledge bases.

- **Endpoints**:
  - `/search/search`: Search documents using semantic or keyword search
  - `/search/semantic-search`: Semantic search using LLM for relevance determination
  - `/search/compare-documents`: Compare two documents and analyze similarities/differences
  - `/search/query-knowledge-base`: Query a knowledge base and generate answers
  - `/search/query-reformulation`: Reformulate search queries to improve results

### 6. Workflow Automation

Classify tickets, analyze requirements, identify dependencies, and automate workflows.

- **Endpoints**:
  - `/workflow/classify-ticket`: Classify support tickets and determine routing, priority, etc.
  - `/workflow/analyze-requirements`: Analyze software requirements and identify issues
  - `/workflow/identify-dependencies`: Identify dependencies between components
  - `/workflow/suggest-workflow`: Suggest workflow improvements
  - `/workflow/analyze-change-impact`: Analyze impact of proposed changes

### 7. Content Personalization

Personalize content based on user preferences, generate A/B test variations, and localize content.

- **Endpoints**:
  - `/personalization/personalize`: Personalize content based on user profile
  - `/personalization/generate-ab-test`: Generate variations of content for A/B testing
  - `/personalization/localize`: Localize content for a specific target locale
  - `/personalization/personalize-recommendation`: Generate personalized recommendations

### 8. Quality Assurance

Review code, scan for security issues, verify documentation, and predict bugs.

- **Endpoints**:
  - `/quality/code-review`: Review code and identify issues
  - `/quality/security-scan`: Scan code for security vulnerabilities
  - `/quality/verify-documentation`: Verify documentation completeness and accuracy
  - `/quality/check-compliance`: Check code for compliance with standards
  - `/quality/bug-prediction`: Predict potential bugs in code

## Architecture

The platform is built using:

- **Python 3.12**: Core programming language
- **FastAPI**: Web framework for building APIs
- **Pydantic**: Data validation and settings management
- **SQLAlchemy**: Database ORM (Object-Relational Mapping)
- **PostgreSQL**: Database for persistent storage
- **AWS Managed Services**: For scalable, production-ready deployment
  - **Amazon Cognito**: For user authentication
  - **Amazon DynamoDB**: For storing conversation states, configurations, etc.
  - **Amazon S3**: For storing files and large content
  - **Amazon Comprehend**: For natural language processing tasks
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