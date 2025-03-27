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

- **Python**: Core programming language
- **FastAPI**: Web framework for building APIs
- **Pydantic**: Data validation and settings management
- **SQLAlchemy**: Database ORM (Object-Relational Mapping)
- **Claude Sonnet 3.5** or **GPT-4o**: Large Language Models (LLMs)

The architecture follows a modular design where each service is implemented as a separate FastAPI application, which are then combined in the main application.

## Shared Utilities

The platform includes several shared utilities for common functionality:

- **LLM Client**: Generic client for interacting with different LLM providers
- **Structured Logger**: Consistent logging across services
- **Database**: Common database setup and utilities
- **Authentication**: JWT-based authentication
- **Prompt Templates**: Reusable prompt templates for different tasks

## Setup and Deployment

### Prerequisites

- Python 3.8+
- API keys for either OpenAI (GPT-4o) or Anthropic (Claude Sonnet 3.5)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/generative-ai-services.git
   cd generative-ai-services
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set environment variables:
   ```bash
   export OPENAI_API_KEY=your_openai_api_key
   export ANTHROPIC_API_KEY=your_anthropic_api_key
   export SECRET_KEY=your_jwt_secret_key
   export DATABASE_URL=your_database_url
   ```

4. Run the application:
   ```bash
   python main.py
   ```

### Docker Deployment

1. Build the Docker image:
   ```bash
   docker build -t generative-ai-services .
   ```

2. Run the Docker container:
   ```bash
   docker run -p 8000:8000 \
     -e OPENAI_API_KEY=your_openai_api_key \
     -e ANTHROPIC_API_KEY=your_anthropic_api_key \
     -e SECRET_KEY=your_jwt_secret_key \
     -e DATABASE_URL=your_database_url \
     generative-ai-services
   ```

## API Documentation

Once the server is running, access the interactive API documentation at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Extending the Platform

To add a new service:

1. Create a new directory under `services/`
2. Implement your service as a FastAPI application
3. Add your service router to `main.py`
4. Update the service list in the root endpoint in `main.py`

## Security Considerations

- **API Keys**: Securely manage your LLM provider API keys
- **Authentication**: Implement proper authentication for production use
- **Rate Limiting**: Consider adding rate limiting for production deployment
- **Content Filtering**: Implement content filtering for user-generated inputs
- **Data Privacy**: Be mindful of data privacy regulations when processing user data

## License

[MIT License](LICENSE)