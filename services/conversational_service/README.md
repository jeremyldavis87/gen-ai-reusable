# Conversational Interfaces Service

A reusable AWS-powered service for building conversational AI applications with domain-specific chatbots, question answering systems, contextual response generation, conversation summarization, and multi-turn dialogue management.

## Features

- **Domain-specific chatbot frameworks**: Create specialized chatbots for different domains with custom knowledge bases
- **Question answering systems**: Retrieve precise answers from knowledge bases with context awareness
- **Contextual response generation**: Generate responses based on conversation history and context
- **Conversation summarization**: Extract key points, action items, and sentiment from conversations
- **Multi-turn dialogue management**: Manage complex conversation flows with state transitions

## AWS Integration

This service leverages several AWS managed services:

- **Amazon Comprehend**: For entity recognition, sentiment analysis, and key phrase extraction
- **Amazon DynamoDB**: For storing conversation history, states, and domain configurations
- **Amazon S3**: For storing knowledge base content, conversation summaries, and analysis results
- **Amazon Cognito**: For user authentication (configured separately)

## API Endpoints

- `/chat`: Standard chat endpoint for general conversational interactions
- `/domain-chat`: Domain-specific chatbot endpoint with specialized knowledge
- `/contextual-chat`: Context-aware chat endpoint that maintains conversation state
- `/summarize`: Summarize conversation history with key points and action items
- `/multi-turn-qa`: Specialized Q&A endpoint with knowledge base integration
- `/dialogue-management`: Manage state transitions in complex dialogue flows
- `/analyze-dialogue`: Analyze conversations for intents, topics, and sentiment

## Setup

### Prerequisites

- Python 3.12
- AWS account with appropriate permissions
- PostgreSQL database (for user management)

### Environment Variables

Create a `.env` file with the following variables:

```
# AWS Configuration
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key

# Database Configuration
DATABASE_URL=postgresql://user:password@localhost/db_name

# DynamoDB Tables
DOMAIN_CONFIG_TABLE=domain_configurations
KNOWLEDGE_BASE_TABLE=knowledge_base_entries
CONVERSATIONS_TABLE=conversations
DIALOGUE_STATES_TABLE=dialogue_states

# S3 Buckets
S3_BUCKET=conversation-service-data

# LLM API Keys
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
```

### Installation

```bash
pip install -r requirements.txt
```

### Running the Service

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Deployment

This service is designed to be deployed using Docker and AWS infrastructure:

1. Build the Docker image
2. Deploy to AWS using Terraform/Terragrunt
3. Configure AWS Cognito for authentication
4. Set up necessary DynamoDB tables and S3 buckets

## Usage Examples

### Basic Chat

```python
import requests
import json

response = requests.post(
    "http://localhost:8000/chat",
    headers={"Authorization": "Bearer YOUR_TOKEN"},
    json={
        "conversation_type": "chat",
        "messages": [
            {"role": "user", "content": "Hello, how can you help me today?"}
        ]
    }
)

print(json.dumps(response.json(), indent=2))
```

### Domain-Specific Chat

```python
response = requests.post(
    "http://localhost:8000/domain-chat",
    headers={"Authorization": "Bearer YOUR_TOKEN"},
    json={
        "conversation_type": "domain_specific",
        "domain": "healthcare",
        "messages": [
            {"role": "user", "content": "What are the symptoms of the flu?"}
        ]
    }
)
```

### Conversation Summarization

```python
response = requests.post(
    "http://localhost:8000/summarize",
    headers={"Authorization": "Bearer YOUR_TOKEN"},
    json={
        "conversation_id": "12345",
        "messages": [
            # Array of message objects
        ]
    }
)
```
