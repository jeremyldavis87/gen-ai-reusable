# API Documentation

## Overview

This document provides detailed information about the APIs exposed by the Gen AI Services platform. Each service exposes a RESTful API with standard HTTP methods and JSON payloads.

## Authentication

### JWT Authentication

```http
Authorization: Bearer <jwt_token>
```

### API Key Authentication

```http
X-API-Key: <api_key>
```

## Format Conversion Service

### Base URL
```
http://localhost:8001/api/v1
```

### Endpoints

#### Convert Document Format

```http
POST /convert
Content-Type: application/json

{
  "document_id": "string",
  "source_format": "string",
  "target_format": "string",
  "options": {
    "quality": "high|medium|low",
    "preserve_layout": boolean
  }
}
```

Response:
```json
{
  "conversion_id": "string",
  "status": "pending|completed|failed",
  "result_url": "string",
  "error": "string"
}
```

#### Get Conversion Status

```http
GET /convert/{conversion_id}
```

Response:
```json
{
  "conversion_id": "string",
  "status": "pending|completed|failed",
  "result_url": "string",
  "error": "string"
}
```

## Classification Service

### Base URL
```
http://localhost:8002/api/v1
```

### Endpoints

#### Classify Document

```http
POST /classify
Content-Type: application/json

{
  "document_id": "string",
  "document_type": "string",
  "content": "string",
  "metadata": {
    "source": "string",
    "timestamp": "string"
  }
}
```

Response:
```json
{
  "classification_id": "string",
  "categories": [
    {
      "name": "string",
      "confidence": float,
      "metadata": {}
    }
  ],
  "status": "completed|failed",
  "error": "string"
}
```

#### Get Classification

```http
GET /classify/{classification_id}
```

Response:
```json
{
  "classification_id": "string",
  "categories": [
    {
      "name": "string",
      "confidence": float,
      "metadata": {}
    }
  ],
  "status": "completed|failed",
  "error": "string"
}
```

## Workflow Service

### Base URL
```
http://localhost:8003/api/v1
```

### Endpoints

#### Create Workflow

```http
POST /workflows
Content-Type: application/json

{
  "name": "string",
  "description": "string",
  "steps": [
    {
      "service": "string",
      "action": "string",
      "parameters": {}
    }
  ]
}
```

Response:
```json
{
  "workflow_id": "string",
  "status": "created|running|completed|failed",
  "current_step": "string",
  "error": "string"
}
```

#### Get Workflow Status

```http
GET /workflows/{workflow_id}
```

Response:
```json
{
  "workflow_id": "string",
  "status": "created|running|completed|failed",
  "current_step": "string",
  "steps": [
    {
      "service": "string",
      "action": "string",
      "status": "pending|completed|failed",
      "error": "string"
    }
  ]
}
```

## Search Service

### Base URL
```
http://localhost:8004/api/v1
```

### Endpoints

#### Search Documents

```http
POST /search
Content-Type: application/json

{
  "query": "string",
  "filters": {
    "document_type": "string",
    "date_range": {
      "start": "string",
      "end": "string"
    }
  },
  "page": integer,
  "page_size": integer
}
```

Response:
```json
{
  "results": [
    {
      "document_id": "string",
      "title": "string",
      "snippet": "string",
      "metadata": {}
    }
  ],
  "total": integer,
  "page": integer,
  "page_size": integer
}
```

## Quality Service

### Base URL
```
http://localhost:8005/api/v1
```

### Endpoints

#### Analyze Quality

```http
POST /analyze
Content-Type: application/json

{
  "document_id": "string",
  "document_type": "string",
  "content": "string",
  "metrics": ["completeness", "accuracy", "consistency"]
}
```

Response:
```json
{
  "analysis_id": "string",
  "metrics": {
    "completeness": float,
    "accuracy": float,
    "consistency": float
  },
  "issues": [
    {
      "type": "string",
      "severity": "high|medium|low",
      "description": "string"
    }
  ],
  "status": "completed|failed",
  "error": "string"
}
```

## Personalization Service

### Base URL
```
http://localhost:8006/api/v1
```

### Endpoints

#### Get Recommendations

```http
POST /recommendations
Content-Type: application/json

{
  "user_id": "string",
  "context": {
    "document_type": "string",
    "interests": ["string"],
    "history": ["string"]
  },
  "limit": integer
}
```

Response:
```json
{
  "recommendations": [
    {
      "document_id": "string",
      "title": "string",
      "relevance_score": float,
      "metadata": {}
    }
  ],
  "total": integer
}
```

## Document Extraction Service

### Base URL
```
http://localhost:8007/api/v1
```

### Endpoints

#### Extract Information

```http
POST /extract
Content-Type: application/json

{
  "document_id": "string",
  "document_type": "string",
  "content": "string",
  "fields": ["string"]
}
```

Response:
```json
{
  "extraction_id": "string",
  "fields": {
    "field_name": "string",
    "value": "string",
    "confidence": float
  },
  "status": "completed|failed",
  "error": "string"
}
```

## Conversational Service

### Base URL
```
http://localhost:8008/api/v1
```

### Endpoints

#### Process Query

```http
POST /query
Content-Type: application/json

{
  "query": "string",
  "context": {
    "user_id": "string",
    "session_id": "string",
    "history": ["string"]
  }
}
```

Response:
```json
{
  "response": "string",
  "suggestions": ["string"],
  "confidence": float,
  "metadata": {}
}
```

## Code Service

### Base URL
```
http://localhost:8009/api/v1
```

### Endpoints

#### Analyze Code

```http
POST /analyze
Content-Type: application/json

{
  "code": "string",
  "language": "string",
  "analysis_type": ["complexity", "security", "style"]
}
```

Response:
```json
{
  "analysis_id": "string",
  "results": {
    "complexity": {
      "score": float,
      "issues": ["string"]
    },
    "security": {
      "score": float,
      "vulnerabilities": ["string"]
    },
    "style": {
      "score": float,
      "violations": ["string"]
    }
  },
  "status": "completed|failed",
  "error": "string"
}
```

## Error Handling

### Error Response Format

```json
{
  "error": {
    "code": "string",
    "message": "string",
    "details": {}
  }
}
```

### Common Error Codes

- 400: Bad Request
- 401: Unauthorized
- 403: Forbidden
- 404: Not Found
- 429: Too Many Requests
- 500: Internal Server Error

## Rate Limiting

- Rate limit: 100 requests per minute per API key
- Rate limit headers:
  ```
  X-RateLimit-Limit: 100
  X-RateLimit-Remaining: 95
  X-RateLimit-Reset: 1612345678
  ```

## Versioning

- API version included in URL path
- Current version: v1
- Version header also supported:
  ```
  Accept: application/vnd.genai.v1+json
  ```

## Webhooks

### Webhook Events

- document.converted
- document.classified
- workflow.completed
- quality.analyzed

### Webhook Payload

```json
{
  "event": "string",
  "timestamp": "string",
  "data": {}
}
```

## SDK Examples

### Python

```python
from genai_sdk import GenAIClient

client = GenAIClient(api_key="your_api_key")

# Convert document
result = client.format_service.convert(
    document_id="doc123",
    source_format="pdf",
    target_format="docx"
)

# Classify document
result = client.classification_service.classify(
    document_id="doc123",
    content="document content"
)
```

### JavaScript

```javascript
const { GenAIClient } = require('genai-sdk');

const client = new GenAIClient({
  apiKey: 'your_api_key'
});

// Convert document
const result = await client.formatService.convert({
  documentId: 'doc123',
  sourceFormat: 'pdf',
  targetFormat: 'docx'
});

// Classify document
const result = await client.classificationService.classify({
  documentId: 'doc123',
  content: 'document content'
});
``` 