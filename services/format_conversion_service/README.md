# Format Conversion Service

## Overview

The Format Conversion Service transforms unstructured text into structured, formatted outputs according to specified instructions. It supports multiple output formats including JSON, YAML, CSV, XML, Markdown, HTML, and SQL.

## Features

- Convert unstructured text to structured formats based on detailed instructions
- Support for multiple output formats (JSON, YAML, CSV, XML, Markdown, HTML, SQL)
- Automatic chunking for large documents to handle texts of any size
- Streaming API for real-time progress updates on large conversions
- Schema validation and enforcement for structured outputs
- File upload support for processing documents directly

## AWS Integration

- **Amazon Bedrock**: Used for LLM-based text structuring and formatting
- **Amazon S3**: Stores conversion templates and large input/output documents
- **Amazon RDS Aurora PostgreSQL**: Caches common conversion patterns for improved performance
- **Amazon CloudWatch**: Monitors service performance and logs conversion activities
- **Amazon Cognito**: Handles user authentication for secure access to the service

## Endpoints

### `/convert`

Converts unstructured text to a specified format.

**Method**: POST

**Request Body**:
```json
{
  "text": "The unstructured text to convert",
  "instructions": "Instructions for how to structure the output",
  "output_format": "json|yaml|csv|xml|markdown|html|sql",
  "schema": {"optional": "schema definition"},
  "max_tokens": 4000,
  "temperature": 0.2
}
```

**Response**:
```json
{
  "conversion_id": "unique-id",
  "output": "The formatted output",
  "format": "json",
  "timestamp": "2025-03-27T15:08:09"
}
```

### `/convert/file`

Converts an uploaded file to a specified format.

**Method**: POST

**Request**: Multipart form data with:
- `instructions`: Instructions for how to structure the output
- `output_format`: The desired output format
- `schema`: Optional schema definition
- `max_tokens`: Maximum tokens for LLM response (default: 4000)
- `temperature`: Temperature for LLM generation (default: 0.2)
- `file`: The file to convert

**Response**: Same as `/convert`

### `/convert/stream`

Streams the conversion process for large texts with progress updates.

**Method**: POST

**Request Body**: Same as `/convert`

**Response**: Server-sent events with progress updates and chunks of converted output

## Usage Examples

### Converting Unstructured Text to JSON

```python
import requests

response = requests.post(
    "https://api.example.com/convert",
    json={
        "text": "Product: Widget X\nPrice: $19.99\nDescription: A versatile widget for all your needs\nFeatures: Durable, Lightweight, Versatile",
        "instructions": "Extract product information including name, price, description, and features. For features, create an array of individual features.",
        "output_format": "json"
    },
    headers={"Authorization": "Bearer YOUR_API_KEY"}
)

print(response.json())
```

**Response**:
```json
{
  "conversion_id": "550e8400-e29b-41d4-a716-446655440000",
  "output": {
    "product": "Widget X",
    "price": "$19.99",
    "description": "A versatile widget for all your needs",
    "features": ["Durable", "Lightweight", "Versatile"]
  },
  "format": "json",
  "timestamp": "2025-03-27T15:08:09"
}
```

### Converting a Document to CSV

```python
import requests

files = {
    'file': open('meeting_notes.txt', 'rb')
}

data = {
    'instructions': 'Extract action items from the meeting notes. Include columns for task, assignee, due date, and priority.',
    'output_format': 'csv'
}

response = requests.post(
    "https://api.example.com/convert/file",
    files=files,
    data=data,
    headers={"Authorization": "Bearer YOUR_API_KEY"}
)

print(response.json())
```

**Response**:
```json
{
  "conversion_id": "550e8400-e29b-41d4-a716-446655440001",
  "output": "Task,Assignee,Due Date,Priority\nUpdate documentation,John,2025-04-01,High\nFix login bug,Sarah,2025-03-30,Critical\nPrepare demo,Michael,2025-04-05,Medium",
  "format": "csv",
  "timestamp": "2025-03-27T15:10:22"
}
```

## Environment Variables

- `AWS_REGION`: AWS region for services
- `AWS_ACCESS_KEY_ID`: AWS access key
- `AWS_SECRET_ACCESS_KEY`: AWS secret key
- `DATABASE_URL`: PostgreSQL connection string
- `S3_BUCKET`: S3 bucket for document storage
- `BEDROCK_MODEL_ID`: Amazon Bedrock model ID for text processing

## Deployment

This service is designed to be deployed as a Docker container on AWS infrastructure. It can be deployed using AWS Amplify Gen 2 for the front-end components and integrated with other AWS managed services.

```bash
# Build the Docker image
docker build -t format-conversion-service -f services/format_conversion_service/Dockerfile .

# Run the container locally
docker run -p 8000:8000 \
  -e AWS_REGION=us-west-2 \
  -e AWS_ACCESS_KEY_ID=your_access_key \
  -e AWS_SECRET_ACCESS_KEY=your_secret_key \
  -e DATABASE_URL=postgresql://user:password@host:port/dbname \
  -e S3_BUCKET=your-s3-bucket \
  -e BEDROCK_MODEL_ID=anthropic.claude-v2 \
  format-conversion-service
```

## Security Considerations

- All API endpoints are protected with Amazon Cognito authentication
- API keys and credentials are stored securely using environment variables
- Input validation is performed to prevent injection attacks
- Rate limiting is applied to prevent abuse
