# Format Conversion Service

## Overview

The Format Conversion Service is responsible for converting documents between different formats while maintaining content integrity and formatting. It supports a wide range of document formats and provides high-quality conversion capabilities.

## Features

- Document format conversion
- Format preservation
- Batch processing
- Quality control
- Progress tracking
- Error handling
- Format validation

## Supported Formats

### Input Formats
- PDF (Portable Document Format)
- DOCX (Microsoft Word)
- DOC (Legacy Microsoft Word)
- RTF (Rich Text Format)
- TXT (Plain Text)
- HTML
- Markdown
- LaTeX

### Output Formats
- PDF
- DOCX
- RTF
- TXT
- HTML
- Markdown
- LaTeX
- EPUB
- Mobi

## Architecture

### Components

1. API Layer
   - FastAPI application
   - Request validation
   - Response formatting
   - Error handling

2. Conversion Engine
   - Format detection
   - Content extraction
   - Format conversion
   - Quality assurance

3. Storage Layer
   - Temporary storage
   - Result caching
   - Cleanup management

4. Monitoring
   - Performance metrics
   - Error tracking
   - Usage statistics

### Dependencies

- Python 3.8+
- FastAPI
- Pydantic
- python-multipart
- aiofiles
- PyPDF2
- python-docx
- markdown
- LaTeX tools

## API Endpoints

### Convert Document

```http
POST /api/v1/convert
Content-Type: application/json

{
  "document_id": "string",
  "source_format": "string",
  "target_format": "string",
  "options": {
    "quality": "high|medium|low",
    "preserve_layout": boolean,
    "ocr": boolean
  }
}
```

### Get Conversion Status

```http
GET /api/v1/convert/{conversion_id}
```

### Cancel Conversion

```http
DELETE /api/v1/convert/{conversion_id}
```

## Configuration

### Environment Variables

```env
# Service Configuration
PORT=8001
HOST=0.0.0.0
LOG_LEVEL=INFO

# Storage Configuration
TEMP_DIR=/tmp/conversions
MAX_FILE_SIZE=100MB
CLEANUP_INTERVAL=3600

# Conversion Settings
DEFAULT_QUALITY=high
MAX_CONCURRENT_CONVERSIONS=10
TIMEOUT=300

# AWS Configuration
AWS_REGION=us-east-1
S3_BUCKET=conversion-results
```

## Performance

### Benchmarks

- Average conversion time: < 5 seconds
- Maximum file size: 100MB
- Concurrent conversions: 10
- Success rate: > 99%

### Optimization

1. Caching
   - Result caching
   - Format detection caching
   - Template caching

2. Resource Management
   - Memory limits
   - CPU limits
   - Disk space management

3. Scaling
   - Horizontal scaling
   - Load balancing
   - Auto-scaling

## Error Handling

### Common Errors

1. Format Errors
   - Unsupported format
   - Corrupted file
   - Invalid content

2. Resource Errors
   - File too large
   - Insufficient memory
   - Disk space full

3. System Errors
   - Service unavailable
   - Timeout
   - Network issues

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

## Monitoring

### Metrics

1. Performance Metrics
   - Conversion time
   - Success rate
   - Error rate
   - Queue length

2. Resource Metrics
   - CPU usage
   - Memory usage
   - Disk usage
   - Network I/O

3. Business Metrics
   - Conversions per hour
   - Popular formats
   - Error types
   - User satisfaction

### Logging

1. Log Levels
   - ERROR: Critical issues
   - WARNING: Potential issues
   - INFO: Normal operations
   - DEBUG: Detailed information

2. Log Format
   ```json
   {
     "timestamp": "string",
     "level": "string",
     "service": "format_conversion",
     "message": "string",
     "metadata": {}
   }
   ```

## Security

### Authentication

- JWT token validation
- API key authentication
- Rate limiting

### Authorization

- Role-based access
- Resource permissions
- Operation restrictions

### Data Protection

- File encryption
- Secure storage
- Secure transmission
- Data cleanup

## Deployment

### Docker

```dockerfile
FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: format-conversion-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: format-conversion
  template:
    metadata:
      labels:
        app: format-conversion
    spec:
      containers:
      - name: format-conversion
        image: format-conversion-service:latest
        ports:
        - containerPort: 8001
        resources:
          limits:
            cpu: "1"
            memory: "1Gi"
          requests:
            cpu: "500m"
            memory: "512Mi"
```

## Development

### Local Setup

1. Clone repository
   ```bash
   git clone https://github.com/your-org/gen-ai-reusable.git
   cd gen-ai-reusable/services/format_conversion_service
   ```

2. Create virtual environment
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

4. Run service
   ```bash
   uvicorn main:app --reload
   ```

### Testing

1. Unit Tests
   ```bash
   pytest tests/unit/
   ```

2. Integration Tests
   ```bash
   pytest tests/integration/
   ```

3. Performance Tests
   ```bash
   pytest tests/performance/
   ```

## Contributing

### Guidelines

1. Code Style
   - Follow PEP 8
   - Use type hints
   - Write docstrings
   - Keep functions focused

2. Testing
   - Write unit tests
   - Update integration tests
   - Maintain test coverage
   - Document test cases

3. Documentation
   - Update API docs
   - Add code comments
   - Update README
   - Document changes

### Pull Request Process

1. Create feature branch
2. Make changes
3. Run tests
4. Update documentation
5. Submit PR
6. Address review comments
7. Merge after approval

## License

This service is part of the Gen AI Services platform and is licensed under the MIT License. 