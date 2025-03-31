#!/usr/bin/env python3

import os
import sys
import argparse
from pathlib import Path

def create_directory_structure(service_name: str) -> None:
    """Create the directory structure for a new service."""
    base_path = Path("services") / service_name
    directories = [
        "",
        "config",
        "models",
        "routes",
        "services",
        "utils",
        "tests",
        "tests/unit",
        "tests/integration",
        "tests/e2e"
    ]
    
    for directory in directories:
        (base_path / directory).mkdir(parents=True, exist_ok=True)

def create_dockerfile(service_name: str) -> None:
    """Create a Dockerfile for the service."""
    dockerfile_content = f"""FROM python:3.8-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
"""
    
    with open(f"services/{service_name}/Dockerfile", "w") as f:
        f.write(dockerfile_content)

def create_requirements(service_name: str) -> None:
    """Create requirements.txt for the service."""
    requirements_content = """fastapi>=0.68.0
uvicorn>=0.15.0
pydantic>=1.8.0
python-dotenv>=0.19.0
pytest>=6.2.5
httpx>=0.24.0
"""
    
    with open(f"services/{service_name}/requirements.txt", "w") as f:
        f.write(requirements_content)

def create_main(service_name: str) -> None:
    """Create the main FastAPI application file."""
    main_content = f"""from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="{service_name.title()} Service",
    description="Service for {service_name} functionality",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {{"message": "Welcome to {service_name} service"}}

@app.get("/health")
async def health_check():
    return {{"status": "healthy"}}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""
    
    with open(f"services/{service_name}/main.py", "w") as f:
        f.write(main_content)

def create_test(service_name: str) -> None:
    """Create a basic test file."""
    test_content = """import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to service"}

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}
"""
    
    with open(f"services/{service_name}/tests/unit/test_main.py", "w") as f:
        f.write(test_content)

def create_readme(service_name: str) -> None:
    """Create a README.md file for the service."""
    readme_content = f"""# {service_name.title()} Service

This service provides functionality for {service_name}.

## Features

- Feature 1
- Feature 2
- Feature 3

## API Endpoints

- GET /: Welcome message
- GET /health: Health check endpoint

## Development

### Prerequisites

- Python 3.8+
- Docker
- Docker Compose

### Local Development

1. Create virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the service:
   ```bash
   uvicorn main:app --reload
   ```

### Docker Development

1. Build the image:
   ```bash
   docker build -t {service_name}-service .
   ```

2. Run the container:
   ```bash
   docker run -p 8000:8000 {service_name}-service
   ```

## Testing

### Unit Tests

```bash
pytest tests/unit/
```

### Integration Tests

```bash
pytest tests/integration/
```

### End-to-End Tests

```bash
pytest tests/e2e/
```

## Deployment

The service can be deployed using Docker and AWS ECS.

## Documentation

API documentation is available at `/docs` when running the service.
"""
    
    with open(f"services/{service_name}/README.md", "w") as f:
        f.write(readme_content)

def create_service(service_name: str) -> None:
    """Create a new service with all necessary files."""
    print(f"Creating new service: {service_name}")
    
    # Create directory structure
    create_directory_structure(service_name)
    
    # Create files
    create_dockerfile(service_name)
    create_requirements(service_name)
    create_main(service_name)
    create_test(service_name)
    create_readme(service_name)
    
    print(f"Service {service_name} created successfully!")

def main():
    parser = argparse.ArgumentParser(description="Create a new service template")
    parser.add_argument("--name", required=True, help="Name of the service")
    args = parser.parse_args()
    
    create_service(args.name)

if __name__ == "__main__":
    main() 