# Development Guide

## Getting Started

### Prerequisites

1. Development Environment
   - Python 3.8+
   - Docker and Docker Compose
   - Git
   - AWS CLI
   - IDE (VS Code recommended)

2. AWS Setup
   - AWS account with appropriate permissions
   - AWS CLI configured
   - Access to required AWS services

### Local Setup

1. Clone the repository
   ```bash
   git clone https://github.com/your-org/gen-ai-reusable.git
   cd gen-ai-reusable
   ```

2. Set up Python environment
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Configure environment variables
   ```bash
   cp config/.env.sample config/.env.development
   # Edit config/.env.development with your settings
   ```

4. Start local development
   ```bash
   docker-compose up --build
   ```

## Development Workflow

### Code Organization

1. Service Structure
   ```
   service/
   ├── Dockerfile
   ├── requirements.txt
   ├── main.py
   ├── config/
   ├── models/
   ├── routes/
   ├── services/
   └── utils/
   ```

2. Code Style
   - Follow PEP 8 guidelines
   - Use type hints
   - Write docstrings
   - Keep functions focused and small

### Git Workflow

1. Branch Strategy
   - main: Production code
   - develop: Development code
   - feature/*: New features
   - bugfix/*: Bug fixes
   - release/*: Release preparation

2. Commit Messages
   ```
   type(scope): description

   [optional body]

   [optional footer]
   ```

3. Pull Requests
   - Create from feature/bugfix branch to develop
   - Include tests
   - Update documentation
   - Get code review

### Testing

1. Unit Tests
   ```bash
   pytest tests/unit/
   ```

2. Integration Tests
   ```bash
   pytest tests/integration/
   ```

3. End-to-End Tests
   ```bash
   pytest tests/e2e/
   ```

## Service Development

### Creating a New Service

1. Generate service template
   ```bash
   python scripts/create_service.py --name new_service
   ```

2. Implement service logic
   - Add routes
   - Implement business logic
   - Add tests
   - Update documentation

3. Add to docker-compose
   ```yaml
   new_service:
     build:
       context: ./services/new_service
       dockerfile: Dockerfile
     ports:
       - "8010:8000"
     env_file:
       - ./config/.env.development
   ```

### Adding Dependencies

1. Service-specific dependencies
   ```bash
   # Add to service/requirements.txt
   new_dependency==1.0.0
   ```

2. Shared dependencies
   ```bash
   # Add to root requirements.txt
   shared_dependency==1.0.0
   ```

## Debugging

### Local Debugging

1. VS Code Configuration
   ```json
   {
     "version": "0.2.0",
     "configurations": [
       {
         "name": "Python: Service",
         "type": "python",
         "request": "launch",
         "program": "${workspaceFolder}/services/service_name/main.py",
         "env": {
           "PYTHONPATH": "${workspaceFolder}"
         }
       }
     ]
   }
   ```

2. Logging
   ```python
   import logging

   logging.basicConfig(level=logging.DEBUG)
   logger = logging.getLogger(__name__)
   ```

### Remote Debugging

1. Enable remote debugging
   ```python
   import debugpy
   debugpy.listen(("0.0.0.0", 5678))
   debugpy.wait_for_client()
   ```

2. Attach debugger
   ```bash
   python -m debugpy --connect localhost:5678
   ```

## Performance Optimization

### Code Optimization

1. Profiling
   ```python
   import cProfile
   import pstats

   profiler = cProfile.Profile()
   profiler.enable()
   # Your code here
   profiler.disable()
   stats = pstats.Stats(profiler)
   stats.sort_stats('cumulative')
   stats.print_stats()
   ```

2. Memory Profiling
   ```python
   from memory_profiler import profile

   @profile
   def your_function():
       # Your code here
       pass
   ```

### Database Optimization

1. Query Optimization
   - Use indexes
   - Optimize joins
   - Use connection pooling
   - Implement caching

2. Monitoring
   - Track query performance
   - Monitor connection pool
   - Set up alerts

## Security

### Best Practices

1. Input Validation
   ```python
   from pydantic import BaseModel

   class InputModel(BaseModel):
       field: str
       value: int
   ```

2. Authentication
   ```python
   from fastapi import Depends, HTTPException
   from fastapi.security import OAuth2PasswordBearer

   oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

   async def get_current_user(token: str = Depends(oauth2_scheme)):
       # Validate token
       pass
   ```

3. Authorization
   ```python
   from fastapi import Security
   from fastapi.security import APIKeyHeader

   api_key_header = APIKeyHeader(name="X-API-Key")

   async def get_api_key(api_key: str = Security(api_key_header)):
       # Validate API key
       pass
   ```

## Deployment

### Local Deployment

1. Build images
   ```bash
   docker-compose build
   ```

2. Run services
   ```bash
   docker-compose up
   ```

### AWS Deployment

1. Configure AWS
   ```bash
   aws configure
   ```

2. Deploy services
   ```bash
   ./scripts/deploy.sh
   ```

## Troubleshooting

### Common Issues

1. Service Not Starting
   - Check logs
   - Verify environment variables
   - Check port conflicts
   - Verify dependencies

2. Database Issues
   - Check connection string
   - Verify credentials
   - Check network access
   - Monitor connections

3. Performance Issues
   - Check resource usage
   - Monitor logs
   - Profile code
   - Check database queries

### Getting Help

1. Internal Resources
   - Team documentation
   - Internal wiki
   - Team chat
   - Code reviews

2. External Resources
   - AWS documentation
   - Python documentation
   - Stack Overflow
   - GitHub issues 