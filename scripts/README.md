# Gen AI Reusable Services Scripts

This directory contains utility scripts for managing the Gen AI Reusable Services project, including local development, deployment, and service management.

## Local Development Scripts

### `run_local.py`

A Python script for running services locally during development. It provides flexibility in how services are started and includes features like hot-reloading. You can run one or more services in a single command.

```bash
# Basic usage - run a single service
python scripts/run_local.py format

# Run multiple services with default ports
python scripts/run_local.py format classification workflow

# Run services with custom ports
python scripts/run_local.py format:8001 classification:8002 workflow:8003

# Run with hot-reloading enabled
python scripts/run_local.py format classification --reload

# Run with direct mode (avoids importing other services)
python scripts/run_local.py format classification --direct

# Run with custom host
python scripts/run_local.py format classification --host 0.0.0.0

# Available services and their default ports:
# - format: Format Conversion Service (8001)
# - classification: Classification Service (8002)
# - code: Code Generation Service (8009)
# - conversation: Conversational Service (8008)
# - document: Document Extraction Service (8007)
# - personalization: Personalization Service (8006)
# - quality: Quality Service (8005)
# - search: Search Service (8004)
# - workflow: Workflow Service (8003)
```

#### Running Services with run_local.py

The `run_local.py` script supports various ways to run services:

1. **Running a Single Service**
   ```bash
   # Run format service on default port (8001)
   python scripts/run_local.py format

   # Run classification service with custom port
   python scripts/run_local.py classification:8002

   # Run with hot-reload
   python scripts/run_local.py format --reload
   ```

2. **Running Multiple Services**
   ```bash
   # Run format and classification services with default ports
   python scripts/run_local.py format classification

   # Run services with custom ports
   python scripts/run_local.py format:8001 classification:8002 workflow:8003

   # Run with hot-reload enabled
   python scripts/run_local.py format classification --reload
   ```

3. **Running All Services**
   ```bash
   # Run all services with default ports
   python scripts/run_local.py format classification workflow search quality personalization document conversation code

   # Or use the run_all_services.sh script for convenience
   ./scripts/run_all_services.sh
   ```

4. **Advanced Options**
   ```bash
   # Run with custom host
   python scripts/run_local.py format classification --host 0.0.0.0

   # Run with both hot-reload and direct mode
   python scripts/run_local.py format classification --reload --direct

   # Run with specific environment variables
   ENV=development python scripts/run_local.py format classification
   ```

Note: When running multiple services, the script will manage all processes and handle graceful shutdown when you press Ctrl+C.

### `run_all_services.sh`

A shell script that starts all services simultaneously for local development. It handles virtual environment setup and ensures all services are running on their designated ports.

```bash
# Start all services
./scripts/run_all_services.sh

# The script will:
# 1. Check for Python 3 installation
# 2. Create/activate virtual environment if needed
# 3. Install dependencies
# 4. Start all services on their respective ports
```

## Deployment Scripts

### `deploy.sh`

Main deployment script for AWS ECS. It handles the complete deployment process including building images, pushing to ECR, and updating services.

```bash
# Deploy all services to AWS
./scripts/deploy.sh

# The script will:
# 1. Check AWS CLI and credentials
# 2. Build Docker images
# 3. Push images to ECR
# 4. Update ECS services
# 5. Wait for services to stabilize
# 6. Check service health
```

### `build_images.sh`

Builds Docker images for all services. This script can be used independently when you only need to build images without deploying.

```bash
# Build all service images
./scripts/build_images.sh

# Build specific service image
./scripts/build_images.sh format-conversion
```

### `push_images.sh`

Pushes Docker images to Amazon ECR. This script handles authentication and image tagging.

```bash
# Push all images to ECR
./scripts/push_images.sh

# Push specific service image
./scripts/push_images.sh format-conversion
```

## Service Management

### `create_service.py`

A utility script for creating new service templates with the standard project structure.

```bash
# Create a new service
python scripts/create_service.py my_new_service

# The script will:
# 1. Create service directory structure
# 2. Generate boilerplate code
# 3. Set up configuration files
# 4. Add necessary dependencies
```

## Environment Variables

The following environment variables are required for deployment:

- `AWS_REGION`: AWS region (default: us-east-1)
- `AWS_ACCOUNT_ID`: Your AWS account ID
- `ECS_CLUSTER`: ECS cluster name (default: gen-ai-cluster)

These can be set in your `.env` file or exported in your shell.

## Best Practices

1. **Local Development**
   - Use `run_local.py` for individual service development
   - Use `run_all_services.sh` when working with multiple services
   - Always use the virtual environment for consistency

2. **Deployment**
   - Always test locally before deploying
   - Use `build_images.sh` and `push_images.sh` separately for more control
   - Monitor the deployment process using AWS Console or CLI

3. **Service Creation**
   - Use `create_service.py` for new services to maintain consistency
   - Follow the generated template structure
   - Update the service list in relevant scripts when adding new services

## Troubleshooting

1. **Service Won't Start**
   - Check if the port is already in use
   - Verify virtual environment is activated
   - Check service logs for errors

2. **Deployment Issues**
   - Verify AWS credentials are configured
   - Check ECR repository exists
   - Monitor ECS service events

3. **Image Build/Push Failures**
   - Check Docker daemon is running
   - Verify ECR login token is valid
   - Check image naming conventions 