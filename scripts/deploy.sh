#!/bin/bash

# Exit on error
set -e

# Configuration
AWS_REGION="us-east-1"
ECS_CLUSTER="gen-ai-cluster"
SERVICES=(
    "format-conversion"
    "classification"
    "workflow"
    "search"
    "quality"
    "personalization"
    "document-extraction"
    "conversational"
    "code"
)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check AWS CLI
check_aws_cli() {
    if ! command -v aws &> /dev/null; then
        log_error "AWS CLI is not installed"
        exit 1
    fi
}

# Check AWS credentials
check_aws_credentials() {
    if ! aws sts get-caller-identity &> /dev/null; then
        log_error "AWS credentials are not configured"
        exit 1
    fi
}

# Build Docker images
build_images() {
    log_info "Building Docker images..."
    
    for service in "${SERVICES[@]}"; do
        log_info "Building $service service..."
        docker build -t "gen-ai-$service:latest" "./services/$service"
    done
}

# Push images to ECR
push_images() {
    log_info "Pushing images to ECR..."
    
    # Get ECR login token
    aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com
    
    for service in "${SERVICES[@]}"; do
        log_info "Pushing $service service..."
        docker tag "gen-ai-$service:latest" "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/gen-ai-$service:latest"
        docker push "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/gen-ai-$service:latest"
    done
}

# Update ECS services
update_services() {
    log_info "Updating ECS services..."
    
    for service in "${SERVICES[@]}"; do
        log_info "Updating $service service..."
        aws ecs update-service \
            --cluster $ECS_CLUSTER \
            --service "gen-ai-$service" \
            --force-new-deployment \
            --region $AWS_REGION
    done
}

# Wait for services to stabilize
wait_for_stable() {
    log_info "Waiting for services to stabilize..."
    
    for service in "${SERVICES[@]}"; do
        log_info "Waiting for $service service..."
        aws ecs wait services-stable \
            --cluster $ECS_CLUSTER \
            --services "gen-ai-$service" \
            --region $AWS_REGION
    done
}

# Check service health
check_health() {
    log_info "Checking service health..."
    
    for service in "${SERVICES[@]}"; do
        log_info "Checking $service service..."
        # Add health check logic here
        # For example, curl the health endpoint
        # curl -f http://$service.gen-ai.internal/health || exit 1
    done
}

# Main deployment function
deploy() {
    log_info "Starting deployment..."
    
    # Check prerequisites
    check_aws_cli
    check_aws_credentials
    
    # Build and push images
    build_images
    push_images
    
    # Update services
    update_services
    
    # Wait for stability
    wait_for_stable
    
    # Check health
    check_health
    
    log_info "Deployment completed successfully!"
}

# Handle errors
handle_error() {
    log_error "Deployment failed!"
    exit 1
}

# Set up error handling
trap handle_error ERR

# Run deployment
deploy 