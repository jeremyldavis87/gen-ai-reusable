#!/bin/bash

# Exit on error
set -e

# Configuration
AWS_REGION="us-east-1"
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

# Check Docker
check_docker() {
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
}

# Get AWS account ID
get_aws_account_id() {
    aws sts get-caller-identity --query Account --output text
}

# Push Docker image
push_image() {
    local service=$1
    local account_id=$2
    local repository="$account_id.dkr.ecr.$AWS_REGION.amazonaws.com/gen-ai-$service"
    
    log_info "Pushing $service service..."
    
    # Tag image
    docker tag "gen-ai-$service:latest" "$repository:latest"
    
    # Push image
    docker push "$repository:latest"
}

# Main push function
push() {
    log_info "Starting image push..."
    
    # Check prerequisites
    check_aws_cli
    check_aws_credentials
    check_docker
    
    # Get AWS account ID
    AWS_ACCOUNT_ID=$(get_aws_account_id)
    
    # Login to ECR
    log_info "Logging in to ECR..."
    aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com
    
    # Push images
    for service in "${SERVICES[@]}"; do
        push_image "$service" "$AWS_ACCOUNT_ID"
    done
    
    log_info "Image push completed successfully!"
}

# Handle errors
handle_error() {
    log_error "Push failed!"
    exit 1
}

# Set up error handling
trap handle_error ERR

# Run push
push 