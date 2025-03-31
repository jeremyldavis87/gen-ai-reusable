#!/bin/bash

# Exit on error
set -e

# Configuration
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

# Check Docker
check_docker() {
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
}

# Build Docker image
build_image() {
    local service=$1
    local context="./services/$service"
    local dockerfile="$context/Dockerfile"
    
    if [ ! -f "$dockerfile" ]; then
        log_warn "Dockerfile not found for $service, skipping..."
        return
    fi
    
    log_info "Building $service service..."
    docker build \
        --no-cache \
        --tag "gen-ai-$service:latest" \
        --file "$dockerfile" \
        "$context"
}

# Main build function
build() {
    log_info "Starting image build..."
    
    # Check prerequisites
    check_docker
    
    # Build images
    for service in "${SERVICES[@]}"; do
        build_image "$service"
    done
    
    log_info "Image build completed successfully!"
}

# Handle errors
handle_error() {
    log_error "Build failed!"
    exit 1
}

# Set up error handling
trap handle_error ERR

# Run build
build 