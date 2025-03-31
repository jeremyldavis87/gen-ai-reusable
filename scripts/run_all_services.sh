#!/bin/bash

# Exit on error
set -e

# Configuration
SERVICES=(
    "format_conversion_service:8001"
    "classification_service:8002"
    "workflow_service:8003"
    "search_service:8004"
    "quality_service:8005"
    "personalization_service:8006"
    "document_extraction:8007"
    "conversational_service:8008"
    "code_service:8009"
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

# Check Python
check_python() {
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is not installed"
        exit 1
    fi
}

# Check virtual environment
check_venv() {
    if [ ! -d ".venv" ]; then
        log_warn "Virtual environment not found. Creating one..."
        python3 -m venv .venv
        source .venv/bin/activate
        pip install -r requirements.txt
    else
        source .venv/bin/activate
    fi
}

# Run service
run_service() {
    local service=$1
    local port=$2
    local service_dir="services/$service"
    local current_dir=$(pwd)
    
    if [ ! -d "$service_dir" ]; then
        log_warn "Service directory $service_dir not found, skipping..."
        return
    fi
    
    if [ ! -f "$service_dir/main.py" ]; then
        log_warn "main.py not found in $service_dir, skipping..."
        return
    fi
    
    log_info "Starting $service on port $port..."
    cd "$service_dir" && uvicorn main:app --host 0.0.0.0 --port "$port" --reload &
    cd "$current_dir"
}

# Main function
main() {
    log_info "Starting all services..."
    
    # Check prerequisites
    check_python
    check_venv
    
    # Add project root to PYTHONPATH
    export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$(pwd)"
    
    # Run each service
    for service_config in "${SERVICES[@]}"; do
        IFS=':' read -r service port <<< "$service_config"
        run_service "$service" "$port"
    done
    
    log_info "All services started! Press Ctrl+C to stop."
    
    # Wait for all background processes
    wait
}

# Handle errors
handle_error() {
    log_error "Failed to start services!"
    exit 1
}

# Set up error handling
trap handle_error ERR

# Run main function
main 