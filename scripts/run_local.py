#!/usr/bin/env python3

"""
Local development runner for Gen AI Reusable Services

This script allows running any of the services locally with uvicorn
while maintaining compatibility with containerized deployment.
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.absolute())
sys.path.insert(0, project_root)

import argparse
import uvicorn
import importlib.util
from dotenv import load_dotenv
import subprocess
import signal
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi

# Load environment variables from .env file in the project root
load_dotenv()

def fix_schema_refs(schema_dict):
    """Recursively fix schema references in a dictionary."""
    if isinstance(schema_dict, dict):
        for key, value in schema_dict.items():
            if key == "$ref" and isinstance(value, str):
                if value.startswith("#/schemas/"):
                    schema_dict[key] = value.replace("/schemas/", "/components/schemas/")
            else:
                fix_schema_refs(value)
    elif isinstance(schema_dict, list):
        for item in schema_dict:
            fix_schema_refs(item)

def configure_service_app(app):
    """Configure FastAPI app with OpenAPI customization and CORS."""
    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema

        openapi_schema = get_openapi(
            title="AI Services API",
            version="1.0.0",
            description="API for AI services including classification, extraction, and more",
            routes=app.routes,
        )

        # Initialize components if not present
        if "components" not in openapi_schema:
            openapi_schema["components"] = {}
        if "schemas" not in openapi_schema["components"]:
            openapi_schema["components"]["schemas"] = {}

        # Add security scheme for JWT Bearer token
        openapi_schema["components"]["securitySchemes"] = {
            "Bearer": {
                "type": "http",
                "scheme": "bearer",
                "bearerFormat": "JWT",
                "description": "Enter the JWT token in the format: Bearer <token>"
            }
        }

        # Fix all schema references in the entire OpenAPI schema
        fix_schema_refs(openapi_schema)

        # Apply security globally to all endpoints
        openapi_schema["security"] = [{"Bearer": []}]

        app.openapi_schema = openapi_schema
        return app.openapi_schema

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure this appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Set custom OpenAPI schema
    app.openapi = custom_openapi

    # Update app title and description
    app.title = "AI Services API"
    app.description = "API for AI services including classification, extraction, and more"
    app.version = "1.0.0"

# Define available services
AVAILABLE_SERVICES = {
    "format": {
        "module": "services.format_conversion_service.main",
        "app": "app",
        "description": "Format Conversion Service",
        "default_port": 8001
    },
    "classification": {
        "module": "services.classification_service.main",
        "app": "app",
        "description": "Classification Service",
        "default_port": 8002
    },
    "code": {
        "module": "services.code_service.main",
        "app": "app",
        "description": "Code Generation Service",
        "default_port": 8009
    },
    "conversation": {
        "module": "services.conversational_service.main",
        "app": "app",
        "description": "Conversational Service",
        "default_port": 8008
    },
    "document": {
        "module": "services.document_extraction.main",
        "app": "app",
        "description": "Document Extraction Service",
        "default_port": 8007
    },
    "personalization": {
        "module": "services.personalization_service.main",
        "app": "app",
        "description": "Personalization Service",
        "default_port": 8006
    },
    "quality": {
        "module": "services.quality_service.main",
        "app": "app",
        "description": "Quality Service",
        "default_port": 8005
    },
    "search": {
        "module": "services.search_service.main",
        "app": "app",
        "description": "Search Service",
        "default_port": 8004
    },
    "workflow": {
        "module": "services.workflow_service.main",
        "app": "app",
        "description": "Workflow Service",
        "default_port": 8003
    }
}

def setup_environment():
    """
    Set up environment variables for local development that would normally be 
    provided by the container environment or AWS
    """
    # Set environment flag to indicate we're running locally, not in a container
    os.environ["CONTAINER_ENV"] = "false"
    os.environ["LOCAL_DEV"] = "true"
    
    # Note: PYTHONPATH is now handled at the script level

def run_service(service_name, host, port, reload):
    """
    Run the specified service using uvicorn by importing the module
    This is the standard way but may cause issues if other services have dependency problems
    """
    if service_name not in AVAILABLE_SERVICES:
        print(f"Error: Service '{service_name}' not found. Available services: {', '.join(AVAILABLE_SERVICES.keys())}")
        return None
    
    # Set up environment variables for local development
    setup_environment()
    
    service_info = AVAILABLE_SERVICES[service_name]
    module_path = service_info["module"]
    app_name = service_info["app"]
    
    print(f"Starting {service_info['description']} on http://{host}:{port}")
    print(f"Environment: Local Development")
    print(f"Using environment variables from: {os.path.abspath('.env')}")
    
    # Import the service module and configure its app
    try:
        module = importlib.import_module(module_path)
        app = getattr(module, app_name)
        configure_service_app(app)
    except Exception as e:
        print(f"Error configuring service: {str(e)}")
        return None
    
    # Run with uvicorn
    return subprocess.Popen([
        sys.executable, "-m", "uvicorn",
        f"{module_path}:{app_name}",
        "--host", host,
        "--port", str(port),
        *(["--reload"] if reload else [])
    ])

def run_service_direct(service_name, host, port, reload):
    """
    Run the specified service directly using uvicorn without importing other services
    This avoids dependency issues with other services that may not be fully implemented
    """
    if service_name not in AVAILABLE_SERVICES:
        print(f"Error: Service '{service_name}' not found. Available services: {', '.join(AVAILABLE_SERVICES.keys())}")
        return None
    
    # Set up environment variables for local development
    setup_environment()
    
    service_info = AVAILABLE_SERVICES[service_name]
    module_path = service_info["module"]
    app_name = service_info["app"]
    
    print(f"Starting {service_info['description']} on http://{host}:{port}")
    print(f"Environment: Local Development")
    print(f"Using environment variables from: {os.path.abspath('.env')}")
    
    # Run with uvicorn directly (not importing the module)
    cmd = f"uvicorn {module_path}:{app_name} --host {host} --port {port} {'--reload' if reload else ''}"
    return subprocess.Popen(cmd.split())

def parse_service_spec(spec):
    """
    Parse a service specification string in the format 'service:port' or 'service'
    Returns tuple of (service_name, port)
    """
    if ':' in spec:
        service, port = spec.split(':')
        return service, int(port)
    service_name = spec
    return service_name, AVAILABLE_SERVICES[service_name]["default_port"]

def run_services(services, host, reload, direct):
    """
    Run multiple services and handle their processes
    """
    processes = []
    
    try:
        for service_spec in services:
            service_name, port = parse_service_spec(service_spec)
            
            if direct:
                process = run_service_direct(service_name, host, port, reload)
            else:
                process = run_service(service_name, host, port, reload)
            
            if process:
                processes.append(process)
        
        print("\nAll services started! Press Ctrl+C to stop.")
        
        # Wait for all processes
        for process in processes:
            process.wait()
            
    except KeyboardInterrupt:
        print("\nStopping all services...")
        for process in processes:
            process.terminate()
        sys.exit(0)

def main():
    parser = argparse.ArgumentParser(description="Run Gen AI Reusable Services locally")
    parser.add_argument(
        "services",
        nargs="+",
        type=str,
        help="Services to run. Can be specified as 'service' or 'service:port'"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind the server to (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    parser.add_argument(
        "--direct",
        action="store_true",
        help="Run service directly without importing other services (avoids dependency issues)"
    )
    
    args = parser.parse_args()
    
    # Validate services
    for service_spec in args.services:
        service_name = service_spec.split(':')[0]
        if service_name not in AVAILABLE_SERVICES:
            print(f"Error: Service '{service_name}' not found. Available services: {', '.join(AVAILABLE_SERVICES.keys())}")
            sys.exit(1)
    
    run_services(args.services, args.host, args.reload, args.direct)

if __name__ == "__main__":
    main()
