import pytest
from fastapi.testclient import TestClient
import sys
import os
import json
from datetime import datetime, timedelta

# Add the project root to the Python path to ensure imports work correctly
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from services.classification_service.main import app
from utilities.auth import create_access_token

# Initialize test client
client = TestClient(app)

# Create a test token
test_token = create_access_token(
    data={"sub": "test_user", "name": "Test User"},
    expires_delta=timedelta(minutes=30)
)

# Add authentication headers to all requests
client.headers = {
    "Authorization": f"Bearer {test_token}"
}

def test_classify_text_endpoint():
    # Test data
    test_data = {
        "content": "I love this product! It's amazing and works perfectly. The customer service was excellent and the delivery was fast.",
        "content_type": "text",
        "classification_type": "sentiment",
        "categories": [
            {
                "id": "positive",
                "name": "Positive",
                "description": "Expresses positive sentiment"
            },
            {
                "id": "negative",
                "name": "Negative",
                "description": "Expresses negative sentiment"
            },
            {
                "id": "neutral",
                "name": "Neutral",
                "description": "Expresses neutral sentiment"
            }
        ],
        "confidence_threshold": 0.5,
        "max_categories": 1
    }

    # Make request to endpoint
    response = client.post("/classify", json=test_data)

    # Print response for debugging
    print("\nResponse status code:", response.status_code)
    print("\nResponse data:")
    print(json.dumps(response.json(), indent=2))

    # Assert response status code
    assert response.status_code == 200

    # Parse response
    response_data = response.json()

    # Assert response structure
    assert "request_id" in response_data
    assert "results" in response_data
    assert "content_excerpt" in response_data
    assert "processing_time" in response_data

    # Assert results
    assert len(response_data["results"]) > 0
    result = response_data["results"][0]
    assert "category_id" in result
    assert "category_name" in result
    assert "confidence" in result
    assert result["category_id"] == "positive"
    assert result["confidence"] >= 0.5

    # Assert content excerpt
    assert len(response_data["content_excerpt"]) > 0
    assert len(response_data["content_excerpt"]) <= 150

    # Assert processing time
    assert response_data["processing_time"] > 0

def test_classify_document_endpoint():
    # Test data
    test_data = {
        "content_type": "document",
        "classification_type": "category",
        "categories": [
            {
                "id": "technical",
                "name": "Technical Documentation",
                "description": "Technical documentation and guides"
            },
            {
                "id": "business",
                "name": "Business Document",
                "description": "Business-related documents"
            },
            {
                "id": "legal",
                "name": "Legal Document",
                "description": "Legal documents and contracts"
            }
        ],
        "confidence_threshold": 0.5
    }

    # Create a test document file
    test_file = {
        "file": ("test_doc.txt", "This is a technical document containing API documentation and implementation details.")
    }

    # Make request to endpoint
    response = client.post("/classify", data=test_data, files=test_file)

    # Print response for debugging
    print("\nResponse status code:", response.status_code)
    print("\nResponse data:")
    print(json.dumps(response.json(), indent=2))

    # Assert response status code
    assert response.status_code == 200

    # Parse response
    response_data = response.json()

    # Assert response structure
    assert "request_id" in response_data
    assert "results" in response_data
    assert "content_excerpt" in response_data
    assert "processing_time" in response_data

    # Assert results
    assert len(response_data["results"]) > 0
    result = response_data["results"][0]
    assert "category_id" in result
    assert "category_name" in result
    assert "confidence" in result
    assert result["category_id"] == "technical"
    assert result["confidence"] >= 0.5

def test_batch_classify_endpoint():
    # Test data
    test_data = {
        "content_type": "text",
        "classification_type": "topic",
        "categories": [
            {
                "id": "technology",
                "name": "Technology",
                "description": "Technology-related content"
            },
            {
                "id": "business",
                "name": "Business",
                "description": "Business-related content"
            },
            {
                "id": "health",
                "name": "Health",
                "description": "Health-related content"
            }
        ],
        "confidence_threshold": 0.5
    }

    # Create test files
    test_files = [
        ("file1.txt", "The latest developments in artificial intelligence and machine learning."),
        ("file2.txt", "Stock market trends and investment strategies."),
        ("file3.txt", "New research findings on healthy eating habits.")
    ]

    # Make request to endpoint
    files = [("files", (filename, content)) for filename, content in test_files]
    response = client.post("/batch-classify", data=test_data, files=files)

    # Print response for debugging
    print("\nResponse status code:", response.status_code)
    print("\nResponse data:")
    print(json.dumps(response.json(), indent=2))

    # Assert response status code
    assert response.status_code == 200

    # Parse response
    response_data = response.json()

    # Assert response structure
    assert "request_id" in response_data
    assert "results" in response_data
    assert "summary" in response_data

    # Assert results
    assert len(response_data["results"]) == 3
    for result in response_data["results"]:
        assert "filename" in result
        assert "classifications" in result
        assert "success" in result
        assert result["success"] is True
        assert len(result["classifications"]) > 0

    # Assert summary
    summary = response_data["summary"]
    assert "total_files" in summary
    assert "successful_classifications" in summary
    assert "failed_classifications" in summary
    assert "processing_time" in summary
    assert summary["total_files"] == 3
    assert summary["successful_classifications"] == 3
    assert summary["failed_classifications"] == 0

def test_classify_tabular_endpoint():
    # Test data
    test_data = {
        "content_type": "tabular",
        "classification_type": "category",
        "categories": [
            {
                "id": "high_value",
                "name": "High Value Customer",
                "description": "Customers with high lifetime value"
            },
            {
                "id": "medium_value",
                "name": "Medium Value Customer",
                "description": "Customers with medium lifetime value"
            },
            {
                "id": "low_value",
                "name": "Low Value Customer",
                "description": "Customers with low lifetime value"
            }
        ],
        "confidence_threshold": 0.5
    }

    # Create a test CSV file
    csv_content = "customer_id,spend,orders\n1,5000,10\n2,2000,5\n3,500,2"
    test_file = {
        "file": ("test_data.csv", csv_content)
    }

    # Make request to endpoint
    response = client.post(
        "/classify-tabular",
        data={
            **test_data,
            "column_to_classify": "spend",
            "id_column": "customer_id"
        },
        files=test_file
    )

    # Print response for debugging
    print("\nResponse status code:", response.status_code)
    print("\nResponse data:")
    print(json.dumps(response.json(), indent=2))

    # Assert response status code
    assert response.status_code == 200

    # Parse response
    response_data = response.json()

    # Assert response structure
    assert "request_id" in response_data
    assert "results" in response_data
    assert "summary" in response_data

    # Assert results
    assert len(response_data["results"]) == 3
    for result in response_data["results"]:
        assert "id" in result
        assert "content" in result
        assert "classifications" in result
        assert "success" in result
        assert result["success"] is True
        assert len(result["classifications"]) > 0

    # Assert summary
    summary = response_data["summary"]
    assert "total_rows" in summary
    assert "successful_classifications" in summary
    assert "failed_classifications" in summary
    assert summary["total_rows"] == 3
    assert summary["successful_classifications"] == 3
    assert summary["failed_classifications"] == 0

def test_hierarchical_classification():
    # Test data with hierarchical categories
    test_data = {
        "content": "The new iPhone 15 Pro Max features advanced camera capabilities and improved battery life.",
        "content_type": "text",
        "classification_type": "hierarchical",
        "categories": [
            {
                "id": "electronics",
                "name": "Electronics",
                "description": "Electronic devices and accessories"
            },
            {
                "id": "smartphones",
                "name": "Smartphones",
                "description": "Mobile phones",
                "parent_id": "electronics"
            },
            {
                "id": "apple",
                "name": "Apple",
                "description": "Apple products",
                "parent_id": "smartphones"
            }
        ],
        "confidence_threshold": 0.5
    }

    # Make request to endpoint
    response = client.post("/classify", json=test_data)

    # Print response for debugging
    print("\nResponse status code:", response.status_code)
    print("\nResponse data:")
    print(json.dumps(response.json(), indent=2))

    # Assert response status code
    assert response.status_code == 200

    # Parse response
    response_data = response.json()

    # Assert response structure
    assert "request_id" in response_data
    assert "results" in response_data

    # Assert results contain hierarchical categories
    results = response_data["results"]
    assert len(results) > 0
    
    # Check that we have classifications for both parent and child categories
    category_ids = [result["category_id"] for result in results]
    assert "electronics" in category_ids
    assert "smartphones" in category_ids
    assert "apple" in category_ids

def test_multi_label_classification():
    # Test data for multi-label classification
    test_data = {
        "content": "The new gaming laptop features high-performance graphics and long battery life, making it perfect for both gaming and work.",
        "content_type": "text",
        "classification_type": "multi_label",
        "categories": [
            {
                "id": "gaming",
                "name": "Gaming",
                "description": "Gaming-related content"
            },
            {
                "id": "work",
                "name": "Work",
                "description": "Work-related content"
            },
            {
                "id": "technology",
                "name": "Technology",
                "description": "Technology-related content"
            }
        ],
        "confidence_threshold": 0.5
    }

    # Make request to endpoint
    response = client.post("/classify", json=test_data)

    # Print response for debugging
    print("\nResponse status code:", response.status_code)
    print("\nResponse data:")
    print(json.dumps(response.json(), indent=2))

    # Assert response status code
    assert response.status_code == 200

    # Parse response
    response_data = response.json()

    # Assert response structure
    assert "request_id" in response_data
    assert "results" in response_data

    # Assert results contain multiple labels
    results = response_data["results"]
    assert len(results) > 1  # Should have multiple classifications
    
    # Check that we have classifications for multiple categories
    category_ids = [result["category_id"] for result in results]
    assert "gaming" in category_ids
    assert "work" in category_ids
    assert "technology" in category_ids 