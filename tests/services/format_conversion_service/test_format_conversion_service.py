import pytest
from fastapi.testclient import TestClient
import sys
import os

# Add the project root to the Python path to ensure imports work correctly
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from services.format_conversion_service.main import app
import json
from datetime import datetime

# Initialize test client
client = TestClient(app)

def test_convert_text_endpoint():
    # Test data
    test_data = {
        "text": "SecureGuard Pro X7 Smart Security System\nProduct ID: f47ac10b-58cc-4372-a567-0e02b2c3d479\nCategory: Electronics\nPrice: $499.99 USD\nProduct Description:\nThe SecureGuard Pro X7 is our most advanced home security system designed for modern homeowners who prioritize safety without compromising on style or convenience. Featuring AI-powered motion detection, seamless smart home integration, and military-grade encryption, this system provides comprehensive protection against intrusions while offering intuitive control through our award-winning mobile app.\n\nSpecifications:\n- Resolution: 4K Ultra HD (pixels)\n- Connectivity: Wi-Fi 6, Bluetooth 5.2, Z-Wave, Zigbee\n- Storage: 500 GB\n- Battery Life: 72 hours\n- Weather Resistance: IP67\n- Motion Detection Range: 30 feet\n\nPackage Options:\n1. Starter Package\n   Price: $499.99\n   Includes: 2 cameras, keypad\n   Color: White\n\n2. Premium Package\n   Price: $699.99\n   Includes: 4 cameras, keypad\n   Color: Black\n\n3. Ultimate Package\n   Price: $999.99\n   Includes: 6 cameras, keypad, doorbell, motion sensors\n   Color: Graphite\n\nAvailability:\nIn stock: Yes (42 units available)\nBackorders accepted if stock is depleted.\n\nCustomer Reviews:\nAverage Rating: 4.8/5 (from 127 reviews)\nFeatured Reviews:\n\"Best Investment For Our Family's Peace of Mind\" ★★★★★\nAfter a break-in attempt in our neighborhood, we installed the SecureGuard Pro X7 system. The setup was incredibly easy, and the app interface is intuitive. Within the first week, the system detected suspicious activity near our garage and sent immediate alerts with crystal clear video.\nVerified Purchase: March 1, 2025\n42 people found this helpful\n\n\"Exceeded My High Expectations\" ★★★★★\nAs someone who works in IT security, I have high standards for technology, especially security systems. The SecureGuard Pro X7 exceeded my expectations in every way. The encryption protocols are top-notch, the false positive rate is remarkably low, and the customization options are extensive.\nVerified Purchase: February 15, 2025\n38 people found this helpful",
        "instructions": "Convert the provided product information into a structured JSON object following the schema. Extract all relevant details including product ID, name, description, category, price, specifications, variants, reviews, and availability information. Generate UUIDs for any missing identifiers. Format dates in ISO 8601 format (YYYY-MM-DD). For metadata, use today's date (2025-03-28T14:30:00Z) as updatedAt and February 28, 2025 (2025-02-28T09:00:00Z) for createdAt. Include relevant tags and keywords extracted from the product description. Ensure all required fields are properly populated and formatted according to the schema constraints. IMPORTANT: Return ONLY the JSON object without any additional text, explanations, or markdown formatting. The output should be a valid JSON object that can be parsed directly.",
        "output_format": "json",
        "schema": {
            "product": {
                "id": "",
                "name": "",
                "description": "",
                "category": "",
                "price": {
                    "amount": None,
                    "currency": ""
                },
                "availability": {
                    "inStock": None,
                    "quantity": None,
                    "backorderAllowed": None,
                    "estimatedRestockDate": None
                },
                "specifications": [
                    {
                        "name": "",
                        "value": "",
                        "unit": ""
                    }
                ],
                "variants": [
                    {
                        "id": "",
                        "attributes": {
                            "package": "",
                            "cameraCount": "",
                            "includesKeypad": "",
                            "color": ""
                        },
                        "price": None,
                        "images": [
                            {
                                "url": "",
                                "altText": "",
                                "isPrimary": None
                            }
                        ]
                    }
                ],
                "reviews": {
                    "average": None,
                    "count": None,
                    "featured": [
                        {
                            "authorId": "",
                            "rating": None,
                            "title": "",
                            "text": "",
                            "date": "",
                            "helpfulVotes": None
                        }
                    ]
                },
                "metadata": {
                    "createdAt": "",
                    "updatedAt": "",
                    "tags": [""],
                    "searchKeywords": [""]
                }
            }
        },
        "max_tokens": 4000,
        "temperature": 0.2
    }

    # Make request to endpoint
    response = client.post("/convert", json=test_data)

    # Print response for debugging
    print("\nResponse status code:", response.status_code)
    print("\nResponse data:")
    print(json.dumps(response.json(), indent=2))

    # Assert response status code
    assert response.status_code == 200

    # Parse response
    response_data = response.json()

    # Assert response structure
    assert "conversion_id" in response_data
    assert "output" in response_data
    assert "format" in response_data
    assert "timestamp" in response_data

    try:
        # Parse the output JSON
        output_json = json.loads(response_data["output"])
    except json.JSONDecodeError as e:
        print("\nError parsing output JSON:")
        print(f"Error: {str(e)}")
        print("\nRaw output:")
        print(response_data["output"])
        raise

    # Assert basic product information
    product = output_json["product"]
    assert product["id"] == "f47ac10b-58cc-4372-a567-0e02b2c3d479"
    assert product["name"] == "SecureGuard Pro X7 Smart Security System"
    assert product["category"] == "Electronics"
    assert product["price"]["amount"] == 499.99
    assert product["price"]["currency"] == "USD"

    # Assert availability information
    assert product["availability"]["inStock"] is True
    assert product["availability"]["quantity"] == 42
    assert product["availability"]["backorderAllowed"] is True

    # Assert specifications
    assert len(product["specifications"]) >= 6
    resolution_spec = next(s for s in product["specifications"] if s["name"] == "Resolution")
    assert resolution_spec["value"] == "4K Ultra HD"
    assert resolution_spec["unit"] == "pixels"

    # Assert variants
    assert len(product["variants"]) == 3
    starter_package = next(v for v in product["variants"] if v["attributes"]["package"] == "Starter Package")
    assert starter_package["price"] == 499.99
    assert starter_package["attributes"]["cameraCount"] == "2"
    assert starter_package["attributes"]["color"] == "White"

    # Assert reviews
    assert product["reviews"]["average"] == 4.8
    assert product["reviews"]["count"] == 127
    assert len(product["reviews"]["featured"]) == 2
    first_review = product["reviews"]["featured"][0]
    assert first_review["rating"] == 5
    assert first_review["helpfulVotes"] == 42

    # Assert metadata
    assert product["metadata"]["createdAt"] == "2025-02-28T09:00:00Z"
    assert product["metadata"]["updatedAt"] == "2025-03-28T14:30:00Z"
    assert len(product["metadata"]["tags"]) > 0
    assert len(product["metadata"]["searchKeywords"]) > 0

    # Assert format
    assert response_data["format"] == "json"

    # Assert timestamp is recent
    timestamp = datetime.fromisoformat(response_data["timestamp"].replace("Z", "+00:00"))
    assert (datetime.now() - timestamp).total_seconds() < 60  # Within last minute 