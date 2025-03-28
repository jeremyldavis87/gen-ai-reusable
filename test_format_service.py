#!/usr/bin/env python3

"""
Test script for the Format Conversion Service

This script allows testing the Format Conversion Service locally without
relying on external API calls. It sends requests to the service and
displays the responses.
"""

import requests
import json
import argparse
from enum import Enum


class OutputFormat(str, Enum):
    JSON = "json"
    YAML = "yaml"
    CSV = "csv"
    XML = "xml"
    MARKDOWN = "markdown"
    HTML = "html"
    SQL = "sql"


def test_convert_text(text, instructions, output_format):
    """Test the /convert endpoint"""
    url = "http://localhost:8000/convert"
    
    payload = {
        "text": text,
        "instructions": instructions,
        "output_format": output_format
    }
    
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json"
    }
    
    response = requests.post(url, json=payload, headers=headers)
    
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        print("Response:")
        try:
            response_json = response.json()
            print(json.dumps(response_json, indent=2))
            print("\nFormatted Output:")
            print(response_json["output"])
        except Exception as e:
            print(f"Error parsing response: {e}")
            print(response.text)
    else:
        print(f"Error: {response.text}")


def main():
    parser = argparse.ArgumentParser(description="Test the Format Conversion Service")
    parser.add_argument(
        "--format", 
        type=str, 
        choices=[f.value for f in OutputFormat], 
        default="json",
        help="Output format"
    )
    parser.add_argument(
        "--text", 
        type=str, 
        default="Name: John Smith\nAge: 35\nOccupation: Software Engineer\nSkills: Python, JavaScript, Docker",
        help="Text to convert"
    )
    parser.add_argument(
        "--instructions", 
        type=str, 
        default="Convert this person information into structured data",
        help="Instructions for conversion"
    )
    
    args = parser.parse_args()
    
    test_convert_text(args.text, args.instructions, args.format)


if __name__ == "__main__":
    main()
