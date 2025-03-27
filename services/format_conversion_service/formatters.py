# services/format_conversion_service/formatters.py

import json
import yaml
import csv
import io
import xml.dom.minidom
import re
from typing import Dict, List, Optional, Union, Any
from enum import Enum

class OutputFormat(str, Enum):
    JSON = "json"
    YAML = "yaml"
    CSV = "csv"
    XML = "xml"
    MARKDOWN = "markdown"
    HTML = "html"
    SQL = "sql"

class FormatValidator:
    """Validates and prettifies formatted outputs"""
    
    @staticmethod
    def validate_json(text: str) -> tuple[bool, str]:
        """Validate and prettify JSON"""
        try:
            json_obj = json.loads(text)
            return True, json.dumps(json_obj, indent=2)
        except json.JSONDecodeError:
            return False, text
    
    @staticmethod
    def validate_yaml(text: str) -> tuple[bool, str]:
        """Validate and prettify YAML"""
        try:
            yaml_obj = yaml.safe_load(text)
            return True, yaml.dump(yaml_obj, sort_keys=False, default_flow_style=False)
        except yaml.YAMLError:
            return False, text
    
    @staticmethod
    def validate_csv(text: str) -> tuple[bool, str]:
        """Validate CSV format"""
        try:
            # Try to parse as CSV
            csv_reader = csv.reader(io.StringIO(text))
            rows = list(csv_reader)
            
            if not rows:
                return False, text
            
            # Check if all rows have the same number of columns
            num_cols = len(rows[0])
            for row in rows[1:]:
                if len(row) != num_cols:
                    return False, text
            
            return True, text
        except Exception:
            return False, text
    
    @staticmethod
    def validate_xml(text: str) -> tuple[bool, str]:
        """Validate and prettify XML"""
        try:
            dom = xml.dom.minidom.parseString(text)
            pretty_xml = dom.toprettyxml(indent="  ")
            # Remove extra whitespace that minidom sometimes adds
            pretty_xml = '\n'.join([line for line in pretty_xml.split('\n') if line.strip()])
            return True, pretty_xml
        except Exception:
            return False, text
    
    @staticmethod
    def validate_sql(text: str) -> tuple[bool, str]:
        """Basic SQL validation and formatting"""
        # This is a very basic check - a real implementation would use a SQL parser
        sql_keywords = ["SELECT", "FROM", "WHERE", "INSERT", "UPDATE", "DELETE", "CREATE", "ALTER", "DROP", "TABLE"]
        
        # Check if the text contains SQL keywords
        has_keywords = any(keyword in text.upper() for keyword in sql_keywords)
        
        if not has_keywords:
            return False, text
        
        # Basic SQL formatting
        formatted = text
        
        # Capitalize SQL keywords
        for keyword in sql_keywords:
            pattern = re.compile(r'\b' + keyword + r'\b', re.IGNORECASE)
            formatted = pattern.sub(keyword, formatted)
        
        # Add newlines after certain keywords
        for keyword in ["SELECT", "FROM", "WHERE", "GROUP BY", "ORDER BY", "HAVING"]:
            pattern = re.compile(r'\b' + keyword + r'\b', re.IGNORECASE)
            formatted = pattern.sub("\n" + keyword, formatted)
        
        return True, formatted

class FormatMerger:
    """Merges multiple formatted chunks into a single output"""
    
    @staticmethod
    def merge_json(chunks: List[str]) -> str:
        """Merge multiple JSON chunks"""
        combined_result = {}
        
        for chunk in chunks:
            try:
                chunk_json = json.loads(chunk)
                if isinstance(chunk_json, dict):
                    combined_result.update(chunk_json)
                elif isinstance(chunk_json, list):
                    if not combined_result:
                        combined_result = []
                    if isinstance(combined_result, list):
                        combined_result.extend(chunk_json)
            except json.JSONDecodeError:
                continue
        
        return json.dumps(combined_result, indent=2)
    
    @staticmethod
    def merge_yaml(chunks: List[str]) -> str:
        """Merge multiple YAML chunks"""
        combined_result = {}
        
        for chunk in chunks:
            try:
                chunk_yaml = yaml.safe_load(chunk)
                if isinstance(chunk_yaml, dict):
                    combined_result.update(chunk_yaml)
                elif isinstance(chunk_yaml, list):
                    if not combined_result:
                        combined_result = []
                    if isinstance(combined_result, list):
                        combined_result.extend(chunk_yaml)
            except yaml.YAMLError:
                continue
        
        return yaml.dump(combined_result, sort_keys=False, default_flow_style=False)
    
    @staticmethod
    def merge_csv(chunks: List[str]) -> str:
        """Merge multiple CSV chunks"""
        if not chunks:
            return ""
        
        combined_rows = []
        header = None
        
        for i, chunk in enumerate(chunks):
            # Parse CSV
            csv_reader = csv.reader(io.StringIO(chunk))
            rows = list(csv_reader)
            
            if not rows:
                continue
            
            if i == 0:
                # Keep the header from the first chunk
                header = rows[0]
                combined_rows.extend(rows)
            else:
                # Skip the header for subsequent chunks
                combined_rows.extend(rows[1:] if len(rows) > 1 else [])
        
        # Write back to CSV
        output = io.StringIO()
        csv_writer = csv.writer(output)
        csv_writer.writerows(combined_rows)
        return output.getvalue()
    
    @staticmethod
    def merge_xml(chunks: List[str]) -> str:
        """Merge multiple XML chunks"""
        # This is a simplified approach - proper XML merging is complex
        # We'll wrap all chunks in a root element
        merged = "<root>\n"
        
        for chunk in chunks:
            # Try to extract content inside root tags if present
            root_match = re.search(r'<\?xml.*?\?>\s*<([^>]+)>(.*?)</\1>', chunk, re.DOTALL)
            if root_match:
                merged += root_match.group(2) + "\n"
            else:
                merged += chunk + "\n"
        
        merged += "</root>"
        
        # Try to prettify
        try:
            dom = xml.dom.minidom.parseString(merged)
            pretty_xml = dom.toprettyxml(indent="  ")
            # Remove extra whitespace
            pretty_xml = '\n'.join([line for line in pretty_xml.split('\n') if line.strip()])
            return pretty_xml
        except Exception:
            return merged
    
    @staticmethod
    def merge_markdown(chunks: List[str]) -> str:
        """Merge multiple Markdown chunks"""
        return "\n\n".join(chunks)
    
    @staticmethod
    def merge_html(chunks: List[str]) -> str:
        """Merge multiple HTML chunks"""
        merged = "<!DOCTYPE html>\n<html>\n<body>\n"
        
        for chunk in chunks:
            # Try to extract content inside body tags if present
            body_match = re.search(r'<body>(.*?)</body>', chunk, re.DOTALL)
            if body_match:
                merged += body_match.group(1) + "\n"
            else:
                merged += chunk + "\n"
        
        merged += "</body>\n</html>"
        return merged
    
    @staticmethod
    def merge_sql(chunks: List[str]) -> str:
        """Merge multiple SQL chunks"""
        return "\n\n".join(chunks)
