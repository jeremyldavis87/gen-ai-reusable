# utilities/logging_utils.py
"""Logging utilities for structured logging."""

import logging
import structlog
from typing import Any, Dict, Optional

class StructuredLogger:
    """Structured logger for consistent logging across services."""
    
    def __init__(self, service_name: str):
        """Initialize structured logger.
        
        Args:
            service_name: Name of the service using the logger
        """
        self.logger = structlog.get_logger(service_name)
        self.service_name = service_name
        
        # Configure structlog
        structlog.configure(
            processors=[
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.add_log_level,
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(),
            wrapper_class=structlog.BoundLogger,
            cache_logger_on_first_use=True,
        )
    
    def _log(self, level: str, message: str, request_id: Optional[str] = None, extra: Optional[Dict[str, Any]] = None):
        """Log a message with structured data.
        
        Args:
            level: Log level
            message: Log message
            request_id: Optional request ID for tracing
            extra: Optional additional structured data
        """
        log_data = {
            "service": self.service_name,
            "message": message
        }
        
        if request_id:
            log_data["request_id"] = request_id
            
        if extra:
            log_data.update(extra)
            
        logger = self.logger.bind(**log_data)
        getattr(logger, level)(message)
    
    def info(self, message: str, request_id: Optional[str] = None, extra: Optional[Dict[str, Any]] = None):
        """Log an info message.
        
        Args:
            message: Log message
            request_id: Optional request ID for tracing
            extra: Optional additional structured data
        """
        self._log("info", message, request_id, extra)
    
    def warning(self, message: str, request_id: Optional[str] = None, extra: Optional[Dict[str, Any]] = None):
        """Log a warning message.
        
        Args:
            message: Log message
            request_id: Optional request ID for tracing
            extra: Optional additional structured data
        """
        self._log("warning", message, request_id, extra)
    
    def error(self, message: str, request_id: Optional[str] = None, extra: Optional[Dict[str, Any]] = None):
        """Log an error message.
        
        Args:
            message: Log message
            request_id: Optional request ID for tracing
            extra: Optional additional structured data
        """
        self._log("error", message, request_id, extra)
    
    def debug(self, message: str, request_id: Optional[str] = None, extra: Optional[Dict[str, Any]] = None):
        """Log a debug message.
        
        Args:
            message: Log message
            request_id: Optional request ID for tracing
            extra: Optional additional structured data
        """
        self._log("debug", message, request_id, extra)
    
    def critical(self, message: str, request_id: Optional[str] = None, extra: Optional[Dict[str, Any]] = None):
        """Log a critical message.
        
        Args:
            message: Log message
            request_id: Optional request ID for tracing
            extra: Optional additional structured data
        """
        self._log("critical", message, request_id, extra)
