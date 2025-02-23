"""Logging configuration module for Moatless.

This module provides a state-of-the-art logging configuration with:
- Structured logging with JSON format option
- Log rotation with size and time-based policies
- Different handlers for different severity levels
- Custom formatters with extra contextual information
- Environment-aware configuration
"""

import logging
import logging.handlers
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from functools import partial
import threading
import traceback

class StructuredJsonFormatter(logging.Formatter):
    """JSON formatter that creates structured logs with extra contextual information."""
    
    def __init__(self, **kwargs):
        super().__init__()
        self.extras = kwargs

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as a JSON string with extra context."""
        message = {
            'timestamp': datetime.utcfromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'thread': threading.current_thread().name,
            'process': os.getpid()
        }

        # Add exception info if present
        if record.exc_info:
            message['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'stacktrace': traceback.format_exception(*record.exc_info)
            }

        # Add custom extras
        message.update(self.extras)

        return json.dumps(message)


def setup_logging(
    log_level: str = "INFO",
    log_dir: Optional[Path] = None,
    service_name: str = "moatless",
    environment: str = "development",
    use_json: bool = False,
    **kwargs
) -> None:
    """Configure logging with advanced features.
    
    Args:
        log_level: Minimum log level to capture
        log_dir: Directory to store log files
        service_name: Name of the service for log identification
        environment: Deployment environment
        use_json: Whether to use JSON structured logging
        **kwargs: Additional configuration options
    """
    # Create log directory if specified
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

    # Base configuration
    config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'detailed': {
                '()': 'logging.Formatter',
                'format': '%(asctime)s | %(levelname)-8s | %(name)s | %(thread)d | %(message)s'
            },
            'json': {
                '()': StructuredJsonFormatter,
                'service': service_name,
                'environment': environment
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'formatter': 'json' if use_json else 'detailed',
                'stream': sys.stdout
            }
        },
        'loggers': {
            'azure': {'level': 'WARNING'},
            'azure.core': {'level': 'WARNING'},
            'azure.identity': {'level': 'WARNING'},
            'azure.storage': {'level': 'WARNING'},
            'azure.monitor': {'level': 'WARNING'},
            'azure.mgmt': {'level': 'WARNING'},
            'msrest': {'level': 'WARNING'},
            'msal': {'level': 'WARNING'},
        },
        'root': {
            'level': log_level.upper(),
            'handlers': ['console']
        }
    }

    # Add file handlers if log directory is specified
    if log_dir:
        config['handlers'].update({
            'file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'filename': str(log_dir / f'{service_name}.log'),
                'formatter': 'json' if use_json else 'detailed',
                'maxBytes': 10 * 1024 * 1024,  # 10MB
                'backupCount': 5
            },
            'error_file': {
                'class': 'logging.handlers.TimedRotatingFileHandler',
                'filename': str(log_dir / f'{service_name}_error.log'),
                'formatter': 'json' if use_json else 'detailed',
                'when': 'midnight',
                'interval': 1,
                'backupCount': 30,
                'level': 'ERROR'
            }
        })
        config['root']['handlers'].extend(['file', 'error_file'])

    # Configure logging
    logging.config.dictConfig(config)

    # Set default logging level for third-party libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

def get_logger(
    name: str,
    extra_fields: Optional[Dict[str, Any]] = None
) -> logging.Logger:
    """Get a logger with optional extra fields.
    
    Args:
        name: Logger name
        extra_fields: Additional fields to include in all log messages
        
    Returns:
        Logger instance with extra fields adapter if specified
    """
    logger = logging.getLogger(name)
    
    if extra_fields:
        # Convert any non-string values to strings to avoid OpenTelemetry warnings
        sanitized_fields = {
            k: str(v) if not isinstance(v, (bool, str, bytes, int, float)) else v
            for k, v in extra_fields.items()
        }
        return logging.LoggerAdapter(logger, {'extra_fields': sanitized_fields})
    
    return logger 