"""
Utilities for filtering warnings in the moatless project.
"""

import logging
import warnings

logger = logging.getLogger(__name__)


def filter_external_warnings():
    """
    Filter out warnings from external dependencies that we can't control.
    This should be called early in the application startup.
    """
    # Filter Pydantic deprecation warnings from dependencies
    warnings.filterwarnings(
        "ignore", message="Support for class-based `config` is deprecated", category=DeprecationWarning
    )

    # Filter litellm importlib warnings
    warnings.filterwarnings("ignore", message="open_text is deprecated", category=DeprecationWarning)

    # Filter json_encoders warnings from dependencies
    warnings.filterwarnings("ignore", message="`json_encoders` is deprecated", category=DeprecationWarning)

    logger.debug("External dependency warnings have been filtered")
