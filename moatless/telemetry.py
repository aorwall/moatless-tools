"""OpenTelemetry tracing for Moatless."""

import logging
import os
import asyncio
from contextlib import contextmanager
from typing import Optional, Dict, Any, Iterator, Callable, TypeVar, ParamSpec, Literal, Union
from functools import wraps

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Span, Status, StatusCode

try:
    from azure.monitor.opentelemetry import configure_azure_monitor
    HAS_AZURE_MONITOR = True
except ImportError:
    HAS_AZURE_MONITOR = False

logger = logging.getLogger(__name__)

# Global tracer instance
_tracer = None

# Type variables for generic function signature
P = ParamSpec('P')
R = TypeVar('R')


def setup_azure_monitor(
    service_name: str = "moatless",
    logger_name: Optional[str] = None,
    resource_attributes: Optional[Dict[str, str]] = None,
) -> None:
    """Set up Azure Monitor OpenTelemetry integration.
    
    Args:
        service_name: Name of the service
        logger_name: Optional logger namespace for collecting telemetry
        resource_attributes: Additional resource attributes
    """
    if not HAS_AZURE_MONITOR:
        raise ImportError(
            "Azure Monitor OpenTelemetry package not found. "
            "Install with: pip install azure-monitor-opentelemetry"
        )
    
    if not os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING"):
        raise ValueError(
            "APPLICATIONINSIGHTS_CONNECTION_STRING environment variable not set. "
            "Please set it with your Azure Application Insights connection string."
        )

    configure_azure_monitor(
    )

    global _tracer
    _tracer = trace.get_tracer(__name__)
    
    logger.info(f"Initialized Azure Monitor OpenTelemetry tracing for service {service_name}")

def setup_telemetry(
    service_name: str = "moatless",
    endpoint: Optional[str] = None,
    resource_attributes: Optional[Dict[str, str]] = None,
    provider: Literal["otlp", "azure"] = "otlp",
    logger_name: Optional[str] = None,
) -> None:
    """Set up OpenTelemetry tracing.
    
    Args:
        service_name: Name of the service
        endpoint: Optional OTLP endpoint (e.g. http://localhost:4317)
        resource_attributes: Additional resource attributes
        provider: Telemetry provider to use ("otlp" or "azure")
        logger_name: Optional logger namespace for Azure Monitor
    """
    if provider == "azure":
        setup_azure_monitor(
            service_name=service_name,
            logger_name=logger_name,
            resource_attributes=resource_attributes
        )
        return

    global _tracer

    # Create resource
    attributes = {"service.name": service_name}
    if resource_attributes:
        attributes.update(resource_attributes)
    resource = Resource.create(attributes)

    # Set up tracer provider
    provider = TracerProvider(resource=resource)

    # Configure exporter
    if endpoint:
        otlp_exporter = OTLPSpanExporter(endpoint=endpoint)
        provider.add_span_processor(BatchSpanProcessor(otlp_exporter))

    # Set global tracer provider
    trace.set_tracer_provider(provider)
    
    # Get tracer
    _tracer = trace.get_tracer(__name__)
    
    logger.info(f"Initialized OpenTelemetry tracing for service {service_name}")

def get_tracer():
    """Get the global tracer instance."""
    if not _tracer:
        setup_telemetry()
    return _tracer

@contextmanager
def instrument_span(
    name: str,
    attributes: Optional[Dict[str, Any]] = None,
    kind: Optional[trace.SpanKind] = None,
) -> Iterator[Span]:
    """Context manager for creating trace spans.
    
    Args:
        name: Name of the span
        attributes: Optional span attributes
        kind: Optional span kind
        
    Yields:
        The created span
    """
    tracer = get_tracer()
    with tracer.start_as_current_span(name, attributes=attributes, kind=kind) as span:
        try:
            yield span
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR))
            span.record_exception(e)
            raise

def instrument(
    name: Optional[Union[str, Callable[..., str]]] = None,
    attributes: Optional[Dict[str, Any]] = None,
    kind: Optional[trace.SpanKind] = None
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator for tracing functions using OpenTelemetry.
    
    Args:
        name: Name for the span (defaults to function name). Can be a string or a callable that returns a string.
        attributes: Static span attributes
        kind: Optional span kind
        
    Returns:
        A decorator function that wraps the original function with tracing
    """
    def create_wrapper(func: Callable[P, R]) -> Callable[P, R]:
        def get_span_name(*args, **kwargs) -> str:
            # Get class name if method call
            class_prefix = ""
            if args and hasattr(args[0], func.__name__):
                class_prefix = f"{args[0].__class__.__name__}."

            if callable(name):
                # If first arg is self/cls, pass it to the name function
                if args and hasattr(args[0], func.__name__):
                    return f"{class_prefix}{name(args[0])}"
                return name()
            return f"{class_prefix}{func.__name__}"

        @wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            if name is None:
                span_name = get_span_name(*args, **kwargs)
            else:
                span_name = name
            with instrument_span(span_name, attributes=attributes, kind=kind) as span:
                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR))
                    span.record_exception(e)
                    raise

        @wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            if name is None:
                span_name = get_span_name(*args, **kwargs)
            else:
                span_name = name
            with instrument_span(span_name, attributes=attributes, kind=kind) as span:
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR))
                    span.record_exception(e)
                    raise

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return create_wrapper

def get_current_span() -> Optional[Span]:
    """Get the current active span.
    Returns None if there is no active span.
    """
    return trace.get_current_span()

def set_attribute(key: str, value: Any | None) -> None:
    """Set an attribute on the current span.
    Does nothing if there is no active span.
    
    Args:
        key: Attribute key
        value: Attribute value
    """
    span = get_current_span()
    if span and value is not None:
        span.set_attribute(key, value)

def set_attributes(attributes: Dict[str, Any]) -> None:
    """Set multiple attributes on the current span.
    Does nothing if there is no active span.
    
    Args:
        attributes: Dictionary of attributes to set
    """
    span = get_current_span()
    if span:
        span.set_attributes(attributes)

def set_span_status(span: Span, success: bool, message: Optional[str] = None) -> None:
    """Set the status of a span.
    
    Args:
        span: The span to update
        success: Whether the operation was successful
        message: Optional status message
    """
    if success:
        span.set_status(Status(StatusCode.OK))
    else:
        span.set_status(Status(StatusCode.ERROR, message))

def add_span_event(span: Span, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
    """Add an event to a span.
    
    Args:
        span: The span to add the event to
        name: Name of the event
        attributes: Optional event attributes
    """
    span.add_event(name, attributes=attributes)

def add_span_attributes(span: Span, attributes: Dict[str, Any]) -> None:
    """Add attributes to a span.
    
    Args:
        span: The span to add attributes to
        attributes: The attributes to add
    """
    span.set_attributes(attributes) 