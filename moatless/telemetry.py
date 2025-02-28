"""OpenTelemetry tracing for Moatless."""

import contextvars
import logging
import os
import asyncio
from contextlib import contextmanager, asynccontextmanager
from typing import Optional, Dict, Any, Iterator, Callable, TypeVar, ParamSpec, Literal, Union, AsyncIterator
from functools import wraps
from rq import Queue
from opentelemetry import trace, context
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Span, Status, StatusCode
from opentelemetry.propagate import inject, extract
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

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


def setup_azure_monitor() -> None:
    """Set up Azure Monitor OpenTelemetry integration.
    
    Args:
        service_name: Name of the service
        logger_name: Optional logger namespace for collecting telemetry
        resource_attributes: Additional resource attributes
    """
    if not HAS_AZURE_MONITOR:
        logger.warning("Azure Monitor OpenTelemetry package not found. "
            "Install with: pip install azure-monitor-opentelemetry")
        return
    
    if not os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING"):
        raise ValueError(
            "APPLICATIONINSIGHTS_CONNECTION_STRING environment variable not set. "
            "Please set it with your Azure Application Insights connection string."
        )
    logger.info(f"Setting up Azure Monitor OpenTelemetry tracing")
    
    # Configure with shorter timeout for batch span processor
    configure_azure_monitor(
        connection_string=os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING"),
        service_name="moatless-tools",
        export_timeout_millis=5000
    )

    global _tracer
    _tracer = trace.get_tracer(__name__)
    
    logger.info(f"Initialized Azure Monitor OpenTelemetry tracing")

def setup_telemetry() -> None:
    """Set up OpenTelemetry tracing.
    
    """
    logger.info(f"Setting up OpenTelemetry tracing")
    otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    app_insights_conn_string = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
    
    # Choose telemetry provider based on environment variables
    if app_insights_conn_string:
        setup_azure_monitor()
    else:
        attributes = {"service.name": "moatless"}
        resource = Resource.create(attributes)

        # Set up tracer provider
        provider = TracerProvider(resource=resource)
        logger.info(f"Setting up OpenTelemetry provider")

        # Configure exporter
        if otlp_endpoint:
            otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
            provider.add_span_processor(BatchSpanProcessor(otlp_exporter))

        # Set global tracer provider
        trace.set_tracer_provider(provider)
        
    # Get tracer
    global _tracer
    _tracer = trace.get_tracer(__name__)
    
    logger.info(f"Initialized OpenTelemetry tracing")

def get_tracer():
    """Get the global tracer instance."""
    if not _tracer:
        setup_telemetry()
    return _tracer

@contextmanager
def _sync_instrument_span(
    name: str,
    attributes: Optional[Dict[str, Any]] = None,
    kind: Optional[trace.SpanKind] = None,
) -> Iterator[Span]:
    """Synchronous context manager for creating trace spans."""
    tracer = get_tracer()
    with tracer.start_as_current_span(name, attributes=attributes, kind=kind) as span:
        try:
            yield span
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR))
            span.record_exception(e)
            raise

@asynccontextmanager
async def _async_instrument_span(
    name: str,
    attributes: Optional[Dict[str, Any]] = None,
    kind: Optional[trace.SpanKind] = None,
):
    """Asynchronous context manager for creating trace spans."""
    tracer = get_tracer()
    with tracer.start_as_current_span(
        name,
        attributes=attributes,
        kind=kind,
    ) as span:
        try:
            yield span
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR))
            span.record_exception(e)
            raise

def instrument_span(
    name: str,
    attributes: Optional[Dict[str, Any]] = None,
    kind: Optional[trace.SpanKind] = None,
) -> Union[Iterator[Span], AsyncIterator[Span]]:
    """Context manager for creating trace spans. Works with both sync and async code.
    
    Args:
        name: Name of the span
        attributes: Optional span attributes
        kind: Optional span kind
        
    Returns:
        Context manager that yields the created span
    """
    if asyncio.get_event_loop().is_running():
        return _async_instrument_span(name, attributes, kind)
    return _sync_instrument_span(name, attributes, kind)

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
            async with _async_instrument_span(span_name, attributes=attributes, kind=kind) as span:
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
            with _sync_instrument_span(span_name, attributes=attributes, kind=kind) as span:
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

def extract_trace_context() -> Dict[str, str]:
    """Extract current trace context into carrier dict.
    
    This is useful for propagating trace context across process boundaries,
    like when using RQ for job queues.
    
    Returns:
        Dict containing W3C trace context headers
    """
    carrier = {}
    TraceContextTextMapPropagator().inject(carrier)
    return carrier

def restore_trace_context(carrier: Dict[str, str]) -> None:
    """Restore trace context from carrier dict.
    
    This should be called in the worker process to restore the trace context
    that was extracted in the parent process.
    
    Args:
        carrier: Dict containing W3C trace context headers from extract_trace_context()
    """
    ctx = TraceContextTextMapPropagator().extract(carrier=carrier)
    context.attach(ctx)

def wrap_with_trace(
    name: str,
    attributes: Optional[Dict[str, Any]] = None,
    kind: Optional[trace.SpanKind] = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator that wraps a function with trace context handling and span creation.
    
    This is particularly useful for RQ job functions where we want to:
    1. Restore the trace context from the parent process
    2. Create a new span for the job
    3. Handle errors appropriately
    4. Restore context data variables
    
    Args:
        name: Name for the span
        attributes: Optional static span attributes
        kind: Optional span kind
        
    Returns:
        Decorator function that handles trace context
    """
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            # Extract trace context and context data from kwargs if present
            trace_context = kwargs.pop("_trace_context", {})
            context_data = kwargs.pop("_context_data", {})
            
            # Restore trace context if present
            if trace_context:
                restore_trace_context(trace_context)
                
            # Restore context data if present
            context_tokens = []
            if context_data:
                for var_name, value in context_data.items():
                    if hasattr(contextvars, var_name):
                        var = getattr(contextvars, var_name)
                        token = var.set(value)
                        context_tokens.append((var, token))
            
            # Get any dynamic attributes
            span_attributes = {}
            if attributes:
                span_attributes.update(attributes)
            
            # Start new span
            with instrument_span(name, attributes=span_attributes, kind=kind) as span:
                try:
                    result = func(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR), str(e))
                    span.record_exception(e)
                    logger.exception(f"Error in {name}")
                    raise
                finally:
                    # Reset context vars
                    for var, token in context_tokens:
                        var.reset(token)
                
        return wrapper
    return decorator

def extract_context_data() -> Dict[str, Any]:
    """Extract current context data into a dict.
    
    This is useful for propagating context data across process boundaries,
    like when using RQ for job queues.
    
    Returns:
        Dict containing context variable values
    """
    from moatless.context_data import moatless_dir, current_node_id, current_trajectory_id, current_project_id
    context_data = {
        "moatless_dir": moatless_dir.get(),
        "current_node_id": current_node_id.get(),
        "current_trajectory_id": current_trajectory_id.get(),
        "current_project_id": current_project_id.get(),
    }
    logger.info(f"extract_context_data: {context_data}")
    return context_data 



def run_async(coro, span_name: Optional[str] = None):
    """Helper to run coroutines in synchronous context while preserving trace context.
    
    This function ensures that the OpenTelemetry trace context and Moatless context data
    is properly propagated to the event loop and coroutine execution.
    
    Args:
        coro: The coroutine to run
        span_name: Optional name for the span if we want to create one
        
    Returns:
        The result of the coroutine execution
    """
    # Get current trace context and span
    current_context = context.get_current()
    current_span = trace.get_current_span()
    tracer = get_tracer()
    
    # Capture current context data
    from moatless.context_data import moatless_dir, current_node_id, current_trajectory_id, current_project_id
    context_data = {
        "moatless_dir": moatless_dir.get(),
        "current_node_id": current_node_id.get(),
        "current_trajectory_id": current_trajectory_id.get(),
        "current_project_id": current_project_id.get(),
    }
    
    # Create wrapper to run coroutine with trace context
    async def _run_with_context():
        try:
            # Restore context data
            tokens = []
            if "current_trajectory_id" in context_data:
                current_trajectory_id.set(context_data["current_trajectory_id"])
            if "current_project_id" in context_data:
                current_project_id.set(context_data["current_project_id"])
                    
            try:
                # Create a new span if name is provided, otherwise just run with current context
                if span_name:
                    tracer = get_tracer()
                    with tracer.start_as_current_span(
                        span_name,
                        context=current_context if current_span else None
                    ) as span:
                        return await coro
                else:
                    return await coro
            finally:
                # Reset context vars
                for var, token in tokens:
                    var.reset(token)
                    
        except Exception as e:
            if current_span:
                current_span.record_exception(e)
            raise
    
    # Get or create event loop
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
    # Run coroutine with context
    return loop.run_until_complete(_run_with_context())
