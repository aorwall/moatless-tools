from abc import ABC
from typing import Any, Dict, Type
import importlib
import logging
from pydantic import BaseModel

from moatless.component import MoatlessComponent

logger = logging.getLogger(__name__)

class FlowComponentMixin(MoatlessComponent):
    """Base mixin for flow components."""
    pass

    @classmethod
    def model_validate(cls, obj: Any):
        if isinstance(obj, dict):
            obj = obj.copy()
            class_path = obj.pop(f"{cls.get_component_type()}_class", None)

            if class_path:
                try:
                    # Ensure components are loaded
                    cls._initialize_components()
                    
                    # Get component class by name
                    module_name, class_name = class_path.rsplit(".", 1)
                    component_class = cls.get_component_by_name(class_name)
                    
                    if not component_class:
                        available = cls._get_components().keys()
                        logger.warning(f"Invalid {cls.get_component_type()} class: {class_name}. Available: {available}")
                        raise ValueError(f"Invalid {cls.get_component_type()} class: {class_name}")
                    
                    return component_class.model_validate(obj)
                except Exception as e:
                    logger.warning(f"Failed to load {cls.get_component_type()} class {class_path}: {e}")
                    raise ValueError(f"Failed to load {cls.get_component_type()} class {class_path}") from e
            else:
                return None

        return obj

    def model_dump(self, *args, **kwargs) -> Dict[str, Any]:
        data = super().model_dump(*args, **kwargs)
        data[f"{self.get_component_type()}_class"] = self.get_class_name()
        return data

    @classmethod
    def get_class_name(cls) -> str:
        return f"{cls.__module__}.{cls.__name__}"

    @classmethod
    def get_component_type(cls) -> str:
        """Return the type of component (e.g., 'selector', 'value_function')"""
        raise NotImplementedError

    @classmethod
    def get_component_by_name(cls, name: str) -> Type["FlowComponentMixin"]:
        """Get a component class by its name."""
        cls._initialize_components()
        return cls._get_components().get(name)

    @classmethod
    def get_available_components(cls) -> Dict[str, Type["FlowComponentMixin"]]:
        """Get all available component classes."""
        cls._initialize_components()
        return cls._get_components()

    @classmethod
    def _initialize_components(cls):
        """Initialize the component class map."""
        if not hasattr(cls, "_components"):
            cls._components = cls._load_classes(cls._get_package(), cls._get_base_class())

    @classmethod
    def _get_components(cls) -> Dict[str, Type["FlowComponentMixin"]]:
        """Get the component class map."""
        if not hasattr(cls, "_components"):
            cls._initialize_components()
        return cls._components

    @classmethod
    def _get_package(cls) -> str:
        """Get the package name for this component type."""
        raise NotImplementedError

    @classmethod
    def _get_base_class(cls) -> Type:
        """Get the base class for this component type."""
        raise NotImplementedError 