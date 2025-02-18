from abc import ABC
from typing import Any, Dict, Type, ClassVar
import importlib
import logging
import pkgutil
import os
import sys
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class MoatlessComponent(BaseModel, ABC):
    """Base class for dynamically loadable components.
    
    This class provides functionality to:
    1. Automatically discover and load component classes from a package
    2. Support custom components via MOATLESS_COMPONENTS_PATH
    3. Handle component serialization and deserialization
    4. Manage component type registration
    
    Usage:
        class Action(MoatlessComponent):
            @classmethod
            def get_component_type(cls) -> str:
                return "action"
                
            @classmethod
            def _get_package(cls) -> str:
                return "moatless.actions"
                
            @classmethod
            def _get_base_class(cls) -> Type:
                return Action
    """
    
    _components: ClassVar[Dict[str, Type["MoatlessComponent"]]] = {}
    
    @classmethod
    def model_validate(cls, obj: Any):
        if isinstance(obj, dict):
            obj = obj.copy()
            class_path = obj.pop(f"{cls.get_component_type()}_class", None)
            if class_path:
                try:
                    cls._initialize_components()
                    module_name, class_name = class_path.rsplit(".", 1)
                    component_class = cls.get_component_by_name(class_name)
                    
                    if not component_class:
                        available = cls._get_components().keys()
                        logger.warning(f"Invalid {cls.get_component_type()} class: {class_name}. Available: {available}")
                        raise ValueError(f"Invalid {cls.get_component_type()} class: {class_name}")
                    return component_class(**obj)
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
        """Return the type of component (e.g., 'action', 'selector')"""
        raise NotImplementedError

    @classmethod
    def get_component_by_name(cls, name: str) -> Type["MoatlessComponent"]:
        """Get a component class by its name."""
        cls._initialize_components()
        if name in cls._get_components():
            return cls._get_components()[name]
        for qualified_name, component in cls._get_components().items():
            if qualified_name.endswith(f".{name}"):
                return component
        return None

    @classmethod
    def get_available_components(cls) -> Dict[str, Type["MoatlessComponent"]]:
        """Get all available component classes."""
        cls._initialize_components()
        return cls._get_components()

    @classmethod
    def _initialize_components(cls):
        if not hasattr(cls, "_components") or not cls._components:
            cls._components = cls._scan_classes_in_paths(cls._get_package(), cls._get_base_class())

    @classmethod
    def _get_components(cls) -> Dict[str, Type["MoatlessComponent"]]:
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

    @classmethod
    def _scan_classes_in_paths(cls, package: str, base_class: Type) -> Dict[str, Type[Any]]:
        """Scan for component classes in a package."""
        registered_classes = {}
        try:
            logger.info(f"Scanning package {package} for {base_class.__name__} subclasses")
            package_module = importlib.import_module(package)
            package_path = package_module.__path__
            
            # First scan the package itself
            for finder, modname, ispkg in pkgutil.walk_packages(package_path, prefix=package + '.'):
                try:
                    module = importlib.import_module(modname)
                    logger.info(f"Loading module {modname}: module.__dict__: {module.__dict__.keys()}")
                    for name, obj in module.__dict__.items():
                        try:
                            if (isinstance(obj, type) and 
                                issubclass(obj, base_class) and 
                                obj != base_class and 
                                not getattr(obj, '__abstractmethods__', False) and
                                not name.endswith('Mixin')):  # Skip mixin classes
                                # Use class name as key for backward compatibility
                                if name in registered_classes:
                                    logger.info(f"Duplicate class name: {name} from {modname}")
                                else:
                                    logger.info(f"Found {base_class.__name__} subclass: {name}")
                                    registered_classes[name] = obj
                            else:
                                # log why it was skipped
                                if not isinstance(obj, type):
                                    logger.debug(f"Skipping class {name} from {modname} because not a type: {obj}")
                                elif not issubclass(obj, base_class):
                                    logger.debug(f"Skipping class {name} from {modname} because not a subclass of {base_class.__name__}: {obj}")
                                elif obj == base_class:
                                    logger.debug(f"Skipping class {name} from {modname} because it is the base class: {obj}")
                                elif getattr(obj, '__abstractmethods__', False):
                                    logger.debug(f"Skipping class {name} from {modname} because it has abstract methods: {obj}: {getattr(obj, '__abstractmethods__', False)}")
                                elif name.endswith('Mixin'):
                                    logger.debug(f"Skipping class {name} from {modname} because it is a mixin: {obj}")
                                    
                        except TypeError as e:
                            logger.debug(f"Skipping class {name} from {modname} because of TypeError: {e}, issubclass: {issubclass(obj, base_class)}, obj != base_class: {obj != base_class}, not getattr(obj, '__abstractmethods__', False): {not getattr(obj, '__abstractmethods__', False)}, not name.endswith('Mixin'): {not name.endswith('Mixin')}")
                            # Skip objects that can't be checked with issubclass
                            continue
                except Exception as e:
                    logger.exception(f"Failed to load from module {modname}")

            # Log all found classes
            logger.info(f"Found {len(registered_classes)} classes in {package}: {list(registered_classes.keys())}")

            # Then check for custom components if MOATLESS_COMPONENTS_PATH is set
            custom_path = os.getenv("MOATLESS_COMPONENTS_PATH")
            if custom_path and os.path.isdir(custom_path):
                logger.info(f"Custom components path found: {custom_path}")
                sys.path.insert(0, os.path.dirname(custom_path))
                try:
                    package_name = os.path.basename(custom_path)
                    for finder, modname, ispkg in pkgutil.walk_packages([custom_path], prefix=f"{package_name}."):
                        try:
                            module = importlib.import_module(modname)
                            for name, obj in module.__dict__.items():
                                if (isinstance(obj, type) and 
                                    issubclass(obj, base_class) and 
                                    obj != base_class and 
                                    not getattr(obj, '__abstractmethods__', False)):
                                    qualified_name = f"{obj.__module__}.{name}"
                                    if qualified_name in registered_classes:
                                        logger.info(f"Duplicate class: {qualified_name} from {modname}")
                                    else:
                                        logger.info(f"Loaded custom {base_class.__name__}: {qualified_name} from {modname}")
                                        registered_classes[qualified_name] = obj
                        except Exception as e:
                            logger.exception(f"Failed to load from custom module {modname}")
                finally:
                    sys.path.pop(0)
            else:
                logger.info(f"No custom components path found for {cls.get_component_type()}")
        except Exception as e:
            logger.exception(f"Failed to scan package {package}")

        if not registered_classes:
            logger.warning(f"No {cls.get_component_type()} classes found")
            
        return registered_classes 