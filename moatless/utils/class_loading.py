import importlib
import logging
import os
import pkgutil
import sys
from typing import Dict, Type, Any

logger = logging.getLogger(__name__)

class DynamicClassLoadingMixin:
    """Mixin providing dynamic class loading functionality"""
    
    @classmethod
    def _load_classes(cls, base_package: str, base_class: Type) -> Dict[str, Type[Any]]:
        """Load all subclasses from the specified package"""
        logger.debug(f"Loading classes for {base_class.__name__} from {base_package}")

        registered_classes = {}

        # Load built-in classes first
        package = importlib.import_module(base_package)
        registered_classes.update(cls._scan_for_classes(package.__path__, base_package, base_class))
        
        # Support custom path via environment variable
        custom_path = os.getenv("MOATLESS_COMPONENTS_PATH")
        if custom_path and os.path.isdir(custom_path):
            try:
                # Find root package directory
                root_dir = custom_path
                while os.path.exists(os.path.join(os.path.dirname(root_dir), '__init__.py')):
                    root_dir = os.path.dirname(root_dir)
                
                package_name = os.path.basename(root_dir)
                parent_dir = os.path.dirname(root_dir)
                sys.path.insert(0, parent_dir)
                
                # Look for components in the appropriate subpackage
                component_type = cls.get_component_type()  # e.g., "selector", "value_function"
                component_path = os.path.join(root_dir, component_type)
                
                if os.path.isdir(component_path):
                    logger.debug(f"Scanning custom {component_type} components in {component_path}")
                    registered_classes.update(
                        cls._scan_classes_in_paths([component_path], f"{package_name}.{component_type}", base_class)
                    )
                
            finally:
                sys.path.pop(0)
        elif custom_path:
            logger.warning(f"Custom path {custom_path} is not a directory")

        logger.debug(f"Registered {cls.get_component_type()} classes: {registered_classes.keys()}")
        return registered_classes

    @classmethod
    def _scan_for_classes(cls, paths, base_package: str, base_class: Type) -> Dict[str, Type[Any]]:
        return cls._scan_classes_in_paths(paths, base_package, base_class)

    @classmethod
    def _scan_classes_in_paths(cls, paths: list[str], package_prefix: str, base_class: Type) -> Dict[str, Type[Any]]:
        registered_classes = {}
        for finder, modname, ispkg in pkgutil.walk_packages(paths, prefix=package_prefix + '.'):
            try:
                module = importlib.import_module(modname)
                for name, obj in module.__dict__.items():
                    if (isinstance(obj, type) and 
                        issubclass(obj, base_class) and 
                        obj != base_class and 
                        not getattr(obj, '__abstractmethods__', False)):
                        if name in registered_classes:
                            logger.debug(f"Duplicate class: {name} from {modname}")
                        else:
                            logger.debug(f"Loaded {base_class.__name__}: {name} from {modname}")
                            registered_classes[name] = obj
                    
            except Exception as e:
                logger.exception(f"Failed to load from module {modname}")
        return registered_classes
