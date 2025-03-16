from __future__ import annotations  # type: ignore

import importlib
import logging
import os
import pkgutil
import sys
from abc import ABC
from collections.abc import Mapping
from typing import Any, ClassVar, Dict, Generic, Type, TypeVar, cast

from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T")  # TODO: Add bound="MoatlessComponent" once mypy doesnt crash when it's used

# ComponentType -> {QualifiedClassName -> ComponentClass}
_GLOBAL_COMPONENT_CACHE: dict[str, dict[str, type[MoatlessComponent]]] = {}


class MoatlessComponent(BaseModel, ABC, Generic[T]):
    """Base class for dynamically loadable components.

    This class provides functionality to:
    1. Automatically discover and load component classes from a package
    2. Support custom components via MOATLESS_COMPONENTS_PATH
    3. Handle component serialization and deserialization
    4. Manage component type registration with global caching

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

    @classmethod
    def model_validate(cls, obj: Any):
        if not isinstance(obj, dict):
            return obj

        obj = obj.copy()
        discriminator_key = f"{cls.get_component_type()}_class"

        # If there no discriminator we expect it to be a regular initalization of the sub class
        if discriminator_key not in obj:
            return super().model_validate(obj)

        classpath = obj.pop(discriminator_key, None)
        if not classpath:
            raise ValueError(f"No {cls.get_component_type()} class path found on {obj}")

        try:
            component_class = cls.get_component_by_classpath(classpath)

            if not component_class:
                available = list(cls._get_components().keys())
                logger.warning(f"Invalid {cls.get_component_type()} class: {classpath}. Available: {available}")
                raise ValueError(f"Invalid {cls.get_component_type()} class: {classpath}")
            return component_class.model_validate(obj)
        except Exception as e:
            raise Exception(f"Failed to load {cls.get_component_type()} class {classpath}: {e}") from e

    def model_dump(self, *args, **kwargs) -> dict[str, Any]:
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
    def get_component_by_classpath(cls, classpath: str) -> type[T] | None:
        """Get a component class by its name."""
        cls._initialize_components()
        components = cls._get_components()
        if classpath in components:
            return cast(type[T], components[classpath])
        return None

    @classmethod
    def get_available_components(cls) -> dict[str, type[T]]:
        """Get all available component classes."""
        cls._initialize_components()
        components = cls._get_components()
        return cast(dict[str, type[T]], components)

    @classmethod
    def _initialize_components(cls):
        component_type = cls.get_component_type()
        if component_type not in _GLOBAL_COMPONENT_CACHE:
            result = cls._scan_classes_in_paths(cls._get_package(), cls._get_base_class())
            _GLOBAL_COMPONENT_CACHE[component_type] = cast(dict[str, type["MoatlessComponent"]], result)

    @classmethod
    def _get_components(cls) -> dict[str, type[T]]:
        component_type = cls.get_component_type()
        if component_type not in _GLOBAL_COMPONENT_CACHE:
            cls._initialize_components()
        return cast(dict[str, type[T]], _GLOBAL_COMPONENT_CACHE[component_type])

    @classmethod
    def _get_package(cls) -> str:
        """Get the package name for this component type."""
        raise NotImplementedError

    @classmethod
    def _get_base_class(cls) -> type[T]:
        """Get the base class for this component type."""
        raise NotImplementedError

    @classmethod
    def _scan_classes_in_paths(cls, package: str, base_class: type[T]) -> dict[str, type[T]]:
        """Scan for component classes in a package."""
        registered_classes: dict[str, type[T]] = {}
        try:
            logger.debug(f"Scanning package {package} for {base_class.__name__} subclasses")
            package_module = importlib.import_module(package)
            package_path = package_module.__path__

            # First scan the package itself
            for finder, modname, ispkg in pkgutil.walk_packages(package_path, prefix=package + "."):
                try:
                    module = importlib.import_module(modname)
                    logger.debug(f"Loading module {modname}: module.__dict__: {module.__dict__.keys()}")
                    for name, obj in module.__dict__.items():
                        try:
                            if (
                                isinstance(obj, type)
                                and issubclass(obj, base_class)
                                and not getattr(obj, "__abstractmethods__", False)
                                and not name.endswith("Mixin")
                                and ABC not in obj.__bases__
                            ):
                                qualified_name = f"{obj.__module__}.{name}"
                                if qualified_name in registered_classes:
                                    logger.debug(f"Duplicate class name: {qualified_name} from {modname}")
                                else:
                                    logger.debug(f"Found {base_class.__name__} subclass: {qualified_name}")
                                    registered_classes[qualified_name] = obj
                            else:
                                # log why it was skipped
                                if not isinstance(obj, type):
                                    logger.debug(f"Skipping class {name} from {modname} because not a type: {obj}")
                                elif not issubclass(obj, base_class):
                                    logger.debug(
                                        f"Skipping class {name} from {modname} because not a subclass of {base_class.__name__}: {obj}"
                                    )
                                elif obj == base_class:
                                    logger.debug(
                                        f"Skipping class {name} from {modname} because it is the base class: {obj}"
                                    )
                                elif getattr(obj, "__abstractmethods__", False):
                                    logger.debug(
                                        f"Skipping class {name} from {modname} because it has abstract methods: {obj}: {getattr(obj, '__abstractmethods__', False)}"
                                    )
                                elif name.endswith("Mixin"):
                                    logger.debug(f"Skipping class {name} from {modname} because it is a mixin: {obj}")
                                elif ABC in obj.__bases__:
                                    logger.debug(
                                        f"Skipping class {name} from {modname} because it directly inherits from ABC: {obj}"
                                    )

                        except TypeError as e:
                            logger.debug(
                                f"Skipping class {name} from {modname} because of TypeError: {e}, issubclass: {issubclass(obj, base_class)}, obj != base_class: {obj != base_class}, not getattr(obj, '__abstractmethods__', False): {not getattr(obj, '__abstractmethods__', False)}, not name.endswith('Mixin'): {not name.endswith('Mixin')}"
                            )
                            # Skip objects that can't be checked with issubclass
                            continue
                except Exception as e:
                    logger.debug(f"Failed to load from module {modname}: {e}")

            # Log all found classes
            logger.debug(f"Found {len(registered_classes)} classes in {package}: {list(registered_classes.keys())}")

            # Then check for custom components if MOATLESS_COMPONENTS_PATH is set
            custom_path = os.getenv("MOATLESS_COMPONENTS_PATH")
            if custom_path and os.path.isdir(custom_path):
                logger.debug(f"Custom components path found: {custom_path}")
                sys.path.insert(0, os.path.dirname(custom_path))
                try:
                    package_name = os.path.basename(custom_path)
                    for finder, modname, ispkg in pkgutil.walk_packages([custom_path], prefix=f"{package_name}."):
                        try:
                            module = importlib.import_module(modname)
                            for name, obj in module.__dict__.items():
                                if (
                                    isinstance(obj, type)
                                    and issubclass(obj, base_class)
                                    and obj != base_class
                                    and not getattr(obj, "__abstractmethods__", False)
                                    and ABC not in obj.__bases__
                                ):  # Skip classes directly inheriting from ABC
                                    qualified_name = f"{obj.__module__}.{name}"
                                    if qualified_name in registered_classes:
                                        logger.debug(f"Duplicate class: {qualified_name} from {modname}")
                                    else:
                                        logger.debug(
                                            f"Loaded custom {base_class.__name__}: {qualified_name} from {modname}"
                                        )
                                        registered_classes[qualified_name] = obj
                        except Exception as e:
                            logger.exception(f"Failed to load from custom module {modname}: {e}")
                finally:
                    sys.path.pop(0)
            else:
                logger.debug(f"No custom components path found for {cls.get_component_type()}")
        except Exception as e:
            logger.exception(f"Failed to scan package {package}: {e}")

        if not registered_classes:
            logger.warning(f"No {cls.get_component_type()} classes found")

        return registered_classes
