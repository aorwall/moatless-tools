from __future__ import annotations  # type: ignore

import importlib
import logging
import os
import pkgutil
import sys
from abc import ABC
from typing import Any, Generic, TypeVar, cast

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
    def from_dict(cls, data: dict):
        discriminator_key = f"{cls.get_component_type()}_class"
        if discriminator_key not in data:
            logger.error(
                f"Failed to create Discrimnator key {discriminator_key} is missing on {cls.get_component_type()}. Data: {data}"
            )
            raise ValueError(f"Expected discriminator key {discriminator_key} to be set.")

        return cls.model_validate(data)

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
            logger.error(f"Discrimnator key {discriminator_key} is missing on {cls.get_component_type()}. Data: {obj}")
            raise ValueError(f"Expected discrimnator key {discriminator_key} to be set.")

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
            logger.info(
                f"Found {len(registered_classes)} classes in [{package}] for [{cls.get_component_type()}]: {list(registered_classes.keys())}"
            )

            # Then check for custom components if MOATLESS_COMPONENTS_PATH is set
            custom_path = os.getenv("MOATLESS_COMPONENTS_PATH")
            if not custom_path:
                logger.info(
                    f"MOATLESS_COMPONENTS_PATH must be set to load custom components for [{cls.get_component_type()}]."
                )
            elif not os.path.isdir(custom_path):
                logger.warning(
                    f"Custom components path [{custom_path}] does not exist for [{cls.get_component_type()}]"
                )
            else:
                logger.debug(f"Custom components path found: [{custom_path}]")

                custom_classes = []

                sys.path.insert(0, custom_path)
                try:
                    # Look for all Python modules directly in the custom path
                    for finder, modname, ispkg in pkgutil.iter_modules([custom_path]):
                        # Only process directories (packages) in the custom path
                        if ispkg:
                            pkg_path = os.path.join(custom_path, modname)
                            # Now walk all modules in this package
                            for subfinder, submodname, subispkg in pkgutil.walk_packages(
                                [pkg_path], prefix=f"{modname}."
                            ):
                                try:
                                    logger.debug(f"Attempting to import {submodname}")
                                    module = importlib.import_module(submodname)
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
                                                logger.debug(f"Duplicate class: {qualified_name} from {submodname}")
                                            else:
                                                logger.debug(
                                                    f"Loaded custom {base_class.__name__}: {qualified_name} from {submodname}"
                                                )
                                                registered_classes[qualified_name] = obj
                                                custom_classes.append(qualified_name)
                                except Exception as e:
                                    logger.exception(f"Failed to load from custom module {submodname}: {e}")
                except Exception as e:
                    logger.exception(
                        f"Failed to load custom components for [{cls.get_component_type()}] from [{custom_path}]"
                    )
                finally:
                    # Remove the path we added
                    sys.path.pop(0)

                logger.info(
                    f"Found [{len(custom_classes)}] custom [{cls.get_component_type()}] classes in [{custom_path}]: {custom_classes}"
                )

        except Exception as e:
            logger.exception(f"Failed to scan package {package}: {e}")

        return registered_classes
