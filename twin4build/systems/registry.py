"""Registry of ``System`` classes that live outside the library.

The built-in systems are resolved by class name from
:mod:`twin4build.systems` and need no registration.  A component class that
is developed in another package -- an ordinary
:class:`~twin4build.systems.saref4syst.system.System` subclass with the same
``initialize`` / ``forward`` / ``do_step`` contract -- is made known to the
library by registering it under a stable, namespaced *type id*::

    import twin4build as tb

    tb.system_registry.register(
        CustomCoilSystem,
        type_id="acme:CoilSystem@1",
        provider="acme-t4b-components",
        version="1.4.2",
    )

A serialized model then records the type id, the provider and the provider
version of every component of a registered class, and
``Model.load(filename=...)`` resolves the class from the type id through
the registry of the model.  Nothing is imported from the model file: a type
that is not registered fails with an :class:`UnknownSystemTypeError` that
names the component, the type id and the provider to install.

**Type id.**  ``<namespace>:<name>`` with an optional ``@<revision>``
suffix.  The type id, not the Python class name or module path, is the
contract between a serialized model and the provider: keep it when a class
is renamed or moved, and change the revision when models written by the old
class can no longer be loaded by the new one.  The provider version is
recorded for diagnostics and reproducibility; it does not take part in the
resolution.

**Default and custom registries.**  :data:`system_registry` (also available
as ``twin4build.system_registry``) is the registry every model uses unless
it is given another one.  An application that wants an explicit whitelist
creates its own :class:`SystemRegistry` and passes it as
``Model(id=..., system_registry=registry)`` and
``Translator(system_registry=registry)``.

**Models serialized before.**  A component without a recorded type id is
resolved by its class name: first among the built-in systems, then among
the registered classes.
"""

from __future__ import annotations

# Standard library imports
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

# Local application imports
import twin4build.systems as systems_module
from twin4build.systems.saref4syst.system import System

#: Literal keys a serialized component of a registered class carries.
TYPE_ID_KEY = "system_type_id"
PROVIDER_KEY = "system_provider"
PROVIDER_VERSION_KEY = "system_provider_version"
SERIALIZED_KEYS = (TYPE_ID_KEY, PROVIDER_KEY, PROVIDER_VERSION_KEY)


class UnknownSystemTypeError(LookupError):
    """A serialized component names a system type that cannot be resolved.

    The attributes repeat what the serialized model records about the
    component: ``component_id``, ``type_id``, ``class_name``, ``provider``
    and ``version`` (``None`` where the model records nothing).
    """

    def __init__(
        self,
        message: str,
        *,
        component_id: Optional[str] = None,
        type_id: Optional[str] = None,
        class_name: Optional[str] = None,
        provider: Optional[str] = None,
        version: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.component_id = component_id
        self.type_id = type_id
        self.class_name = class_name
        self.provider = provider
        self.version = version


@dataclass(frozen=True)
class SystemRegistration:
    """One entry of a :class:`SystemRegistry`.

    Args:
        system: The registered ``System`` class.
        type_id: The stable type id the class is serialized under.
        provider: Name of the package that provides the class.
        version: Version of that package.
    """

    system: type
    type_id: str
    provider: Optional[str] = None
    version: Optional[str] = None

    def literals(self) -> Dict[str, Optional[str]]:
        """What a serialized component of this class records, by literal key."""
        return {
            TYPE_ID_KEY: self.type_id,
            PROVIDER_KEY: self.provider,
            PROVIDER_VERSION_KEY: self.version,
        }


def is_builtin(system: type) -> bool:
    """True for a system class that the library itself exports.

    Decided from the export table of :mod:`twin4build.systems`, so a class
    that was only set as an attribute of that module is not built-in.
    """
    module_name = systems_module._MODULES.get(getattr(system, "__name__", None))
    if module_name is None:
        return False
    return system.__module__ == f"{systems_module.__name__}.{module_name}"


def _check_type_id(type_id) -> str:
    if (
        not isinstance(type_id, str)
        or any(c.isspace() for c in type_id)
        or len(type_id.split(":", 1)) != 2
        or not all(type_id.split(":", 1))
    ):
        raise ValueError(
            "A system type id must be a string of the form "
            f"'<namespace>:<name>' or '<namespace>:<name>@<revision>' without "
            f"whitespace; got {type_id!r}."
        )
    return type_id


def _describe(registration: SystemRegistration) -> str:
    text = f"'{registration.type_id}' ({registration.system.__name__}"
    if registration.provider is not None:
        text += f", provider '{registration.provider}'"
    if registration.version is not None:
        text += f" version {registration.version}"
    return text + ")"


class SystemRegistry:
    """Maps stable type ids to ``System`` classes that are not built in.

    The registry is the explicit list of external component classes an
    application accepts: a model resolves serialized components of external
    classes only through its registry.  Built-in classes are always known
    and cannot be registered.
    """

    def __init__(self) -> None:
        self._by_type_id: Dict[str, SystemRegistration] = {}
        self._by_system: Dict[type, SystemRegistration] = {}

    def __repr__(self) -> str:
        entries = ", ".join(sorted(self._by_type_id))
        return f"{type(self).__name__}([{entries}])"

    def register(
        self,
        system: type,
        *,
        type_id: str,
        provider: Optional[str] = None,
        version: Optional[str] = None,
        replace: bool = False,
    ) -> SystemRegistration:
        """Register ``system`` under ``type_id``.

        Registering the same class again with the same type id, provider and
        version is a no-op, so a provider's registration function can be
        called more than once.

        Args:
            system: A ``System`` subclass that is not built in.
            type_id: ``<namespace>:<name>`` with an optional ``@<revision>``.
            provider: Name of the package that provides the class.
            version: Version of that package; stored as a string.
            replace: Replace what is registered for ``type_id`` and for
                ``system`` instead of raising.

        Returns:
            The registration.

        Raises:
            TypeError: If ``system`` is not a ``System`` subclass.
            ValueError: If ``type_id`` is malformed, if ``system`` is a
                built-in class, or (without ``replace``) if ``type_id`` is
                taken by another class or ``system`` is registered under
                another type id.
        """
        if not (isinstance(system, type) and issubclass(system, System)):
            raise TypeError(
                "SystemRegistry.register expects a System subclass; "
                f"got {system!r}."
            )
        _check_type_id(type_id)
        if is_builtin(system):
            raise ValueError(
                f"{system.__name__} is a built-in twin4build system; built-in "
                "systems are resolved by class name and are not registered."
            )
        registration = SystemRegistration(
            system=system,
            type_id=type_id,
            provider=None if provider is None else str(provider),
            version=None if version is None else str(version),
        )
        by_type_id = self._by_type_id.get(type_id)
        by_system = self._by_system.get(system)
        if by_type_id == registration and by_system == registration:
            return by_type_id
        if not replace:
            if by_type_id is not None:
                raise ValueError(
                    f"The system type id '{type_id}' is already registered: "
                    f"{_describe(by_type_id)}.  Type ids are unique; pass "
                    "replace=True to replace the registration."
                )
            if by_system is not None:
                raise ValueError(
                    f"{system.__name__} is already registered as "
                    f"{_describe(by_system)}.  A class has one type id; pass "
                    "replace=True to replace the registration."
                )
        for stale in (by_type_id, by_system):
            if stale is not None:
                self._by_type_id.pop(stale.type_id, None)
                self._by_system.pop(stale.system, None)
        self._by_type_id[type_id] = registration
        self._by_system[system] = registration
        return registration

    def unregister(self, type_id: str) -> None:
        """Remove the registration of ``type_id``, if there is one."""
        registration = self._by_type_id.pop(type_id, None)
        if registration is not None:
            self._by_system.pop(registration.system, None)

    def registrations(self) -> Tuple[SystemRegistration, ...]:
        """All registrations, ordered by type id."""
        return tuple(self._by_type_id[k] for k in sorted(self._by_type_id))

    def get(self, type_id: str) -> Optional[SystemRegistration]:
        """The registration of ``type_id``, or ``None``."""
        return self._by_type_id.get(type_id)

    def registration_of(self, system: type) -> Optional[SystemRegistration]:
        """The registration of the class ``system`` itself, or ``None``.

        A subclass of a registered class is a class of its own and has no
        registration until it is registered.
        """
        return self._by_system.get(system)

    def is_known(self, system: type) -> bool:
        """True if ``system`` is built in or registered here."""
        return system in self._by_system or is_builtin(system)

    def resolve(
        self,
        type_id: Optional[str] = None,
        *,
        class_name: Optional[str] = None,
        provider: Optional[str] = None,
        version: Optional[str] = None,
        component_id: Optional[str] = None,
    ) -> type:
        """The class of a serialized component.

        With a ``type_id`` the class is the one registered under it.
        Without one (a model serialized before type ids were recorded, or a
        built-in class) the class is resolved by ``class_name``: a built-in
        system first, then the registered class of that name.

        Args:
            type_id: The type id the serialized component records.
            class_name: The class name the serialized component records.
            provider: The provider the serialized component records.
            version: The provider version the serialized component records.
            component_id: Id of the component, for the error message.

        Returns:
            The ``System`` class to instantiate.

        Raises:
            UnknownSystemTypeError: If the type cannot be resolved.
        """
        described = dict(
            component_id=component_id,
            type_id=type_id,
            class_name=class_name,
            provider=provider,
            version=version,
        )
        who = "A component" if component_id is None else f"Component '{component_id}'"
        if type_id is not None:
            registration = self._by_type_id.get(type_id)
            if registration is not None:
                return registration.system
            message = f"{who} has the system type '{type_id}'"
            if class_name is not None:
                message += f" (class {class_name})"
            message += ", which is not registered."
            if provider is not None:
                message += f"  The model was serialized with provider '{provider}'"
                message += "." if version is None else f" version {version}."
            base = type_id.split("@", 1)[0]
            related = [
                r
                for r in self.registrations()
                if r.type_id.split("@", 1)[0] == base
            ]
            if related:
                message += (
                    "  The registry has "
                    + ", ".join(_describe(r) for r in related)
                    + ": another revision of the type, which does not load "
                    "this model."
                )
            message += (
                "  Install "
                + ("the provider" if provider is None else f"'{provider}'")
                + f" and register its class under '{type_id}' on the system "
                "registry of the model before loading."
            )
            raise UnknownSystemTypeError(message, **described)

        if class_name is None:
            raise UnknownSystemTypeError(
                f"{who} records neither a system type id nor a class name.",
                **described,
            )
        system = getattr(systems_module, class_name, None)
        if isinstance(system, type) and issubclass(system, System):
            return system
        named = [r for r in self.registrations() if r.system.__name__ == class_name]
        if len(named) == 1:
            return named[0].system
        if len(named) > 1:
            raise UnknownSystemTypeError(
                f"{who} has the class '{class_name}' and no system type id, and "
                "the class name is ambiguous: the registry has "
                + ", ".join(_describe(r) for r in named)
                + ".",
                **described,
            )
        raise UnknownSystemTypeError(
            f"{who} has the class '{class_name}', which is not a built-in "
            "twin4build system, and the model records no system type id for it.  "
            f"Install the package that provides {class_name} and register the "
            "class on the system registry of the model before loading.",
            **described,
        )


#: The registry models use unless they are given another one.
system_registry = SystemRegistry()


__all__ = [
    "PROVIDER_KEY",
    "PROVIDER_VERSION_KEY",
    "SERIALIZED_KEYS",
    "TYPE_ID_KEY",
    "SystemRegistration",
    "SystemRegistry",
    "UnknownSystemTypeError",
    "is_builtin",
    "system_registry",
]
