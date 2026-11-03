"""The resolver contract for bare names. The caller's catalogue answers before the rules' families table, and
says which version of itself answered."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@runtime_checkable
class Resolver(Protocol):
    """A catalogue that names the vendors a bare model name may belong to.

    ``candidates`` returns canonical ``vendor/model`` strings (or bare vendor names) for an exact model
    match; zero means unknown to the catalogue and leaves the name to the families table, one resolves, more
    than one is ambiguous and the parse fails closed.
    ``fingerprint`` identifies the catalogue version and is recorded in the value's provenance.
    """

    fingerprint: str | None

    def candidates(self, bare_name: str) -> Sequence[str]: ...


@dataclass(frozen=True)
class CallableResolver:
    """Adapts a plain function to the :class:`Resolver` protocol."""

    function: Callable[[str], Sequence[str]]
    fingerprint: str | None = None

    def candidates(self, bare_name: str) -> Sequence[str]:
        result = self.function(bare_name)
        if isinstance(result, str):
            raise TypeError("a resolver returns a sequence of candidates, not one string")
        return tuple(result)


def vendor_of_candidate(candidate: str) -> str:
    """The vendor named by a resolver candidate, which may be ``vendor/model`` or a bare vendor."""
    return candidate.split("/", 1)[0]
