"""Immutable strategy registry — single source of truth for name -> class mapping.

Provides dict-like read access for backward compatibility but prevents
mutation.  For request-scoped dynamic strategies (e.g. ``indicator_combo``),
callers pass an ``overrides`` dict to :meth:`resolve` rather than mutating
the global registry.
"""
from __future__ import annotations

from strategy import BaseStrategy


class StrategyRegistry:
    """Thread-safe, immutable strategy name -> class mapping.

    Parameters
    ----------
    strategies : dict
        ``{name: strategy_class}`` to register at construction time.
        A defensive copy is made; the original dict is never mutated.
    """

    __slots__ = ("_strategies",)

    def __init__(self, strategies: dict[str, type[BaseStrategy]]) -> None:
        object.__setattr__(self, "_strategies", dict(strategies))

    def __setattr__(self, name, value):
        raise AttributeError(
            f"StrategyRegistry is immutable — cannot set '{name}'"
        )

    def __delattr__(self, name):
        raise AttributeError(
            f"StrategyRegistry is immutable — cannot delete '{name}'"
        )

    # ── resolution ───────────────────────────────────────────────────

    def resolve(
        self,
        name: str,
        *,
        overrides: dict[str, type[BaseStrategy]] | None = None,
    ) -> type[BaseStrategy]:
        """Resolve *name* to a strategy class.

        Parameters
        ----------
        name : str
            Strategy name.
        overrides : dict, optional
            Request-scoped ``{name: cls}`` that take precedence over the
            base registry.  Use this for dynamically-bound strategies
            such as ``indicator_combo`` so the global registry stays
            immutable.

        Raises
        ------
        ValueError
            If *name* is not found in the registry or overrides.
        """
        if overrides and name in overrides:
            return overrides[name]
        if name in self._strategies:
            return self._strategies[name]
        available = list(self._strategies)
        if overrides:
            available.extend(k for k in overrides if k not in self._strategies)
        raise ValueError(
            f"Unknown strategy '{name}'. Available: {', '.join(available)}"
        )

    def list_strategies(self) -> dict:
        """Return ``{"strategies": {name: docstring_first_line, ...}}``."""
        entries = {}
        for name, cls in self._strategies.items():
            doc = (cls.__doc__ or "").strip().split("\n")[0]
            entries[name] = doc
        return {"strategies": entries}

    # ── dict-like read interface (backward compat) ───────────────────

    def keys(self):
        return self._strategies.keys()

    def items(self):
        return self._strategies.items()

    def values(self):
        return self._strategies.values()

    def __contains__(self, name: str) -> bool:
        return name in self._strategies

    def __getitem__(self, name: str) -> type[BaseStrategy]:
        return self._strategies[name]

    def __iter__(self):
        return iter(self._strategies)

    def __len__(self) -> int:
        return len(self._strategies)

    def __repr__(self) -> str:
        names = ", ".join(self._strategies)
        return f"StrategyRegistry([{names}])"
