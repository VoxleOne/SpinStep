# control/events.py — MIT License
# Author: Eraldo B. Marques <eraldo.bernardo@gmail.com> — Created: 2025-05-14
# See LICENSE.txt for full terms. This header must be retained in redistributions.

"""Simple callback-based event system for reactive agent behaviour."""

from __future__ import annotations

__all__ = ["EventEmitter"]

from typing import Any, Callable, Dict, List


class EventEmitter:
    """A lightweight event emitter with ``on`` / ``off`` / ``emit`` semantics.

    No external dependencies — callbacks are plain Python functions stored
    in a dictionary keyed by event name.

    Example::

        from spinstep.control.events import EventEmitter

        emitter = EventEmitter()
        emitter.on("state_change", lambda **kw: print("changed!", kw))
        emitter.emit("state_change", node="robot_a")
    """

    def __init__(self) -> None:
        self._listeners: Dict[str, List[Callable[..., Any]]] = {}

    def on(self, event: str, callback: Callable[..., Any]) -> None:
        """Register a callback for *event*.

        Args:
            event: Event name (e.g. ``"state_change"``).
            callback: Function to call when the event fires.
        """
        self._listeners.setdefault(event, []).append(callback)

    def off(self, event: str, callback: Callable[..., Any]) -> None:
        """Remove a previously registered callback.

        Silently does nothing if the callback was never registered.

        Args:
            event: Event name.
            callback: The callback to remove.
        """
        listeners = self._listeners.get(event, [])
        try:
            listeners.remove(callback)
        except ValueError:
            pass

    def emit(self, event: str, **kwargs: Any) -> None:
        """Fire *event*, calling all registered callbacks in order.

        Args:
            event: Event name.
            **kwargs: Keyword arguments forwarded to every callback.
        """
        for cb in self._listeners.get(event, []):
            cb(**kwargs)

    def clear(self, event: str | None = None) -> None:
        """Remove all callbacks, or all callbacks for a specific event.

        Args:
            event: If given, only clear callbacks for that event.
        """
        if event is None:
            self._listeners.clear()
        else:
            self._listeners.pop(event, None)
