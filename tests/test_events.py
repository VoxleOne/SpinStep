# test_events.py — SpinStep Test Suite — MIT License
# Phase 4 tests: EventEmitter

import pytest

from spinstep.control.events import EventEmitter


class TestEventEmitter:
    def test_on_and_emit(self):
        emitter = EventEmitter()
        calls = []
        emitter.on("test", lambda **kw: calls.append(kw))
        emitter.emit("test", x=1)
        assert len(calls) == 1
        assert calls[0] == {"x": 1}

    def test_multiple_listeners(self):
        emitter = EventEmitter()
        results = []
        emitter.on("test", lambda **kw: results.append("a"))
        emitter.on("test", lambda **kw: results.append("b"))
        emitter.emit("test")
        assert results == ["a", "b"]

    def test_off_removes_callback(self):
        emitter = EventEmitter()
        calls = []
        cb = lambda **kw: calls.append(1)
        emitter.on("test", cb)
        emitter.off("test", cb)
        emitter.emit("test")
        assert len(calls) == 0

    def test_off_nonexistent_no_error(self):
        emitter = EventEmitter()
        emitter.off("test", lambda **kw: None)  # no error

    def test_emit_unknown_event_no_error(self):
        emitter = EventEmitter()
        emitter.emit("unknown")  # no error

    def test_clear_all(self):
        emitter = EventEmitter()
        emitter.on("a", lambda **kw: None)
        emitter.on("b", lambda **kw: None)
        emitter.clear()
        assert len(emitter._listeners) == 0

    def test_clear_specific_event(self):
        emitter = EventEmitter()
        calls_a = []
        calls_b = []
        emitter.on("a", lambda **kw: calls_a.append(1))
        emitter.on("b", lambda **kw: calls_b.append(1))
        emitter.clear("a")
        emitter.emit("a")
        emitter.emit("b")
        assert len(calls_a) == 0
        assert len(calls_b) == 1

    def test_deterministic_order(self):
        """Callbacks fire in registration order."""
        emitter = EventEmitter()
        order = []
        emitter.on("test", lambda **kw: order.append(1))
        emitter.on("test", lambda **kw: order.append(2))
        emitter.on("test", lambda **kw: order.append(3))
        emitter.emit("test")
        assert order == [1, 2, 3]

    def test_kwargs_forwarded(self):
        emitter = EventEmitter()
        received = {}
        def cb(**kw):
            received.update(kw)
        emitter.on("test", cb)
        emitter.emit("test", agent="a", time=1.0)
        assert received == {"agent": "a", "time": 1.0}
