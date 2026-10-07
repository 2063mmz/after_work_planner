"""Per-request recording of the agent's tool calls.

Tools push an entry here every time they run. The API turns those entries into
the `agent_trace` the dashboard renders, and the streaming endpoint forwards
them live while the agent is still thinking.
"""

from __future__ import annotations

import queue
import time
from contextvars import ContextVar
from typing import Any

# The user-facing name of each step, in the order the dashboard shows them.
STEP_LABELS: dict[str, str] = {
    "parse_user_context": "Understanding request",
    "load_saved_places": "Loading saved places",
    "check_tomorrow_pressure": "Checking tomorrow pressure",
    "estimate_task_priority": "Ranking tasks",
    "get_weather_context": "Checking weather",
    "estimate_travel_time": "Estimating travel",
    "create_evening_plan": "Calling planner tool",
    "compare_alternative_plans": "Comparing alternatives",
    "recommendation": "Generating recommendation",
}

STEP_ORDER = list(STEP_LABELS)


class TraceRecorder:
    """Collects tool calls for one planning request.

    The recorder is deliberately thread-safe and dependency-free: the agent
    runs in a worker thread and the SSE endpoint drains `events` from the
    event loop thread.
    """

    def __init__(self) -> None:
        self.entries: list[dict[str, Any]] = []
        self.events: queue.Queue[dict[str, Any]] = queue.Queue()
        self.data: dict[str, Any] = {}
        self._started = time.perf_counter()

    def record(
        self,
        tool: str,
        status: str,
        detail: str,
        payload: dict[str, Any] | None = None,
        arguments: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        entry = {
            "step": STEP_LABELS.get(tool, tool.replace("_", " ").capitalize()),
            "tool": tool,
            "status": status,
            "detail": detail,
            "arguments": _compact(arguments or {}),
            "result": payload or {},
            "elapsed_ms": int((time.perf_counter() - self._started) * 1000),
        }
        self.entries.append(entry)
        self.events.put({"type": "trace", "entry": entry})
        return entry

    def remember(self, key: str, value: Any) -> None:
        """Keep a tool result so the API can reuse it as the source of truth."""
        self.data[key] = value

    def emit(self, event: dict[str, Any]) -> None:
        self.events.put(event)

    def steps_used(self) -> list[str]:
        return [entry["tool"] for entry in self.entries]


_CURRENT: ContextVar[TraceRecorder | None] = ContextVar("lifeops_trace", default=None)


def current() -> TraceRecorder:
    """Return the recorder for the running request, or a throwaway one."""
    recorder = _CURRENT.get()
    if recorder is None:
        recorder = TraceRecorder()
        _CURRENT.set(recorder)
    return recorder


def start(recorder: TraceRecorder) -> None:
    _CURRENT.set(recorder)


def _compact(value: Any, limit: int = 400) -> Any:
    """Shorten long tool arguments so trace cards stay readable."""
    if isinstance(value, dict):
        return {key: _compact(item, limit) for key, item in value.items()}
    if isinstance(value, list):
        return [_compact(item, limit) for item in value[:12]]
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + "…"
    return value
