"""Event Bus — tools communicate through state changes, not direct calls."""

from __future__ import annotations

from collections import defaultdict
from typing import Callable

from pydantic import BaseModel


class StateEvent(BaseModel):
    type: str
    source_tool: str
    data_id: str = ""
    round: int = 0


class EventBus:
    """
    The shared rhythm. Tools publish events when they change state.
    Other tools subscribe and react. No conductor — just resonance.
    """

    def __init__(self):
        self._subscribers: dict[str, list[Callable]] = defaultdict(list)
        self._log: list[StateEvent] = []

    def subscribe(self, event_type: str, handler: Callable):
        self._subscribers[event_type].append(handler)

    def publish(self, event: StateEvent):
        self._log.append(event)
        for handler in self._subscribers.get(event.type, []):
            handler(event)

    @property
    def log(self) -> list[StateEvent]:
        return self._log
