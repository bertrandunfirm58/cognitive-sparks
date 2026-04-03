"""Base tool interface — all 13 tools inherit from this.

Each tool has a should_run() local rule (like a ganglion making autonomous decisions).
No central TOOL_ORDER needed. Tools sense state and decide for themselves.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from sparks.cost import CostTracker
from sparks.events import EventBus, StateEvent
from sparks.state import CognitiveState


class BaseTool(ABC):
    name: str = "base"

    def __init__(self, event_bus: EventBus, tracker: CostTracker):
        self.bus = event_bus
        self.tracker = tracker

    @abstractmethod
    def run(self, state: CognitiveState, **kwargs) -> None:
        """Execute the tool, mutating state in place."""
        ...

    def should_run(self, state: CognitiveState) -> bool:
        """Local rule: should this tool run now?

        Like a ganglion deciding based on local conditions,
        not a brain commanding from above.
        Default: always run. Override in subclasses.
        """
        return True

    def emit(self, event_type: str, data_id: str = "", round: int = 0):
        self.bus.publish(StateEvent(
            type=event_type,
            source_tool=self.name,
            data_id=data_id,
            round=round,
        ))
