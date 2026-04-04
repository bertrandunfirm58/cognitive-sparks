"""Tool registry — 13 built-in + user plugins from ~/.sparks/tools/."""

import importlib.util
import sys
from pathlib import Path

from sparks.tools.observe import ObserveTool
from sparks.tools.imagine import ImagineTool
from sparks.tools.patterns import RecognizePatternsTool, FormPatternsTool
from sparks.tools.abstract import AbstractTool
from sparks.tools.analogize import AnalogizerTool
from sparks.tools.body_think import BodyThinkTool
from sparks.tools.empathize import EmpathizeTool
from sparks.tools.shift_dimension import ShiftDimensionTool
from sparks.tools.model_tool import ModelTool
from sparks.tools.play import PlayTool
from sparks.tools.transform import TransformTool
from sparks.tools.synthesize import SynthesizeTool

TOOL_REGISTRY: dict[str, type] = {
    "observe": ObserveTool,
    "imagine": ImagineTool,
    "recognize_patterns": RecognizePatternsTool,
    "form_patterns": FormPatternsTool,
    "abstract": AbstractTool,
    "analogize": AnalogizerTool,
    "body_think": BodyThinkTool,
    "empathize": EmpathizeTool,
    "shift_dimension": ShiftDimensionTool,
    "model": ModelTool,
    "play": PlayTool,
    "transform": TransformTool,
    "synthesize": SynthesizeTool,
}


def _load_plugins():
    """Auto-discover custom tools from ~/.sparks/tools/*.py

    Each plugin file must define a class inheriting from BaseTool.
    The tool's `name` attribute is used as the registry key.

    Example plugin (~/.sparks/tools/my_tool.py):
        from sparks.tools.base import BaseTool
        from sparks.state import CognitiveState

        class MyTool(BaseTool):
            name = "my_custom_tool"
            def should_run(self, state): return len(state.principles) >= 1
            def run(self, state, **kwargs): ...
    """
    plugin_dir = Path.home() / ".sparks" / "tools"
    if not plugin_dir.exists():
        return

    from sparks.tools.base import BaseTool

    for py_file in sorted(plugin_dir.glob("*.py")):
        if py_file.name.startswith("_"):
            continue
        try:
            spec = importlib.util.spec_from_file_location(
                f"sparks_plugin_{py_file.stem}", py_file
            )
            if not spec or not spec.loader:
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)

            # Find all BaseTool subclasses in the module
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and issubclass(attr, BaseTool)
                        and attr is not BaseTool and hasattr(attr, 'name')):
                    TOOL_REGISTRY[attr.name] = attr
        except Exception:
            continue  # Skip broken plugins silently


_load_plugins()

__all__ = ["TOOL_REGISTRY"]
