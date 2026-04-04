"""Tool registry — all 13 thinking tools."""

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

__all__ = ["TOOL_REGISTRY"]
