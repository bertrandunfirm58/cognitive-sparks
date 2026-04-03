"""Tool registry."""

from sparks.tools.observe import ObserveTool
from sparks.tools.patterns import RecognizePatternsTool, FormPatternsTool
from sparks.tools.abstract import AbstractTool
from sparks.tools.analogize import AnalogizerTool
from sparks.tools.model_tool import ModelTool
from sparks.tools.synthesize import SynthesizeTool

TOOL_REGISTRY: dict[str, type] = {
    "observe": ObserveTool,
    "recognize_patterns": RecognizePatternsTool,
    "form_patterns": FormPatternsTool,
    "abstract": AbstractTool,
    "analogize": AnalogizerTool,
    "model": ModelTool,
    "synthesize": SynthesizeTool,
}

__all__ = ["TOOL_REGISTRY"]
