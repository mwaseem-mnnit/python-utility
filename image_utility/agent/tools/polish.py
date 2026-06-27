"""
Polish Tool - Wraps the existing PolishPhase for agent use.

Applies subtle contrast, sharpness, and clarity enhancement.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from image_utility.pipeline.context import PipelineContext
from image_utility.polish.config import PolishConfig, load_polish_config
from image_utility.polish.processor import process_polish

from ..contracts import ToolCategory, ToolDefinition, ToolInput, ToolResult, CostClass
from ..registry import register_tool, TOOL_DEFINITIONS
from .base import BaseTool, merge_config

LOGGER = logging.getLogger(__name__)


class PolishTool(BaseTool):
    """
    Tool wrapper for PolishPhase.
    
    Requires RGB input (from compose or shadow).
    Config overrides merge with env defaults.
    """
    
    @property
    def definition(self) -> ToolDefinition:
        return TOOL_DEFINITIONS["polish"]
    
    def _load_default_config(self) -> PolishConfig:
        return load_polish_config()
    
    def _execute(self, input: ToolInput, config: PolishConfig | None) -> ToolResult:
        """Execute polish phase and return tool result."""
        
        # Load RGB image
        image_path = input.image_path
        if not image_path.exists():
            return ToolResult.failure(f"Image not found: {image_path}")
        
        try:
            with Image.open(image_path) as img:
                rgb = np.array(img.convert("RGB"))
        except Exception as e:
            return ToolResult.failure(f"Failed to load image: {e}")
        
        # Create pipeline context
        ctx = PipelineContext(
            input_path=input.state.source_path,
            output_path=input.workdir,
        )
        ctx.current_image = rgb
        
        # Run polish
        try:
            ctx = process_polish(ctx, cfg=config)
        except OSError as e:
            return ToolResult.failure(f"Polish failed: {e}")
        except Exception as e:
            LOGGER.exception("Polish error")
            return ToolResult.failure(f"Polish error: {e}")
        
        # Check for valid output
        if ctx.current_image is None:
            return ToolResult.failure("Polish produced no output")
        
        # Save output
        stem = input.state.source_path.stem
        version = input.state.next_version()
        
        output_path = input.workdir / f"{stem}_v{version}_polished.jpg"
        quality = 94
        Image.fromarray(ctx.current_image, "RGB").save(output_path, "JPEG", quality=quality)
        
        return ToolResult(
            success=True,
            output_image_path=output_path,
            state_updates={
                "polished": True,
                "working_image_path": output_path,
            },
            metadata={
                "sharpen_strength": config.sharpen_strength if config else 0.14,
                "contrast_factor": config.contrast_factor if config else 1.05,
                "clarity_strength": config.clarity_strength if config else 0.28,
            },
            confidence=0.95,
        )


# Register the tool
register_tool(PolishTool())
