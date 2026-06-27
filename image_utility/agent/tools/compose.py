"""
Compose Tool - Wraps the existing ComposePhase for agent use.

Places isolated product on 2000x2000 white canvas with balanced positioning.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from image_utility.pipeline.context import PipelineContext
from image_utility.compose.config import ComposeConfig, load_compose_config
from image_utility.compose.processor import process_compose

from ..contracts import ToolCategory, ToolDefinition, ToolInput, ToolResult, CostClass
from ..registry import register_tool, TOOL_DEFINITIONS
from .base import BaseTool, merge_config

LOGGER = logging.getLogger(__name__)


class ComposeTool(BaseTool):
    """
    Tool wrapper for ComposePhase.
    
    Requires RGBA input (from isolate tool).
    Config overrides merge with env defaults.
    """
    
    @property
    def definition(self) -> ToolDefinition:
        return TOOL_DEFINITIONS["compose"]
    
    def _load_default_config(self) -> ComposeConfig:
        return load_compose_config()
    
    def _execute(self, input: ToolInput, config: ComposeConfig | None) -> ToolResult:
        """Execute compose phase and return tool result."""
        
        # Determine input image
        # Prefer rgba_path from state, fallback to working_image_path
        if input.state.rgba_path and input.state.rgba_path.exists():
            image_path = input.state.rgba_path
        else:
            image_path = input.image_path
        
        if not image_path.exists():
            return ToolResult.failure(f"Image not found: {image_path}")
        
        # Load RGBA image
        try:
            with Image.open(image_path) as img:
                if img.mode != "RGBA":
                    return ToolResult.failure(f"Compose requires RGBA image, got {img.mode}")
                rgba = np.array(img)
        except Exception as e:
            return ToolResult.failure(f"Failed to load image: {e}")
        
        # Create pipeline context
        ctx = PipelineContext(
            input_path=input.state.source_path,
            output_path=input.workdir,
        )
        ctx.current_rgba = rgba
        
        # Run compose
        try:
            ctx = process_compose(ctx, cfg=config)
        except OSError as e:
            return ToolResult.failure(f"Compose failed: {e}")
        except Exception as e:
            LOGGER.exception("Compose error")
            return ToolResult.failure(f"Compose error: {e}")
        
        # Check for valid output
        if ctx.current_image is None:
            return ToolResult.failure("Compose produced no output")
        
        # Save output
        stem = input.state.source_path.stem
        version = input.state.next_version()
        
        output_path = input.workdir / f"{stem}_v{version}_composed.jpg"
        quality = config.jpeg_quality if config else 94
        Image.fromarray(ctx.current_image, "RGB").save(output_path, "JPEG", quality=quality)
        
        # Extract metadata from context
        scale = ctx.debug.get("compose_scale", 1.0)
        origin = ctx.debug.get("compose_origin", (0, 0))
        canvas = ctx.debug.get("compose_canvas", (2000, 2000))
        
        return ToolResult(
            success=True,
            output_image_path=output_path,
            state_updates={
                "composed_on_white": True,
                "composed_path": output_path,
                "working_image_path": output_path,
            },
            metadata={
                "scale_factor": round(scale, 4),
                "position": list(origin),
                "canvas_size": list(canvas),
            },
            confidence=0.95,
        )


# Register the tool
register_tool(ComposeTool())
