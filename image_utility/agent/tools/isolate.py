"""
Isolate Tool - Wraps the existing IsolatePhase for agent use.

Segments product from background, producing RGBA with alpha mask.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from image_utility.pipeline.context import PipelineContext
from image_utility.isolate.config import IsolateConfig, load_isolate_config
from image_utility.isolate.processor import process_isolate

from ..contracts import ToolCategory, ToolDefinition, ToolInput, ToolResult, CostClass
from ..registry import register_tool, TOOL_DEFINITIONS
from .base import BaseTool, merge_config

LOGGER = logging.getLogger(__name__)


class IsolateTool(BaseTool):
    """
    Tool wrapper for IsolatePhase.
    
    Converts between agent's ToolInput/ToolResult and pipeline's PipelineContext.
    Config overrides merge with env defaults.
    """
    
    @property
    def definition(self) -> ToolDefinition:
        return TOOL_DEFINITIONS["isolate"]
    
    def _load_default_config(self) -> IsolateConfig:
        return load_isolate_config()
    
    def _execute(self, input: ToolInput, config: IsolateConfig | None) -> ToolResult:
        """Execute isolate phase and return tool result."""
        
        # Load image
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
            input_path=image_path,
            output_path=input.workdir,
        )
        ctx.current_image = rgb
        
        # Run isolate
        try:
            ctx = process_isolate(ctx, cfg=config)
        except OSError as e:
            return ToolResult.failure(f"Isolate failed: {e}")
        except Exception as e:
            LOGGER.exception("Isolate error")
            return ToolResult.failure(f"Isolate error: {e}")
        
        # Check for valid output
        if ctx.current_rgba is None or ctx.alpha_mask is None:
            return ToolResult.failure("Isolate produced no output")
        
        # Save outputs
        stem = image_path.stem
        version = input.state.next_version()
        
        rgba_path = input.workdir / f"{stem}_v{version}_rgba.png"
        mask_path = input.workdir / f"{stem}_v{version}_mask.png"
        
        Image.fromarray(ctx.current_rgba, "RGBA").save(rgba_path)
        Image.fromarray(ctx.alpha_mask).save(mask_path)
        
        # Compute coverage metric
        h, w = ctx.alpha_mask.shape
        coverage = float(np.count_nonzero(ctx.alpha_mask > 8)) / (h * w)
        
        return ToolResult(
            success=True,
            output_image_path=rgba_path,
            state_updates={
                "background_removed": True,
                "rgba_path": rgba_path,
                "alpha_mask_path": mask_path,
                "working_image_path": rgba_path,
            },
            metadata={
                "coverage": round(coverage, 4),
                "rgba_shape": list(ctx.current_rgba.shape),
                "component_count": ctx.debug.get("decomposition", {}).get("semantic_candidate_count", 0),
            },
            confidence=min(0.95, 0.5 + coverage),  # Higher coverage = higher confidence
        )


# Register the tool
register_tool(IsolateTool())
