"""
Shadow Tool - Wraps the existing ShadowPhase for agent use.

Adds subtle grounding shadow beneath product on composed white canvas.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from image_utility.pipeline.context import PipelineContext
from image_utility.shadow.config import ShadowConfig, load_shadow_config
from image_utility.shadow.processor import process_shadow

from ..contracts import ToolCategory, ToolDefinition, ToolInput, ToolResult, CostClass
from ..registry import register_tool, TOOL_DEFINITIONS
from .base import BaseTool, merge_config

LOGGER = logging.getLogger(__name__)


class ShadowTool(BaseTool):
    """
    Tool wrapper for ShadowPhase.
    
    Requires composed RGB + composed_rgba_canvas (from compose tool).
    Config overrides merge with env defaults.
    """
    
    @property
    def definition(self) -> ToolDefinition:
        return TOOL_DEFINITIONS["shadow"]
    
    def _load_default_config(self) -> ShadowConfig:
        return load_shadow_config()
    
    def _execute(self, input: ToolInput, config: ShadowConfig | None) -> ToolResult:
        """Execute shadow phase and return tool result."""
        
        # Load composed image (RGB)
        image_path = input.image_path
        if not image_path.exists():
            return ToolResult.failure(f"Image not found: {image_path}")
        
        try:
            with Image.open(image_path) as img:
                rgb = np.array(img.convert("RGB"))
        except Exception as e:
            return ToolResult.failure(f"Failed to load image: {e}")
        
        # Load RGBA for alpha channel (needed for shadow placement)
        # Try composed_path first, then rgba_path
        rgba_path = input.state.composed_path or input.state.rgba_path
        if rgba_path is None or not Path(rgba_path).exists():
            return ToolResult.failure("Shadow requires rgba_path or composed RGBA")
        
        try:
            with Image.open(rgba_path) as img:
                if img.mode == "RGBA":
                    rgba = np.array(img)
                else:
                    # Create RGBA from RGB + separate alpha
                    alpha_path = input.state.alpha_mask_path
                    if alpha_path and Path(alpha_path).exists():
                        with Image.open(alpha_path) as alpha_img:
                            alpha = np.array(alpha_img.convert("L"))
                        rgba = np.dstack([np.array(img.convert("RGB")), alpha])
                    else:
                        return ToolResult.failure("Cannot construct RGBA for shadow")
        except Exception as e:
            return ToolResult.failure(f"Failed to load RGBA: {e}")
        
        # Ensure RGBA matches RGB dimensions (may need to create canvas-aligned RGBA)
        h, w = rgb.shape[:2]
        if rgba.shape[0] != h or rgba.shape[1] != w:
            # Create canvas-aligned RGBA
            canvas_rgba = np.zeros((h, w, 4), dtype=np.uint8)
            # Simple center placement (shadow phase normally gets this from compose)
            rh, rw = rgba.shape[:2]
            ox = (w - rw) // 2
            oy = (h - rh) // 2
            # Clip to canvas bounds
            ox = max(0, ox)
            oy = max(0, oy)
            end_x = min(w, ox + rw)
            end_y = min(h, oy + rh)
            src_w = end_x - ox
            src_h = end_y - oy
            canvas_rgba[oy:end_y, ox:end_x] = rgba[:src_h, :src_w]
            rgba = canvas_rgba
        
        # Create pipeline context
        ctx = PipelineContext(
            input_path=input.state.source_path,
            output_path=input.workdir,
        )
        ctx.current_image = rgb
        ctx.composed_rgba_canvas = rgba
        
        # Run shadow
        try:
            ctx = process_shadow(ctx, cfg=config)
        except OSError as e:
            return ToolResult.failure(f"Shadow failed: {e}")
        except Exception as e:
            LOGGER.exception("Shadow error")
            return ToolResult.failure(f"Shadow error: {e}")
        
        # Check for valid output
        if ctx.current_image is None:
            return ToolResult.failure("Shadow produced no output")
        
        # Save output
        stem = input.state.source_path.stem
        version = input.state.next_version()
        
        output_path = input.workdir / f"{stem}_v{version}_shadow.jpg"
        quality = 94
        Image.fromarray(ctx.current_image, "RGB").save(output_path, "JPEG", quality=quality)
        
        return ToolResult(
            success=True,
            output_image_path=output_path,
            state_updates={
                "shadow_added": True,
                "working_image_path": output_path,
            },
            metadata={
                "blur_sigma": config.blur_sigma if config else 14.0,
                "shadow_opacity": config.shadow_opacity if config else 0.22,
            },
            confidence=0.95,
        )


# Register the tool
register_tool(ShadowTool())
