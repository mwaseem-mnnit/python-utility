"""
Image Agent - Base tool class with common functionality.

This module provides:
- BaseTool: Abstract base with logging, timing, debug output
- Config merging: Tool config from input overrides env defaults
- Standard execution wrapper with error handling

Every tool should inherit from BaseTool instead of Tool directly.
"""

from __future__ import annotations

import json
import logging
import time
from abc import abstractmethod
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
from PIL import Image

from ..contracts import Tool, ToolDefinition, ToolInput, ToolResult

LOGGER = logging.getLogger(__name__)

T = TypeVar("T")


def merge_config(
    env_config: T,
    overrides: dict[str, Any],
) -> T:
    """
    Merge user-provided config overrides into env-loaded config.
    
    This allows the agent to customize tool behavior per-image while
    still using env defaults for unspecified fields.
    
    Args:
        env_config: Config loaded from environment (dataclass)
        overrides: Dict of field_name -> value to override
        
    Returns:
        New config dataclass with overrides applied
        
    Example:
        env_cfg = load_compose_config()  # From env
        merged = merge_config(env_cfg, {"canvas_width": 1500})
    """
    if not overrides or not is_dataclass(env_config):
        return env_config
    
    # Get current values
    current = asdict(env_config)
    
    # Apply overrides for fields that exist
    valid_fields = {f.name for f in fields(env_config)}
    for key, value in overrides.items():
        if key in valid_fields:
            current[key] = value
        else:
            LOGGER.warning("Unknown config field '%s' ignored", key)
    
    # Create new config instance
    return type(env_config)(**current)


class BaseTool(Tool):
    """
    Base class for all tool implementations.
    
    Provides:
    - Automatic timing and logging
    - Debug artifact writing
    - Error handling wrapper
    - Config merging with env defaults
    
    Subclasses implement:
    - definition property (tool metadata)
    - _execute() method (actual tool logic)
    - _load_default_config() (load from env)
    """
    
    @property
    @abstractmethod
    def definition(self) -> ToolDefinition:
        """Return the tool's metadata definition."""
        pass
    
    @abstractmethod
    def _execute(self, input: ToolInput, config: Any) -> ToolResult:
        """
        Execute the tool logic.
        
        Args:
            input: ToolInput with image path, state, workdir
            config: Merged config (env defaults + input overrides)
            
        Returns:
            ToolResult with success status, output, state updates
        """
        pass
    
    def _load_default_config(self) -> Any:
        """
        Load default config from environment.
        
        Override in subclass to return phase-specific config.
        Returns None if tool has no config.
        """
        return None
    
    def execute(self, input: ToolInput) -> ToolResult:
        """
        Execute tool with logging, timing, and error handling.
        
        This is the public entry point called by the executor.
        """
        start_time = time.perf_counter()
        tool_name = self.name
        image_name = input.image_path.name if input.image_path else "unknown"
        
        LOGGER.info("[TOOL] %s START image=%s", tool_name, image_name)
        
        try:
            # Check preconditions
            passed, reason = self.check_preconditions(input.state)
            if not passed:
                duration_ms = int((time.perf_counter() - start_time) * 1000)
                LOGGER.warning(
                    "[TOOL] %s SKIP precondition failed: %s",
                    tool_name, reason
                )
                return ToolResult.failure(
                    error=f"Precondition failed: {reason}",
                    duration_ms=duration_ms
                )
            
            # Merge config: env defaults + input overrides
            default_config = self._load_default_config()
            if default_config is not None and input.config:
                config = merge_config(default_config, input.config)
            else:
                config = default_config
            
            # Execute tool logic
            result = self._execute(input, config)
            
            # Calculate duration
            duration_ms = int((time.perf_counter() - start_time) * 1000)
            result.duration_ms = duration_ms
            
            LOGGER.info(
                "[TOOL] %s END success=%s duration=%dms confidence=%.2f",
                tool_name, result.success, duration_ms, result.confidence
            )
            
            return result
            
        except Exception as e:
            duration_ms = int((time.perf_counter() - start_time) * 1000)
            LOGGER.exception("[TOOL] %s ERROR: %s", tool_name, e)
            return ToolResult.failure(
                error=str(e),
                duration_ms=duration_ms
            )
    
    def _get_debug_dir(self, input: ToolInput) -> Path:
        """Get debug output directory for this tool."""
        debug_dir = input.workdir / "debug" / self.name
        debug_dir.mkdir(parents=True, exist_ok=True)
        return debug_dir
    
    def _write_debug_image(
        self,
        input: ToolInput,
        image: np.ndarray | Image.Image,
        suffix: str,
        ext: str = "png"
    ) -> Path:
        """
        Write a debug image artifact.
        
        Args:
            input: ToolInput for path context
            image: Image data (numpy array or PIL Image)
            suffix: Filename suffix (e.g., "mask", "overlay")
            ext: File extension
            
        Returns:
            Path to written file
        """
        debug_dir = self._get_debug_dir(input)
        stem = input.image_path.stem
        out_path = debug_dir / f"{stem}_{suffix}.{ext}"
        
        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                # Grayscale
                Image.fromarray(image).save(out_path)
            elif image.shape[2] == 4:
                # RGBA
                Image.fromarray(image, "RGBA").save(out_path)
            else:
                # RGB
                Image.fromarray(image, "RGB").save(out_path)
        else:
            image.save(out_path)
        
        LOGGER.debug("[TOOL] %s debug: %s", self.name, out_path.name)
        return out_path
    
    def _write_debug_json(
        self,
        input: ToolInput,
        data: dict,
        suffix: str = "metadata"
    ) -> Path:
        """Write a debug JSON artifact."""
        debug_dir = self._get_debug_dir(input)
        stem = input.image_path.stem
        out_path = debug_dir / f"{stem}_{suffix}.json"
        
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        
        return out_path
    
    def _load_image_rgb(self, path: Path) -> np.ndarray:
        """Load image as RGB numpy array."""
        with Image.open(path) as img:
            return np.array(img.convert("RGB"))
    
    def _load_image_rgba(self, path: Path) -> np.ndarray:
        """Load image as RGBA numpy array."""
        with Image.open(path) as img:
            return np.array(img.convert("RGBA"))
    
    def _save_image_rgb(
        self,
        image: np.ndarray,
        path: Path,
        quality: int = 94
    ) -> None:
        """Save RGB image."""
        ext = path.suffix.lower()
        img = Image.fromarray(image, "RGB")
        if ext in (".jpg", ".jpeg"):
            img.save(path, "JPEG", quality=quality)
        elif ext == ".webp":
            img.save(path, "WEBP", quality=quality)
        else:
            img.save(path)
    
    def _save_image_rgba(self, image: np.ndarray, path: Path) -> None:
        """Save RGBA image as PNG."""
        Image.fromarray(image, "RGBA").save(path)
