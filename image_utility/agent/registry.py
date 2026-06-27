"""
Image Agent - Tool registry for tool discovery and lookup.

This module provides:
- Tool registration mechanism
- Tool lookup by name
- Tool definition listing for planner
- Precondition checking

The registry is the single source of truth for available tools.
The planner queries the registry to understand tool capabilities.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .contracts import Tool, ToolCategory, ToolDefinition, CostClass

if TYPE_CHECKING:
    from .state import ImageState

LOGGER = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Global Registry
# ─────────────────────────────────────────────────────────────────────────────

_TOOL_REGISTRY: dict[str, Tool] = {}
_DEFINITIONS_CACHE: dict[str, ToolDefinition] = {}


def register_tool(tool: Tool) -> None:
    """
    Register a tool in the global registry.
    
    Args:
        tool: Tool instance to register
        
    Raises:
        ValueError: If tool with same name already exists
    """
    name = tool.name.lower()
    if name in _TOOL_REGISTRY:
        LOGGER.warning("Tool '%s' already registered, overwriting", name)
    
    _TOOL_REGISTRY[name] = tool
    _DEFINITIONS_CACHE[name] = tool.definition
    LOGGER.debug("Registered tool: %s", name)


def get_tool(name: str) -> Tool | None:
    """
    Get a tool by name.
    
    Args:
        name: Tool name (case-insensitive)
        
    Returns:
        Tool instance or None if not found
    """
    _ensure_default_tools()
    return _TOOL_REGISTRY.get(name.lower())


def get_tool_definition(name: str) -> ToolDefinition | None:
    """
    Get a tool's definition without the implementation.
    
    Useful for planner to inspect tool metadata.
    
    Args:
        name: Tool name (case-insensitive)
        
    Returns:
        ToolDefinition or None if not found
    """
    _ensure_default_tools()
    return _DEFINITIONS_CACHE.get(name.lower())


def list_tools() -> list[str]:
    """
    List all registered tool names.
    
    Returns:
        Sorted list of tool names
    """
    _ensure_default_tools()
    return sorted(_TOOL_REGISTRY.keys())


def list_tool_definitions() -> list[ToolDefinition]:
    """
    List all tool definitions.
    
    Returns:
        List of ToolDefinition objects
    """
    _ensure_default_tools()
    return list(_DEFINITIONS_CACHE.values())


def list_tools_by_category(category: ToolCategory) -> list[str]:
    """
    List tools filtered by category.
    
    Args:
        category: ToolCategory to filter by
        
    Returns:
        List of tool names in that category
    """
    _ensure_default_tools()
    return [
        name for name, tool in _TOOL_REGISTRY.items()
        if tool.category == category
    ]


def check_tool_preconditions(name: str, state: "ImageState") -> tuple[bool, str | None]:
    """
    Check if a tool's preconditions are met.
    
    Args:
        name: Tool name
        state: Current ImageState
        
    Returns:
        (True, None) if preconditions pass
        (False, reason) if preconditions fail or tool not found
    """
    tool = get_tool(name)
    if tool is None:
        return False, f"Tool '{name}' not found"
    
    return tool.check_preconditions(state)


def get_tools_for_issue(issue_type: str) -> list[str]:
    """
    Get tools that can resolve a specific issue type.
    
    Args:
        issue_type: Issue type string (e.g., "hand_visible")
        
    Returns:
        List of tool names that can address this issue
    """
    from .contracts import ISSUE_TO_TOOL
    return ISSUE_TO_TOOL.get(issue_type, [])


def format_tools_for_planner() -> str:
    """
    Format tool registry as text for VLM planner prompt.
    
    Returns:
        Formatted string describing all tools
    """
    _ensure_default_tools()
    
    lines = ["Available Tools:", ""]
    
    for category in ToolCategory:
        tools_in_cat = [
            defn for defn in _DEFINITIONS_CACHE.values()
            if defn.category == category
        ]
        if not tools_in_cat:
            continue
            
        lines.append(f"## {category.value.title()} Tools")
        lines.append("")
        
        for defn in sorted(tools_in_cat, key=lambda d: d.name):
            lines.append(f"### {defn.name}")
            lines.append(f"Description: {defn.description}")
            if defn.preconditions:
                lines.append(f"Preconditions: {', '.join(defn.preconditions)}")
            lines.append(f"Cost: {defn.cost_class.value}")
            lines.append("")
    
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Tool Definitions (declared here, implementations in tools/)
# ─────────────────────────────────────────────────────────────────────────────

# These definitions describe what tools WILL exist once implemented.
# They serve as the contract for the planner.

TOOL_DEFINITIONS: dict[str, ToolDefinition] = {
    # ─────────────────────────────────────────────────────────────────────
    # Analysis Tools
    # ─────────────────────────────────────────────────────────────────────
    "classify_scene": ToolDefinition(
        name="classify_scene",
        description="Classify input scene as hand-held, table-surface, or already-white background",
        category=ToolCategory.ANALYSIS,
        state_reads=["source_path"],
        state_writes=["scene_type", "scene_confidence"],
        preconditions=[],
        cost_class=CostClass.FAST,
        idempotent=True,
        failure_modes=["corrupt_image"],
    ),
    
    "detect_hand": ToolDefinition(
        name="detect_hand",
        description="Detect hand presence and generate mask for inpainting",
        category=ToolCategory.ANALYSIS,
        state_reads=["working_image_path"],
        state_writes=["hand_detected", "hand_confidence", "hand_mask_path"],
        preconditions=[],
        cost_class=CostClass.MEDIUM,
        idempotent=True,
        failure_modes=["no_skin_detected"],
    ),
    
    "detect_overlay": ToolDefinition(
        name="detect_overlay",
        description="Detect text overlays and watermarks on image",
        category=ToolCategory.ANALYSIS,
        state_reads=["working_image_path"],
        state_writes=["overlay_detected", "overlay_mask_path"],
        preconditions=[],
        cost_class=CostClass.MEDIUM,
        idempotent=True,
        failure_modes=["no_text_detected"],
    ),
    
    "measure_quality": ToolDefinition(
        name="measure_quality",
        description="Measure brightness, contrast, sharpness and overall quality",
        category=ToolCategory.ANALYSIS,
        state_reads=["working_image_path"],
        state_writes=["brightness_score", "contrast_score", "sharpness_score", "overall_quality"],
        preconditions=[],
        cost_class=CostClass.FAST,
        idempotent=True,
        failure_modes=["corrupt_image"],
    ),
    
    "detect_packaging": ToolDefinition(
        name="detect_packaging",
        description="Detect product packaging/box to preserve in final image",
        category=ToolCategory.ANALYSIS,
        state_reads=["working_image_path"],
        state_writes=["packaging_detected", "packaging_bbox"],
        preconditions=[],
        cost_class=CostClass.MEDIUM,
        idempotent=True,
        failure_modes=["no_packaging_detected"],
    ),
    
    # ─────────────────────────────────────────────────────────────────────
    # Processing Tools
    # ─────────────────────────────────────────────────────────────────────
    "inpaint_hand": ToolDefinition(
        name="inpaint_hand",
        description="Remove detected hand via IOPaint/LaMa inpainting",
        category=ToolCategory.PROCESSING,
        state_reads=["working_image_path", "hand_detected", "hand_mask_path"],
        state_writes=["hand_inpainted", "working_image_path"],
        preconditions=["hand_detected == True"],
        cost_class=CostClass.SLOW,
        idempotent=True,
        failure_modes=["no_hand_mask", "inpaint_artifact"],
        fallback_tool="isolate",  # Skip inpainting if fails
    ),
    
    "remove_overlay": ToolDefinition(
        name="remove_overlay",
        description="Remove detected text overlay via inpainting",
        category=ToolCategory.PROCESSING,
        state_reads=["working_image_path", "overlay_detected", "overlay_mask_path"],
        state_writes=["overlay_removed", "working_image_path"],
        preconditions=["overlay_detected == True"],
        cost_class=CostClass.SLOW,
        idempotent=True,
        failure_modes=["no_overlay_mask", "inpaint_artifact"],
    ),
    
    "correct_exposure": ToolDefinition(
        name="correct_exposure",
        description="Fix brightness and contrast issues",
        category=ToolCategory.PROCESSING,
        state_reads=["working_image_path", "brightness_score"],
        state_writes=["working_image_path"],
        preconditions=[],
        cost_class=CostClass.FAST,
        idempotent=True,
        failure_modes=["overcorrection"],
    ),
    
    "isolate": ToolDefinition(
        name="isolate",
        description="Segment product from background, producing RGBA with alpha mask",
        category=ToolCategory.PROCESSING,
        state_reads=["working_image_path", "hand_inpainted"],
        state_writes=["background_removed", "rgba_path", "alpha_mask_path", "working_image_path"],
        preconditions=["scene_type != 'already_white'"],
        cost_class=CostClass.SLOW,
        idempotent=True,
        failure_modes=["no_foreground", "multiple_products", "timeout"],
    ),
    
    "compose": ToolDefinition(
        name="compose",
        description="Place isolated product on 2000x2000 white canvas",
        category=ToolCategory.PROCESSING,
        state_reads=["rgba_path", "background_removed"],
        state_writes=["composed_on_white", "composed_path", "working_image_path"],
        preconditions=["background_removed == True"],
        cost_class=CostClass.FAST,
        idempotent=True,
        failure_modes=["empty_mask"],
    ),
    
    "shadow": ToolDefinition(
        name="shadow",
        description="Add subtle grounding shadow beneath product",
        category=ToolCategory.PROCESSING,
        state_reads=["working_image_path", "composed_on_white"],
        state_writes=["shadow_added", "working_image_path"],
        preconditions=["composed_on_white == True"],
        cost_class=CostClass.FAST,
        idempotent=True,
        failure_modes=["shadow_misaligned"],
    ),
    
    "polish": ToolDefinition(
        name="polish",
        description="Apply subtle contrast and sharpness enhancement",
        category=ToolCategory.PROCESSING,
        state_reads=["working_image_path", "shadow_added"],
        state_writes=["polished", "working_image_path"],
        preconditions=["shadow_added == True or scene_type == 'already_white'"],
        cost_class=CostClass.FAST,
        idempotent=True,
        failure_modes=["over_sharpened"],
    ),
    
    "compress": ToolDefinition(
        name="compress",
        description="Export final image as WebP/JPEG with thumbnails",
        category=ToolCategory.PROCESSING,
        state_reads=["working_image_path", "polished"],
        state_writes=["exported", "final_path"],
        preconditions=["polished == True"],
        cost_class=CostClass.FAST,
        idempotent=True,
        failure_modes=["disk_full"],
    ),
    
    # ─────────────────────────────────────────────────────────────────────
    # Validation Tools
    # ─────────────────────────────────────────────────────────────────────
    "validate_background": ToolDefinition(
        name="validate_background",
        description="Verify background is pure white with no artifacts",
        category=ToolCategory.VALIDATION,
        state_reads=["working_image_path", "composed_on_white"],
        state_writes=[],
        preconditions=["composed_on_white == True"],
        cost_class=CostClass.FAST,
        idempotent=True,
        failure_modes=["edge_artifacts"],
    ),
    
    "validate_quality": ToolDefinition(
        name="validate_quality",
        description="Final quality gate - check overall quality meets threshold",
        category=ToolCategory.VALIDATION,
        state_reads=["overall_quality"],
        state_writes=[],
        preconditions=[],
        cost_class=CostClass.FAST,
        idempotent=True,
        failure_modes=["quality_below_threshold"],
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Lazy Initialization
# ─────────────────────────────────────────────────────────────────────────────

_default_tools_initialized = False


def _ensure_default_tools() -> None:
    """
    Ensure default tool definitions are loaded.
    
    This populates the definitions cache without requiring
    actual tool implementations to exist yet.
    """
    global _default_tools_initialized
    if _default_tools_initialized:
        return
    _default_tools_initialized = True
    
    # Populate definitions cache from TOOL_DEFINITIONS
    for name, defn in TOOL_DEFINITIONS.items():
        _DEFINITIONS_CACHE[name.lower()] = defn
    
    LOGGER.debug("Loaded %d tool definitions", len(TOOL_DEFINITIONS))


def clear_registry() -> None:
    """
    Clear all registered tools (for testing).
    """
    global _default_tools_initialized
    _TOOL_REGISTRY.clear()
    _DEFINITIONS_CACHE.clear()
    _default_tools_initialized = False
