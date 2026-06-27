"""
Image Agent - Tool contracts and data structures.

This module defines the core contracts that all tools must implement:
- ToolInput: What tools receive
- ToolResult: What tools return
- ToolDefinition: Tool metadata for registry
- ToolCategory: Tool classification

These contracts ensure consistent tool behavior and enable
the planner to reason about tool capabilities without knowing
implementation details.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from .state import ImageState


class ToolCategory(str, Enum):
    """
    Tool categories determine when tools can be used.
    
    - ANALYSIS: Read-only tools that detect issues and measure quality
    - PROCESSING: Tools that modify the image
    - VALIDATION: Tools that verify results meet quality standards
    """
    ANALYSIS = "analysis"
    PROCESSING = "processing"
    VALIDATION = "validation"


class CostClass(str, Enum):
    """
    Execution time classification for planner optimization.
    """
    FAST = "fast"        # < 1 second
    MEDIUM = "medium"    # 1-5 seconds
    SLOW = "slow"        # > 5 seconds


@dataclass
class ToolInput:
    """
    Input structure provided to every tool.
    
    Attributes:
        image_path: Path to the current working image
        state: Current ImageState (read-only view)
        config: Tool-specific configuration overrides
        workdir: Directory for output files
    """
    image_path: Path
    state: ImageState
    config: dict = field(default_factory=dict)
    workdir: Path = field(default_factory=lambda: Path.cwd())
    
    def __post_init__(self):
        """Ensure paths are Path objects."""
        if isinstance(self.image_path, str):
            self.image_path = Path(self.image_path)
        if isinstance(self.workdir, str):
            self.workdir = Path(self.workdir)


@dataclass
class ToolResult:
    """
    Output structure returned by every tool.
    
    Attributes:
        success: Whether tool completed successfully
        output_image_path: Path to new image (if image was modified)
        state_updates: Dictionary of ImageState fields to update
        metadata: Tool-specific output data
        error: Error message if success=False
        confidence: Tool's confidence in the result (0.0-1.0)
        duration_ms: Execution time in milliseconds
    """
    success: bool
    output_image_path: Path | None = None
    state_updates: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    error: str | None = None
    confidence: float = 1.0
    duration_ms: int = 0
    
    @classmethod
    def failure(cls, error: str, duration_ms: int = 0) -> "ToolResult":
        """Create a failure result with error message."""
        return cls(
            success=False,
            error=error,
            confidence=0.0,
            duration_ms=duration_ms
        )
    
    @classmethod
    def success_no_change(cls, metadata: dict | None = None, duration_ms: int = 0) -> "ToolResult":
        """Create a success result for analysis tools that don't modify images."""
        return cls(
            success=True,
            metadata=metadata or {},
            duration_ms=duration_ms
        )


@dataclass
class ToolDefinition:
    """
    Tool metadata for registry and planner.
    
    The planner uses this metadata to:
    - Understand tool capabilities
    - Check preconditions before execution
    - Estimate execution cost
    - Handle failures appropriately
    
    Attributes:
        name: Unique tool identifier (e.g., "isolate")
        description: Human-readable description
        category: analysis | processing | validation
        
        input_schema: JSON Schema for config validation
        output_schema: JSON Schema for metadata validation
        
        state_reads: ImageState fields the tool reads
        state_writes: ImageState fields the tool updates
        preconditions: Conditions that must be true (Python expressions)
        
        cost_class: fast | medium | slow
        idempotent: Safe to re-run without side effects
        
        failure_modes: Known ways the tool can fail
        fallback_tool: Tool to try if this one fails
    """
    name: str
    description: str
    category: ToolCategory
    
    # Schema (for validation)
    input_schema: dict = field(default_factory=dict)
    output_schema: dict = field(default_factory=dict)
    
    # State dependencies
    state_reads: list[str] = field(default_factory=list)
    state_writes: list[str] = field(default_factory=list)
    
    # Preconditions (Python expressions evaluated against state)
    preconditions: list[str] = field(default_factory=list)
    
    # Execution characteristics
    cost_class: CostClass = CostClass.MEDIUM
    idempotent: bool = True
    
    # Failure handling
    failure_modes: list[str] = field(default_factory=list)
    fallback_tool: str | None = None
    
    def check_preconditions(self, state: ImageState) -> tuple[bool, str | None]:
        """
        Check if all preconditions are met.
        
        Args:
            state: Current ImageState
            
        Returns:
            (True, None) if all preconditions pass
            (False, reason) if any precondition fails
        """
        for condition in self.preconditions:
            try:
                # Create evaluation context with state fields
                context = {
                    "state": state,
                    # Direct access to common fields
                    "scene_type": state.scene_type,
                    "hand_detected": state.hand_detected,
                    "hand_inpainted": state.hand_inpainted,
                    "background_removed": state.background_removed,
                    "composed_on_white": state.composed_on_white,
                    "shadow_added": state.shadow_added,
                    "polished": state.polished,
                    "exported": state.exported,
                    "overlay_detected": state.overlay_detected,
                    "overlay_removed": state.overlay_removed,
                }
                
                if not eval(condition, {"__builtins__": {}}, context):
                    return False, f"Precondition failed: {condition}"
                    
            except Exception as e:
                return False, f"Precondition error '{condition}': {e}"
        
        return True, None
    
    def to_dict(self) -> dict:
        """Serialize for JSON export."""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "state_reads": self.state_reads,
            "state_writes": self.state_writes,
            "preconditions": self.preconditions,
            "cost_class": self.cost_class.value,
            "idempotent": self.idempotent,
            "failure_modes": self.failure_modes,
            "fallback_tool": self.fallback_tool
        }


class Tool(ABC):
    """
    Abstract base class for all tools.
    
    Every tool must:
    1. Define a ToolDefinition
    2. Implement execute()
    
    Tools must NOT:
    - Call other tools
    - Make planning decisions
    - Modify state directly (only return state_updates)
    """
    
    @property
    @abstractmethod
    def definition(self) -> ToolDefinition:
        """Return the tool's metadata definition."""
        pass
    
    @property
    def name(self) -> str:
        """Tool name (convenience accessor)."""
        return self.definition.name
    
    @property
    def category(self) -> ToolCategory:
        """Tool category (convenience accessor)."""
        return self.definition.category
    
    @abstractmethod
    def execute(self, input: ToolInput) -> ToolResult:
        """
        Execute the tool.
        
        Args:
            input: ToolInput with image path, state, config
            
        Returns:
            ToolResult with success status, output path, state updates
        """
        pass
    
    def check_preconditions(self, state: ImageState) -> tuple[bool, str | None]:
        """Check if tool can run given current state."""
        return self.definition.check_preconditions(state)
    
    def __repr__(self) -> str:
        return f"<Tool:{self.name}>"


# ─────────────────────────────────────────────────────────────────────────────
# Issue Type Constants
# ─────────────────────────────────────────────────────────────────────────────

class IssueType:
    """
    Standard issue types that tools can detect and planner can resolve.
    
    These are the canonical issue types used in Issue.type field.
    """
    # Scene issues
    HAND_VISIBLE = "hand_visible"
    SUPPORT_OBJECT_VISIBLE = "support_object_visible"
    BACKGROUND_CLUTTER = "background_clutter"
    TEXT_OVERLAY = "text_overlay"
    
    # Quality issues
    UNDEREXPOSED = "underexposed"
    OVEREXPOSED = "overexposed"
    LOW_BRIGHTNESS = "low_brightness"
    LOW_CONTRAST = "low_contrast"
    BLUR_DETECTED = "blur_detected"
    
    # Processing issues
    BACKGROUND_NOT_REMOVED = "background_not_removed"
    NOT_COMPOSED = "not_composed"
    NO_SHADOW = "no_shadow"
    NOT_POLISHED = "not_polished"
    NOT_EXPORTED = "not_exported"


# ─────────────────────────────────────────────────────────────────────────────
# Issue to Tool Mapping (for rule-based planner)
# ─────────────────────────────────────────────────────────────────────────────

ISSUE_TO_TOOL: dict[str, list[str]] = {
    IssueType.HAND_VISIBLE: ["inpaint_hand"],
    IssueType.TEXT_OVERLAY: ["remove_overlay"],
    IssueType.UNDEREXPOSED: ["correct_exposure"],
    IssueType.LOW_BRIGHTNESS: ["correct_exposure"],
    IssueType.BACKGROUND_NOT_REMOVED: ["isolate"],
    IssueType.NOT_COMPOSED: ["compose"],
    IssueType.NO_SHADOW: ["shadow"],
    IssueType.NOT_POLISHED: ["polish"],
    IssueType.NOT_EXPORTED: ["compress"],
}
