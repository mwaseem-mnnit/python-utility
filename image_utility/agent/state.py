"""
Image Agent - Shared state model for agent-tool communication.

This module defines ImageState, the central data structure that:
- Tracks all detected issues and quality metrics
- Records processing state (which tools have run)
- Stores artifact paths (RGBA, masks, composed images)
- Maintains tool execution history
- Controls agent loop behavior

Every tool reads from and writes to ImageState.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TypeAlias

# Type aliases for clarity
SceneType: TypeAlias = Literal["hand_held", "table_surface", "already_white", "unknown"]
BackgroundType: TypeAlias = Literal["cluttered", "plain", "white", "unknown"]
SupportObjectType: TypeAlias = Literal["table", "stand", "hand", "box", "platform", None]
Severity: TypeAlias = Literal["critical", "high", "medium", "low"]
Status: TypeAlias = Literal["pending", "processing", "review", "done", "failed"]


@dataclass
class Issue:
    """
    Represents a detected issue in the image.
    
    Issues are identified by the reviewer and consumed by the planner
    to determine which tools to execute.
    """
    type: str                  # e.g., "hand_visible", "low_brightness", "background_clutter"
    severity: Severity         # Affects tool prioritization
    details: dict = field(default_factory=dict)  # Tool-specific context
    
    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "severity": self.severity,
            "details": self.details
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Issue":
        return cls(
            type=data["type"],
            severity=data.get("severity", "medium"),
            details=data.get("details", {})
        )


@dataclass
class ToolHistoryEntry:
    """
    Records a single tool execution for debugging and rollback.
    """
    tool: str                  # Tool name
    success: bool              # Did tool complete successfully
    timestamp: str             # ISO format timestamp
    duration_ms: int           # Execution duration in milliseconds
    confidence: float = 0.0    # Tool's confidence in result (0-1)
    error: str | None = None   # Error message if failed
    
    def to_dict(self) -> dict:
        return {
            "tool": self.tool,
            "success": self.success,
            "timestamp": self.timestamp,
            "duration_ms": self.duration_ms,
            "confidence": self.confidence,
            "error": self.error
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "ToolHistoryEntry":
        return cls(
            tool=data["tool"],
            success=data["success"],
            timestamp=data["timestamp"],
            duration_ms=data["duration_ms"],
            confidence=data.get("confidence", 0.0),
            error=data.get("error")
        )


@dataclass
class ImageState:
    """
    Central state object shared between agent components and tools.
    
    Field ownership is strictly enforced - each tool may only update
    fields it owns (see TOOL_INTERFACE_AUTHORITY.md).
    
    Categories:
    - Source: Original and working image paths
    - Classification: Scene type detection results
    - Detections: Hand, support object, background analysis
    - Quality: Brightness, contrast, sharpness metrics
    - Processing: Flags for completed processing steps
    - Artifacts: Paths to generated intermediate files
    - Issues: Problems detected by reviewer
    - History: Tool execution log
    - Control: Agent loop state
    """
    
    # ─────────────────────────────────────────────────────────────────────────
    # Source Information
    # ─────────────────────────────────────────────────────────────────────────
    source_path: Path                              # Original input image (never modified)
    working_image_path: Path                       # Current version being processed
    workdir: Path                                  # Working directory for outputs
    
    # ─────────────────────────────────────────────────────────────────────────
    # Scene Classification (owned by: classify_scene)
    # ─────────────────────────────────────────────────────────────────────────
    scene_type: SceneType = "unknown"
    scene_confidence: float = 0.0
    
    # ─────────────────────────────────────────────────────────────────────────
    # Detections (owned by: detect_* tools)
    # ─────────────────────────────────────────────────────────────────────────
    # Hand detection (owned by: detect_hand)
    hand_detected: bool = False
    hand_confidence: float = 0.0
    hand_mask_path: Path | None = None
    
    # Support object detection (owned by: detect_support_object)
    support_object_detected: bool = False
    support_object_type: SupportObjectType = None
    
    # Background analysis (owned by: detect_background)
    background_type: BackgroundType = "unknown"
    
    # Product detection (owned by: detect_product)
    product_detected: bool = False
    product_bbox: tuple[int, int, int, int] | None = None  # (x, y, w, h)
    
    # Box/packaging detection (owned by: detect_packaging)
    packaging_detected: bool = False
    packaging_bbox: tuple[int, int, int, int] | None = None
    keep_packaging: bool = True  # Default: keep product packaging in final image
    
    # Text overlay detection (owned by: detect_overlay)
    overlay_detected: bool = False
    overlay_mask_path: Path | None = None
    
    # ─────────────────────────────────────────────────────────────────────────
    # Quality Metrics (owned by: measure_quality)
    # ─────────────────────────────────────────────────────────────────────────
    brightness_score: float = 0.0        # 0.0-1.0 (0.4-0.6 is ideal)
    contrast_score: float = 0.0          # 0.0-1.0
    sharpness_score: float = 0.0         # 0.0-1.0
    overall_quality: float = 0.0         # 0.0-1.0 composite score
    
    # ─────────────────────────────────────────────────────────────────────────
    # Processing State (owned by: respective processing tools)
    # ─────────────────────────────────────────────────────────────────────────
    overlay_removed: bool = False        # owned by: remove_overlay
    hand_inpainted: bool = False         # owned by: inpaint_hand
    background_removed: bool = False     # owned by: isolate
    composed_on_white: bool = False      # owned by: compose
    shadow_added: bool = False           # owned by: shadow
    polished: bool = False               # owned by: polish
    exported: bool = False               # owned by: compress
    
    # ─────────────────────────────────────────────────────────────────────────
    # Artifact Paths (owned by: respective processing tools)
    # ─────────────────────────────────────────────────────────────────────────
    rgba_path: Path | None = None           # Isolated RGBA (owned by: isolate)
    alpha_mask_path: Path | None = None     # Binary alpha mask (owned by: isolate)
    composed_path: Path | None = None       # Composed on white (owned by: compose)
    final_path: Path | None = None          # Final exported (owned by: compress)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Issues (owned by: reviewer)
    # ─────────────────────────────────────────────────────────────────────────
    issues: list[Issue] = field(default_factory=list)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Tool History (owned by: executor)
    # ─────────────────────────────────────────────────────────────────────────
    tool_history: list[ToolHistoryEntry] = field(default_factory=list)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Agent Control (owned by: agent loop)
    # ─────────────────────────────────────────────────────────────────────────
    iteration: int = 0
    max_iterations: int = 5
    status: Status = "pending"
    
    # ─────────────────────────────────────────────────────────────────────────
    # Version tracking for file naming
    # ─────────────────────────────────────────────────────────────────────────
    _version: int = field(default=0, repr=False)
    
    def next_version(self) -> int:
        """Increment and return the next version number for output files."""
        self._version += 1
        return self._version
    
    @property
    def current_version(self) -> int:
        """Get current version number."""
        return self._version
    
    def get_versioned_path(self, suffix: str, ext: str = "jpg") -> Path:
        """
        Generate a versioned output path.
        
        Example: workdir/product_01_v3.jpg
        """
        stem = self.source_path.stem
        version = self.next_version()
        return self.workdir / f"{stem}_v{version}{suffix}.{ext}"
    
    def to_dict(self) -> dict:
        """Serialize state to dictionary for JSON export."""
        return {
            "source_path": str(self.source_path),
            "working_image_path": str(self.working_image_path),
            "workdir": str(self.workdir),
            "scene_type": self.scene_type,
            "scene_confidence": self.scene_confidence,
            "hand_detected": self.hand_detected,
            "hand_confidence": self.hand_confidence,
            "hand_mask_path": str(self.hand_mask_path) if self.hand_mask_path else None,
            "support_object_detected": self.support_object_detected,
            "support_object_type": self.support_object_type,
            "background_type": self.background_type,
            "product_detected": self.product_detected,
            "product_bbox": self.product_bbox,
            "packaging_detected": self.packaging_detected,
            "packaging_bbox": self.packaging_bbox,
            "keep_packaging": self.keep_packaging,
            "overlay_detected": self.overlay_detected,
            "overlay_mask_path": str(self.overlay_mask_path) if self.overlay_mask_path else None,
            "brightness_score": self.brightness_score,
            "contrast_score": self.contrast_score,
            "sharpness_score": self.sharpness_score,
            "overall_quality": self.overall_quality,
            "overlay_removed": self.overlay_removed,
            "hand_inpainted": self.hand_inpainted,
            "background_removed": self.background_removed,
            "composed_on_white": self.composed_on_white,
            "shadow_added": self.shadow_added,
            "polished": self.polished,
            "exported": self.exported,
            "rgba_path": str(self.rgba_path) if self.rgba_path else None,
            "alpha_mask_path": str(self.alpha_mask_path) if self.alpha_mask_path else None,
            "composed_path": str(self.composed_path) if self.composed_path else None,
            "final_path": str(self.final_path) if self.final_path else None,
            "issues": [i.to_dict() for i in self.issues],
            "tool_history": [t.to_dict() for t in self.tool_history],
            "iteration": self.iteration,
            "max_iterations": self.max_iterations,
            "status": self.status,
            "_version": self._version
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "ImageState":
        """Deserialize state from dictionary."""
        state = cls(
            source_path=Path(data["source_path"]),
            working_image_path=Path(data["working_image_path"]),
            workdir=Path(data["workdir"]),
        )
        # Set all optional fields
        state.scene_type = data.get("scene_type", "unknown")
        state.scene_confidence = data.get("scene_confidence", 0.0)
        state.hand_detected = data.get("hand_detected", False)
        state.hand_confidence = data.get("hand_confidence", 0.0)
        state.hand_mask_path = Path(data["hand_mask_path"]) if data.get("hand_mask_path") else None
        state.support_object_detected = data.get("support_object_detected", False)
        state.support_object_type = data.get("support_object_type")
        state.background_type = data.get("background_type", "unknown")
        state.product_detected = data.get("product_detected", False)
        state.product_bbox = tuple(data["product_bbox"]) if data.get("product_bbox") else None
        state.packaging_detected = data.get("packaging_detected", False)
        state.packaging_bbox = tuple(data["packaging_bbox"]) if data.get("packaging_bbox") else None
        state.keep_packaging = data.get("keep_packaging", True)
        state.overlay_detected = data.get("overlay_detected", False)
        state.overlay_mask_path = Path(data["overlay_mask_path"]) if data.get("overlay_mask_path") else None
        state.brightness_score = data.get("brightness_score", 0.0)
        state.contrast_score = data.get("contrast_score", 0.0)
        state.sharpness_score = data.get("sharpness_score", 0.0)
        state.overall_quality = data.get("overall_quality", 0.0)
        state.overlay_removed = data.get("overlay_removed", False)
        state.hand_inpainted = data.get("hand_inpainted", False)
        state.background_removed = data.get("background_removed", False)
        state.composed_on_white = data.get("composed_on_white", False)
        state.shadow_added = data.get("shadow_added", False)
        state.polished = data.get("polished", False)
        state.exported = data.get("exported", False)
        state.rgba_path = Path(data["rgba_path"]) if data.get("rgba_path") else None
        state.alpha_mask_path = Path(data["alpha_mask_path"]) if data.get("alpha_mask_path") else None
        state.composed_path = Path(data["composed_path"]) if data.get("composed_path") else None
        state.final_path = Path(data["final_path"]) if data.get("final_path") else None
        state.issues = [Issue.from_dict(i) for i in data.get("issues", [])]
        state.tool_history = [ToolHistoryEntry.from_dict(t) for t in data.get("tool_history", [])]
        state.iteration = data.get("iteration", 0)
        state.max_iterations = data.get("max_iterations", 5)
        state.status = data.get("status", "pending")
        state._version = data.get("_version", 0)
        return state


def create_initial_state(source_path: Path, workdir: Path) -> ImageState:
    """
    Create a fresh ImageState for a new image.
    
    Args:
        source_path: Path to the input image
        workdir: Working directory for outputs
    
    Returns:
        Initialized ImageState with defaults
    """
    workdir.mkdir(parents=True, exist_ok=True)
    
    return ImageState(
        source_path=source_path,
        working_image_path=source_path,  # Initially points to source
        workdir=workdir,
    )
