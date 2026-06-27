#!/usr/bin/env python3
"""
Image Agent - Manual Tool Testing CLI.

Test individual tools or run a manual pipeline by specifying tools to execute.

Usage:
    # Test a single tool
    python -m image_utility.agent.tools.test_tool \\
        --tool isolate \\
        --image /path/to/image.jpg \\
        --workdir /tmp/test

    # Test with config overrides
    python -m image_utility.agent.tools.test_tool \\
        --tool compose \\
        --image /path/to/isolated.png \\
        --config '{"canvas_width": 1500}' \\
        --workdir /tmp/test

    # Run multiple tools in sequence (manual pipeline)
    python -m image_utility.agent.tools.test_tool \\
        --tools isolate,compose,shadow,polish \\
        --image /path/to/image.jpg \\
        --workdir /tmp/test

    # Load existing state and continue
    python -m image_utility.agent.tools.test_tool \\
        --tool compose \\
        --state /tmp/test/state.json \\
        --workdir /tmp/test

    # List available tools
    python -m image_utility.agent.tools.test_tool --list

    # Show tool definition
    python -m image_utility.agent.tools.test_tool --info isolate
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Ensure package is importable
if __package__ in (None, ""):
    _root = Path(__file__).resolve().parents[3]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from image_utility.agent import (
    ImageState,
    Issue,
    ToolHistoryEntry,
    ToolInput,
    ToolResult,
    create_initial_state,
    get_tool,
    get_tool_definition,
    list_tools,
    list_tool_definitions,
    format_tools_for_planner,
)
from image_utility.utils import load_image_utility_env


# ─────────────────────────────────────────────────────────────────────────────
# Output Formatting
# ─────────────────────────────────────────────────────────────────────────────

def print_header(text: str) -> None:
    """Print a section header."""
    width = 70
    print("═" * width)
    print(f" {text}")
    print("═" * width)


def print_subheader(text: str) -> None:
    """Print a subsection header."""
    print("─" * 50)
    print(f" {text}")
    print("─" * 50)


def print_result(result: ToolResult, tool_name: str) -> None:
    """Print formatted tool result."""
    print_subheader(f"RESULT: {tool_name}")
    
    status = "✓ SUCCESS" if result.success else "✗ FAILED"
    print(f"  Status:     {status}")
    print(f"  Confidence: {result.confidence:.2f}")
    print(f"  Duration:   {result.duration_ms}ms")
    
    if result.error:
        print(f"  Error:      {result.error}")
    
    if result.output_image_path:
        print(f"  Output:     {result.output_image_path}")
    
    if result.state_updates:
        print()
        print("  State Updates:")
        for key, value in result.state_updates.items():
            print(f"    {key}: {value}")
    
    if result.metadata:
        print()
        print("  Metadata:")
        for key, value in result.metadata.items():
            print(f"    {key}: {value}")


def print_state_summary(state: ImageState) -> None:
    """Print condensed state summary."""
    print_subheader("STATE SUMMARY")
    
    # Scene
    print(f"  Scene:      {state.scene_type} (conf: {state.scene_confidence:.2f})")
    
    # Detections
    detections = []
    if state.hand_detected:
        detections.append(f"hand({state.hand_confidence:.2f})")
    if state.packaging_detected:
        detections.append("packaging")
    if state.overlay_detected:
        detections.append("overlay")
    print(f"  Detections: {', '.join(detections) if detections else 'none'}")
    
    # Quality
    print(f"  Quality:    brightness={state.brightness_score:.2f} "
          f"contrast={state.contrast_score:.2f} "
          f"sharpness={state.sharpness_score:.2f} "
          f"overall={state.overall_quality:.2f}")
    
    # Processing
    steps = []
    if state.hand_inpainted:
        steps.append("inpainted")
    if state.background_removed:
        steps.append("isolated")
    if state.composed_on_white:
        steps.append("composed")
    if state.shadow_added:
        steps.append("shadowed")
    if state.polished:
        steps.append("polished")
    if state.exported:
        steps.append("exported")
    print(f"  Completed:  {' → '.join(steps) if steps else 'none'}")
    
    # Issues
    if state.issues:
        print(f"  Issues:     {len(state.issues)}")
        for issue in state.issues[:3]:  # Show first 3
            print(f"    - {issue.type} ({issue.severity})")
    
    # Working image
    print(f"  Working:    {state.working_image_path.name}")


def print_tool_info(name: str) -> None:
    """Print detailed tool definition."""
    defn = get_tool_definition(name)
    if defn is None:
        print(f"Tool '{name}' not found")
        return
    
    print_header(f"TOOL: {defn.name}")
    print(f"  Description:  {defn.description}")
    print(f"  Category:     {defn.category.value}")
    print(f"  Cost:         {defn.cost_class.value}")
    print(f"  Idempotent:   {defn.idempotent}")
    
    if defn.preconditions:
        print()
        print("  Preconditions:")
        for cond in defn.preconditions:
            print(f"    - {cond}")
    
    if defn.state_reads:
        print()
        print(f"  Reads:  {', '.join(defn.state_reads)}")
    
    if defn.state_writes:
        print(f"  Writes: {', '.join(defn.state_writes)}")
    
    if defn.failure_modes:
        print()
        print("  Failure Modes:")
        for mode in defn.failure_modes:
            print(f"    - {mode}")
    
    if defn.fallback_tool:
        print(f"  Fallback: {defn.fallback_tool}")


def list_all_tools() -> None:
    """Print all available tools."""
    print_header("AVAILABLE TOOLS")
    
    definitions = list_tool_definitions()
    
    # Group by category
    by_category: dict[str, list] = {}
    for defn in definitions:
        cat = defn.category.value
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(defn)
    
    for category in ["analysis", "processing", "validation"]:
        if category not in by_category:
            continue
        print()
        print(f"  {category.upper()}")
        print("  " + "-" * 40)
        for defn in sorted(by_category[category], key=lambda d: d.name):
            cost = defn.cost_class.value[0].upper()  # F/M/S
            print(f"    [{cost}] {defn.name:20} {defn.description[:40]}")


# ─────────────────────────────────────────────────────────────────────────────
# Tool Execution
# ─────────────────────────────────────────────────────────────────────────────

def load_state(state_path: Path | None, source_path: Path, workdir: Path) -> ImageState:
    """Load state from file or create new."""
    if state_path and state_path.exists():
        with open(state_path) as f:
            data = json.load(f)
        state = ImageState.from_dict(data)
        print(f"Loaded state from: {state_path}")
        return state
    
    return create_initial_state(source_path, workdir)


def save_state(state: ImageState, workdir: Path) -> Path:
    """Save state to JSON file."""
    state_path = workdir / "state.json"
    with open(state_path, "w") as f:
        json.dump(state.to_dict(), f, indent=2)
    return state_path


def run_tool(
    tool_name: str,
    state: ImageState,
    config: dict | None = None,
) -> tuple[ToolResult, ImageState]:
    """
    Execute a single tool and update state.
    
    Returns:
        (result, updated_state)
    """
    tool = get_tool(tool_name)
    
    if tool is None:
        # Tool not implemented yet - use definition to simulate
        defn = get_tool_definition(tool_name)
        if defn is None:
            return ToolResult.failure(f"Tool '{tool_name}' not found"), state
        
        print(f"  [WARN] Tool '{tool_name}' not yet implemented - skipping")
        return ToolResult(
            success=True,
            metadata={"simulated": True},
            confidence=0.0
        ), state
    
    # Create input
    tool_input = ToolInput(
        image_path=state.working_image_path,
        state=state,
        config=config or {},
        workdir=state.workdir,
    )
    
    # Execute
    result = tool.execute(tool_input)
    
    # Update state with result
    if result.success and result.state_updates:
        for key, value in result.state_updates.items():
            if hasattr(state, key):
                setattr(state, key, value)
    
    # Update working image path if tool produced output
    if result.success and result.output_image_path:
        state.working_image_path = result.output_image_path
    
    # Add to history
    state.tool_history.append(ToolHistoryEntry(
        tool=tool_name,
        success=result.success,
        timestamp=datetime.now().isoformat(),
        duration_ms=result.duration_ms,
        confidence=result.confidence,
        error=result.error,
    ))
    
    return result, state


def run_tools(
    tool_names: list[str],
    state: ImageState,
    configs: dict[str, dict] | None = None,
) -> ImageState:
    """
    Run multiple tools in sequence.
    
    Args:
        tool_names: List of tool names to execute
        state: Initial state
        configs: Optional per-tool config overrides {tool_name: config}
    
    Returns:
        Final state after all tools
    """
    configs = configs or {}
    
    for tool_name in tool_names:
        config = configs.get(tool_name)
        result, state = run_tool(tool_name, state, config)
        print_result(result, tool_name)
        
        if not result.success:
            print(f"\n[STOP] Tool '{tool_name}' failed, stopping pipeline")
            break
    
    return state


# ─────────────────────────────────────────────────────────────────────────────
# Interactive Mode
# ─────────────────────────────────────────────────────────────────────────────

def interactive_mode(state: ImageState) -> ImageState:
    """
    Interactive tool testing mode.
    
    After viewing the image, user can type tool names to execute.
    """
    print_header("INTERACTIVE MODE")
    print("Commands:")
    print("  <tool_name>         Run a tool (e.g., 'isolate')")
    print("  <tool> {...}        Run with config (e.g., 'compose {\"canvas_width\": 1500}')")
    print("  list                List available tools")
    print("  info <tool>         Show tool details")
    print("  state               Show current state")
    print("  save                Save state to workdir")
    print("  open                Open working image (macOS)")
    print("  quit / exit         Exit interactive mode")
    print()
    
    while True:
        try:
            cmd = input("tool> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        
        if not cmd:
            continue
        
        if cmd in ("quit", "exit", "q"):
            break
        
        if cmd == "list":
            list_all_tools()
            continue
        
        if cmd.startswith("info "):
            tool_name = cmd[5:].strip()
            print_tool_info(tool_name)
            continue
        
        if cmd == "state":
            print_state_summary(state)
            continue
        
        if cmd == "save":
            path = save_state(state, state.workdir)
            print(f"State saved to: {path}")
            continue
        
        if cmd == "open":
            import subprocess
            subprocess.run(["open", str(state.working_image_path)], check=False)
            continue
        
        # Parse tool command with optional config
        parts = cmd.split(" ", 1)
        tool_name = parts[0]
        config = None
        
        if len(parts) > 1 and parts[1].strip().startswith("{"):
            try:
                config = json.loads(parts[1])
            except json.JSONDecodeError as e:
                print(f"Invalid JSON config: {e}")
                continue
        
        # Run tool
        result, state = run_tool(tool_name, state, config)
        print_result(result, tool_name)
        print()
    
    return state


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Test image agent tools manually.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Actions
    parser.add_argument(
        "--list", action="store_true",
        help="List all available tools"
    )
    parser.add_argument(
        "--info", metavar="TOOL",
        help="Show detailed info for a tool"
    )
    
    # Single tool
    parser.add_argument(
        "--tool", "-t", metavar="TOOL",
        help="Single tool to execute"
    )
    
    # Multiple tools
    parser.add_argument(
        "--tools", metavar="TOOL,TOOL,...",
        help="Comma-separated list of tools to run in sequence"
    )
    
    # Input
    parser.add_argument(
        "--image", "-i", metavar="PATH",
        help="Input image path"
    )
    parser.add_argument(
        "--state", "-s", metavar="PATH",
        help="Load state from JSON file"
    )
    
    # Config
    parser.add_argument(
        "--config", "-c", metavar="JSON",
        help="Tool config as JSON string"
    )
    parser.add_argument(
        "--config-file", metavar="PATH",
        help="Tool config from JSON file"
    )
    
    # Output
    parser.add_argument(
        "--workdir", "-w", metavar="PATH",
        default="/tmp/agent_test",
        help="Working directory for outputs (default: /tmp/agent_test)"
    )
    
    # Modes
    parser.add_argument(
        "--interactive", "-I", action="store_true",
        help="Enter interactive mode after running tools"
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Minimal output"
    )
    
    args = parser.parse_args(argv)
    
    # Load env
    load_image_utility_env()
    
    # Handle info/list actions
    if args.list:
        list_all_tools()
        return 0
    
    if args.info:
        print_tool_info(args.info)
        return 0
    
    # Validate inputs for tool execution
    if args.tool or args.tools:
        if not args.image and not args.state:
            parser.error("--image or --state required when running tools")
    
    # Setup workdir
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    
    # Determine source image
    if args.state:
        state_path = Path(args.state)
        with open(state_path) as f:
            state_data = json.load(f)
        source_path = Path(state_data.get("source_path", state_data.get("working_image_path")))
    elif args.image:
        source_path = Path(args.image)
        state_path = None
    else:
        # Interactive mode without image
        if args.interactive:
            print("Interactive mode requires --image or --state")
            return 1
        parser.print_help()
        return 0
    
    # Validate source exists
    if not source_path.exists():
        print(f"Error: Image not found: {source_path}")
        return 1
    
    # Load or create state
    state = load_state(
        Path(args.state) if args.state else None,
        source_path,
        workdir
    )
    
    # Parse config
    config = None
    if args.config:
        config = json.loads(args.config)
    elif args.config_file:
        with open(args.config_file) as f:
            config = json.load(f)
    
    # Print header
    if not args.quiet:
        print_header("TOOL TEST")
        print(f"  Source:  {source_path}")
        print(f"  Workdir: {workdir}")
        print()
    
    # Run tools
    if args.tool:
        result, state = run_tool(args.tool, state, config)
        if not args.quiet:
            print_result(result, args.tool)
    
    elif args.tools:
        tool_list = [t.strip() for t in args.tools.split(",")]
        # If config provided, apply to all tools (or use config-per-tool in future)
        configs = {t: config for t in tool_list} if config else None
        state = run_tools(tool_list, state, configs)
    
    # Save state
    state_path = save_state(state, workdir)
    if not args.quiet:
        print()
        print_state_summary(state)
        print()
        print(f"State saved: {state_path}")
    
    # Interactive mode
    if args.interactive:
        state = interactive_mode(state)
        save_state(state, workdir)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
