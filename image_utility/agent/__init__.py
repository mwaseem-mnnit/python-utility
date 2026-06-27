"""
Image Agent - Adaptive image processing using tools.

This package provides an agent-based architecture for processing
ecommerce product images. The agent:

1. Analyzes images to detect issues (hands, clutter, poor lighting)
2. Plans which tools to execute
3. Executes tools to transform images
4. Reviews results and iterates until quality goals are met

Key components:
- state.py: ImageState - shared state between tools
- contracts.py: ToolInput, ToolResult, Tool - tool interfaces
- registry.py: Tool registry and lookup
- tools/: Individual tool implementations

Usage:
    from image_utility.agent import create_initial_state, get_tool, ImageState
    
    state = create_initial_state(source_path, workdir)
    tool = get_tool("isolate")
    result = tool.execute(ToolInput(image_path, state, config, workdir))
"""

from .state import (
    ImageState,
    Issue,
    ToolHistoryEntry,
    create_initial_state,
)

from .contracts import (
    Tool,
    ToolCategory,
    ToolDefinition,
    ToolInput,
    ToolResult,
    CostClass,
    IssueType,
    ISSUE_TO_TOOL,
)

from .registry import (
    register_tool,
    get_tool,
    get_tool_definition,
    list_tools,
    list_tool_definitions,
    list_tools_by_category,
    check_tool_preconditions,
    get_tools_for_issue,
    format_tools_for_planner,
    TOOL_DEFINITIONS,
)

__all__ = [
    # State
    "ImageState",
    "Issue",
    "ToolHistoryEntry",
    "create_initial_state",
    
    # Contracts
    "Tool",
    "ToolCategory",
    "ToolDefinition",
    "ToolInput",
    "ToolResult",
    "CostClass",
    "IssueType",
    "ISSUE_TO_TOOL",
    
    # Registry
    "register_tool",
    "get_tool",
    "get_tool_definition",
    "list_tools",
    "list_tool_definitions",
    "list_tools_by_category",
    "check_tool_preconditions",
    "get_tools_for_issue",
    "format_tools_for_planner",
    "TOOL_DEFINITIONS",
]
