
from pathlib import Path

from .filesystem import FileTools
from .registry import ToolRegistry


def build_tool_registry(workspace_root: Path) -> ToolRegistry:
    files = FileTools(workspace_root)
    registry = ToolRegistry()

    path_schema = {
        "type": "string",
        "minLength": 1,
        "description": "Relative path inside workspace",
    }

    registry.register(
        name="list_files",
        description="List files and directories inside workspace",
        parameters={
            "type": "object",
            "properties": {
                "path": path_schema,
            },
            "required": ["path"],
            "additionalProperties": False,
        },
        handler=files.list_files,
    )

    registry.register(
        name="read_file",
        description="Read a UTF-8 text file inside workspace",
        parameters={
            "type": "object",
            "properties": {
                "path": path_schema,
            },
            "required": ["path"],
            "additionalProperties": False,
        },
        handler=files.read_file,
    )

    registry.register(
        name="write_file",
        description="Create or replace a UTF-8 text file inside workspace",
        parameters={
            "type": "object",
            "properties": {
                "path": path_schema,
                "content": {
                    "type": "string",
                    "description": "Complete new file content",
                },
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        },
        handler=files.write_file,
    )

    registry.register(
        name="delete_file",
        description="Delete a file inside workspace",
        parameters={
            "type": "object",
            "properties": {
                "path": path_schema,
            },
            "required": ["path"],
            "additionalProperties": False,
        },
        handler=files.delete_file,
    )

    return registry
