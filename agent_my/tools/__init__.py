from pathlib import Path

from .filesystem import FileTools
from .registry import ToolRegistry


def build_tool_registry(workspace_root: Path) -> ToolRegistry:
    files = FileTools(workspace_root)
    registry = ToolRegistry()

    path_schema = {
        "type": "string",
        "minLength": 1,
        "description": "относительный путь внутри рабочей директории",
    }

    registry.register(
        name="list_files",
        description="получить список файлов и директорий",
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
        description="прочитать текстовый файл в кодировке utf-8",
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
        description="создать или перезаписать текстовый файл",
        parameters={
            "type": "object",
            "properties": {
                "path": path_schema,
                "content": {
                    "type": "string",
                    "description": "полное новое содержимое файла",
                },
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        },
        handler=files.write_file,
    )

    registry.register(
        name="delete_file",
        description="удалить файл внутри рабочей директории",
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