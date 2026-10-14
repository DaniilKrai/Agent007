from pathlib import Path
from typing import TYPE_CHECKING

from .filesystem import FileTools
from .registry import ToolRegistry
from agent_my.runtime.planning import PLAN_SCHEMA, submit_plan
from agent_my.runtime.tasks import COMPLETE_TASK_SCHEMA, complete_task

if TYPE_CHECKING:
    from agent_my.runtime.policy import Policy


def build_tool_registry(
    workspace_root: Path,
    policy: "Policy | None" = None,
) -> ToolRegistry:
    files = FileTools(workspace_root)
    registry = ToolRegistry()

    registry.register(
        name="complete_task",
        description="завершить текущую задачу после проверки результатов её инструментов, evidence содержит tool_call_id",
        parameters=COMPLETE_TASK_SCHEMA,
        handler=complete_task,
    )

    registry.register(
        name="submit_plan",
        description="сохранить план с целью, шагами, инструментами, рисками, допущениями и критерием готовности для утверждения пользователем",
        parameters=PLAN_SCHEMA,
        handler=submit_plan,
    )

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

    if policy is not None:
        from .bash import BashTool

        bash = BashTool(policy)

        registry.register(
            name="Bash",
            description='выполнить разрешённую команду через bash, обязательны command и cwd, для корня проекта cwd="."',
            parameters={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "minLength": 1,
                        "description": "команда bash",
                    },
                    "cwd": {
                        "type": "string",
                        "minLength": 1,
                        "description": (
                            'рабочая директория относительно workspace, для корня проекта укажи "."'
                        ),
                    },
                },
                "required": ["command", "cwd"],
                "additionalProperties": False,
            },
            handler=bash.run,
        )

    return registry
