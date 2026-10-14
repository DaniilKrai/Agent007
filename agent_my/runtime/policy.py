import json
import re

from pathlib import Path

from agent_my.tools.filesystem import FileTools


class Policy:
    _SIMPLE_COMMAND = re.compile(
        r"[A-Za-z0-9_./:=+@%-]+(?: [A-Za-z0-9_./:=+@%-]+)*"
    )

    def __init__(self, policy_path: Path, workspace_root: Path):
        self.files = FileTools(workspace_root)
        self.root = self.files.root

        self.policy_path = policy_path.resolve(strict=True)

        if self.policy_path.is_relative_to(self.root):
            raise ValueError(
                "файл policy должен находиться вне рабочей директории"
            )

        try:
            data = json.loads(
                self.policy_path.read_text(encoding="utf-8")
            )
        except json.JSONDecodeError as exc:
            raise ValueError(
                "некорректный json файла policy"
            ) from exc

        if not isinstance(data, dict):
            raise ValueError("policy должен быть объектом")

        commands = data.get("commands")
        directories = data.get("directories")
        files = data.get("files", {})

        if not isinstance(commands, dict):
            raise ValueError("отсутствуют настройки команд")

        if not isinstance(directories, dict):
            raise ValueError("отсутствуют настройки директорий")

        if not isinstance(files, dict):
            raise ValueError("некорректные настройки файлов")

        self.commands_allow = set(
            self._read_list(commands, "allow", "commands")
        )

        self.commands_deny = set(
            self._read_list(commands, "deny", "commands")
        )

        self.directories_allow = tuple(
            self.files._resolve(path)
            for path in self._read_list(
                directories, "allow", "directories"
            )
        )

        self.directories_deny = tuple(
            self.files._resolve(path)
            for path in self._read_list(
                directories, "deny", "directories"
            )
        )

        self.files_deny = set(
            self.files._resolve(path)
            for path in self._read_list(
                files, "deny", "files"
            )
        )

        if not self.directories_allow:
            raise ValueError(
                "не указаны разрешённые директории"
            )

    def reload(self) -> None:
        updated = Policy(self.policy_path, self.root)
        self.commands_allow = updated.commands_allow
        self.commands_deny = updated.commands_deny
        self.directories_allow = updated.directories_allow
        self.directories_deny = updated.directories_deny
        self.files_deny = updated.files_deny

    @staticmethod
    def _read_list(data: dict, key: str, group: str) -> list[str]:
        values = data.get(key, [])

        if not isinstance(values, list):
            raise ValueError(
                f"поле {group}.{key} должно быть массивом"
            )

        if not all(
            isinstance(value, str) and value
            for value in values
        ):
            raise ValueError(
                f"поле {group}.{key} содержит некорректные значения"
            )

        return values

    def _check_path(self, path: str) -> Path:
        target = self.files._resolve(path)

        allowed = any(
            target.is_relative_to(directory)
            for directory in self.directories_allow
        )

        if not allowed:
            raise PermissionError(
                f"доступ к пути не разрешён: {path}"
            )

        denied = any(
            target.is_relative_to(directory)
            for directory in self.directories_deny
        )

        if denied:
            raise PermissionError(
                f"доступ к пути запрещён: {path}"
            )

        return target

    def check_file(self, tool_name: str, path: str) -> Path:
        target = self._check_path(path)

        if tool_name in ("write_file", "delete_file"):
            if target in self.files_deny:
                raise PermissionError(
                    f"изменение файла запрещено: {path}"
                )

        return target

    def check_bash(self, command: str, cwd: str) -> Path:
        if not isinstance(command, str) or not command:
            raise ValueError(
                "команда должна быть непустой строкой"
            )

        if command in self.commands_deny:
            raise PermissionError(
                f"команда запрещена правилами безопасности: {command}"
            )

        if command not in self.commands_allow:
            raise PermissionError(
                f"команда не входит в список разрешённых: {command}"
            )

        if self._SIMPLE_COMMAND.fullmatch(command) is None:
            raise PermissionError(
                "использование сложного синтаксиса bash запрещено"
            )

        arguments = command.split(" ")

        supported = (
            arguments == ["pwd"]
            or arguments == ["python3", "-m", "unittest", "-v"]
            or arguments == ["python", "-m", "unittest", "-v"]
        )

        if not supported:
            raise PermissionError(
                "команда содержит неподдерживаемую программу или аргументы"
            )

        directory = self._check_path(cwd)

        if not directory.is_dir():
            raise NotADirectoryError(
                f"рабочая директория не найдена: {cwd}"
            )

        return directory
