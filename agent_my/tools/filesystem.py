
from pathlib import Path, PureWindowsPath


class FileTools:
    def __init__(self, workspace_root: Path):
        self.root = workspace_root.resolve(strict=True)

        if not self.root.is_dir():
            raise ValueError("Workspace must be a directory")

    def _resolve(self, path: str) -> Path:
        if not isinstance(path, str) or not path:
            raise ValueError("Path must be a non-empty string")

        if "\0" in path or "\\" in path:
            raise ValueError("Invalid path")

        relative = Path(path)

        if relative.is_absolute() or PureWindowsPath(path).drive:
            raise PermissionError("Absolute paths are forbidden")

        if ".." in relative.parts:
            raise PermissionError("Parent traversal is forbidden")

        current = self.root

        for part in relative.parts:
            current = current / part

            if current.is_symlink():
                raise PermissionError("Symbolic links are forbidden")

        resolved = current.resolve(strict=False)

        if not resolved.is_relative_to(self.root):
            raise PermissionError("Path is outside workspace")

        return resolved

    def list_files(self, path: str) -> dict:
        target = self._resolve(path)

        if not target.is_dir():
            raise NotADirectoryError(path)

        items = []

        for item in sorted(target.iterdir(), key=lambda p: p.name):
            if item.is_symlink():
                item_type = "symlink"
            elif item.is_dir():
                item_type = "directory"
            else:
                item_type = "file"

            items.append({
                "name": item.name,
                "path": item.relative_to(self.root).as_posix(),
                "type": item_type,
            })

        return {
            "path": target.relative_to(self.root).as_posix(),
            "items": items,
        }

    def read_file(self, path: str) -> dict:
        target = self._resolve(path)

        if not target.is_file():
            raise FileNotFoundError(path)

        content = target.read_text(encoding="utf-8")

        return {
            "path": target.relative_to(self.root).as_posix(),
            "content": content,
        }

    def write_file(self, path: str, content: str) -> dict:
        target = self._resolve(path)

        if not target.parent.is_dir():
            raise FileNotFoundError("Parent directory does not exist")

        if target.is_dir():
            raise IsADirectoryError(path)

        existed = target.exists()
        target.write_text(content, encoding="utf-8")

        return {
            "path": target.relative_to(self.root).as_posix(),
            "operation": "updated" if existed else "created",
            "bytes_written": len(content.encode("utf-8")),
        }

    def delete_file(self, path: str) -> dict:
        target = self._resolve(path)

        if not target.is_file():
            raise FileNotFoundError(path)

        target.unlink()

        return {
            "path": target.relative_to(self.root).as_posix(),
            "deleted": True,
        }
