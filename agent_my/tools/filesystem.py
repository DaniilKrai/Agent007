from pathlib import Path, PureWindowsPath


class FileTools:
    def __init__(self, workspace_root: Path):
        root = workspace_root.resolve(strict=False)

        if not root.is_dir():
            raise ValueError("рабочая директория не найдена")

        self.root = root.resolve(strict=True)

    def _resolve(self, path: str) -> Path:
        if not isinstance(path, str) or not path:
            raise ValueError("путь должен быть непустой строкой")

        if "\0" in path or "\\" in path:
            raise ValueError("недопустимый путь")

        relative = Path(path)

        if relative.is_absolute() or PureWindowsPath(path).drive:
            raise PermissionError("абсолютные пути запрещены")

        if ".." in relative.parts:
            raise PermissionError("переход в родительскую директорию запрещён")

        current = self.root

        for part in relative.parts:
            current = current / part

            if current.is_symlink():
                raise PermissionError("использование символьных ссылок запрещено")

        resolved = current.resolve(strict=False)

        if not resolved.is_relative_to(self.root):
            raise PermissionError("путь находится за пределами рабочей директории")

        return resolved

    def list_files(self, path: str) -> dict:
        target = self._resolve(path)

        if not target.is_dir():
            raise NotADirectoryError(f"не является директорией: {path}")

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
            raise FileNotFoundError(f"файл не найден: {path}")

        content = target.read_text(encoding="utf-8")

        return {
            "path": target.relative_to(self.root).as_posix(),
            "content": content,
        }

    def write_file(self, path: str, content: str) -> dict:
        target = self._resolve(path)

        if not target.parent.is_dir():
            raise FileNotFoundError(
                f"родительская директория не найдена: {path}"
            )

        if target.is_dir():
            raise IsADirectoryError(
                f"нельзя записать файл вместо директории: {path}"
            )

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
            raise FileNotFoundError(f"файл не найден: {path}")

        target.unlink()

        return {
            "path": target.relative_to(self.root).as_posix(),
            "deleted": True,
        }