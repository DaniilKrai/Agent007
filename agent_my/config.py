
import os

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = PROJECT_ROOT / "creating_space"

@dataclass(frozen=True)
class LLMSettings:
    base_url: str
    api_key: str
    model: str

    @classmethod
    def from_env(cls) -> "LLMSettings":
        base_url = os.getenv(
            "OPENAI_BASE_URL",
            "http://localhost:11434/v1/",
        ).strip()

        api_key = os.getenv(
            "OPENAI_API_KEY",
            "ollama",
        ).strip()

        model = os.getenv(
            "OPENAI_MODEL",
            "qwen3:4b",
        ).strip()

        if not base_url:
            raise ValueError("не задан адрес сервера модели")

        if not api_key:
            raise ValueError("не задан ключ доступа к модели")

        if not model:
            raise ValueError("не указана языковая модель")

        return cls(
            base_url=base_url,
            api_key=api_key,
            model=model,
        )