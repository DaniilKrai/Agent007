
import argparse
import json
import sys

from . import __version__
from .config import LLMSettings, WORKSPACE_ROOT
from .llm import LLMError, OpenAICompatibleClient
from .tools import build_tool_registry

class RussianArgumentParser(argparse.ArgumentParser):
    def format_usage(self):
        return super().format_usage().replace(
            "usage:", "использование:", 1
        )

    def format_help(self):
        return (
            super()
            .format_help()
            .replace("usage:", "использование:", 1)
            .replace("options:", "параметры:", 1)
        )

    def error(self, message):
        self.print_usage(sys.stderr)
        self.exit(
            2,
            "ошибка: некорректные аргументы командной строки\n",
        )

def build_parser() -> argparse.ArgumentParser:
    parser = RussianArgumentParser(
        prog="coding-agent",
        description="консольный агент для работы с кодом",
        add_help=False,
    )

    parser.add_argument(
        "-h",
        "--help",
        action="help",
        help="показать справку и выйти",
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"coding-agent {__version__}",
        help="показать версию программы и выйти",
    )

    subparsers = parser.add_subparsers(dest="command")

    model_parser = subparsers.add_parser(
        "model-test",
        help="проверить подключение к языковой модели",
        description="проверка подключения к языковой модели",
        add_help=False,
    )

    model_parser.add_argument(
        "-h",
        "--help",
        action="help",
        help="показать справку и выйти",
    )

    model_parser.add_argument(
        "--tools",
        action="store_true",
        help="передать модели описания инструментов",
    )

    model_parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="текст запроса к модели",
    )

    return parser

def format_model_text(content: str) -> str:
    text = content.strip()

    if not text:
        return "модель вернула пустой ответ"

    text = text[0].lower() + text[1:]
    return text.rstrip(".。")

def run_model_test(args: argparse.Namespace) -> int:
    if args.prompt is not None and not args.prompt.strip():
        print(
            "ошибка: текст запроса не должен быть пустым",
            file=sys.stderr,
        )
        return 2

    if args.tools:
        prompt = (
            "вызови инструмент list_files с аргументом path='.' "
            "чтобы узнать, какие файлы находятся в рабочей директории"
        )

        tools = build_tool_registry(WORKSPACE_ROOT).openai_tools()

    else:
        prompt = "поприветствуй пользователя одним предложением"
        tools = None

    if args.prompt is not None:
        prompt = args.prompt

    messages = [
        {
            "role": "system",
            "content": (
                "ты помощник по программированию "
                "отвечай пользователю по-русски, "
                "начинай сообщения со строчной буквы "
                "и не ставь точку в конце сообщения "
                "если доступны инструменты, ты можешь "
                "запрашивать их вызовы "
                "не утверждай, что выполнил инструмент, "
                "пока не получишь его результат"
            ),
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    try:
        settings = LLMSettings.from_env()
        client = OpenAICompatibleClient(settings)
        response = client.complete(messages, tools=tools)

    except (LLMError, ValueError) as exc:
        print(f"ошибка: {exc}", file=sys.stderr)
        return 1

    if response.tool_calls:
        for call in response.tool_calls:
            print(f"модель запросила инструмент: {call.name}")
            print(
                "аргументы: "
                + json.dumps(
                    call.arguments,
                    ensure_ascii=False,
                )
            )

        print(
            "инструменты не выполнялись, "
            "это только проверка подключения"
        )

    else:
        content = format_model_text(response.content or "")
        print(f"ответ модели: {content}")

        if args.tools:
            print("модель не запросила инструменты")

    return 0

def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "model-test":
        return run_model_test(args)

    parser.print_help()
    return 0