import argparse
import json
import sys

from . import __version__
from .config import LLMSettings, PROJECT_ROOT, WORKSPACE_ROOT
from .engine import AgentEngine, AgentError
from .llm import LLMError, OpenAICompatibleClient
from .models import ToolCall, ToolResult
from .runtime.executor import Executor
from .runtime.policy import Policy
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
            .replace("positional arguments:", "аргументы:", 1)
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

    subparsers = parser.add_subparsers(
        dest="command",
        parser_class=RussianArgumentParser,
    )

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

    run_parser = subparsers.add_parser(
        "run",
        help="запустить агента для выполнения задачи",
        description="запуск агента для работы с проектом",
        add_help=False,
    )

    run_parser.add_argument(
        "-h",
        "--help",
        action="help",
        help="показать справку и выйти",
    )

    run_parser.add_argument(
        "prompt",
        type=str,
        help="задача для выполнения",
    )

    run_parser.add_argument(
        "--max-steps",
        type=int,
        default=20,
        help="максимальное количество обращений к модели",
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

        tools = build_tool_registry(
            WORKSPACE_ROOT
        ).openai_tools()

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

        response = client.complete(
            messages,
            tools=tools,
        )

    except (LLMError, ValueError) as exc:
        print(f"ошибка: {exc}", file=sys.stderr)
        return 1

    if response.tool_calls:
        for call in response.tool_calls:
            print(
                f"модель запросила инструмент: {call.name}"
            )

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
        content = format_model_text(
            response.content or ""
        )

        print(f"ответ модели: {content}")

        if args.tools:
            print("модель не запросила инструменты")

    return 0


def print_tool_result(
    call: ToolCall,
    result: ToolResult,
) -> None:
    if not result.success:
        print(
            f"инструмент {call.name}: ошибка — {result.error}"
        )
        return

    print(f"инструмент {call.name}: выполнен")

    if call.name == "Bash":
        data = result.data or {}

        print(
            f"код завершения команды: {data.get('exit_code')}"
        )

        if data.get("timed_out"):
            print("превышено время выполнения команды")

        if data.get("truncated"):
            print("вывод команды был обрезан")


def run_agent(args: argparse.Namespace) -> int:
    try:
        settings = LLMSettings.from_env()

        policy = Policy(
            PROJECT_ROOT / "policy.json",
            WORKSPACE_ROOT,
        )

        registry = build_tool_registry(
            WORKSPACE_ROOT,
            policy=policy,
        )

        executor = Executor(
            registry=registry,
            policy=policy,
        )

        llm = OpenAICompatibleClient(settings)

        engine = AgentEngine(
            llm=llm,
            registry=registry,
            executor=executor,
            max_steps=args.max_steps,
            on_tool_result=print_tool_result,
        )

        print("агент запущен")

        result = engine.run(args.prompt)

    except (LLMError, AgentError, ValueError, OSError) as exc:
        print(f"ошибка: {exc}", file=sys.stderr)
        return 1

    print(f"обращений к модели: {result.model_calls}")
    print(f"вызовов инструментов: {result.tool_calls}")

    if result.status == "limit_reached":
        print(
            "достигнут лимит обращений к модели, "
            "выполнение задачи остановлено",
            file=sys.stderr,
        )
        return 2

    print(
        "ответ агента: "
        + format_model_text(result.final_answer or "")
    )

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "model-test":
        return run_model_test(args)

    if args.command == "run":
        return run_agent(args)

    parser.print_help()
    return 0