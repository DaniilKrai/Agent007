import argparse
import json
import sys
import uuid

from . import __version__
from .config import (
    LLMSettings,
    PROJECT_ROOT,
    STATE_DB,
    WORKSPACE_ROOT,
)
from .engine import AgentEngine, AgentError
from .llm import LLMError, OpenAICompatibleClient
from .models import ToolCall, ToolResult
from .runtime.executor import Executor
from .runtime.policy import Policy
from .runtime.store import RunStore, StoreError
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
        help="запустить агента",
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
        help="задача для агента",
    )

    run_parser.add_argument(
        "--max-steps",
        type=int,
        default=20,
        help="максимальное количество обращений к модели",
    )

    run_parser.add_argument(
        "--mode",
        choices=["safe", "auto"],
        default="safe",
        help="режим выполнения агента",
    )

    run_parser.add_argument(
        "--no-verify-tests",
        action="store_true",
        help="разрешить текстовое завершение без проверки тестов для задач анализа",
    )

    run_parser.add_argument(
        "--plan", action="store_true",
        help="изучить проект и предложить план до выполнения изменений",
    )

    resume_parser = subparsers.add_parser(
        "resume",
        help="продолжить сохранённый запуск",
        add_help=False,
    )

    resume_parser.add_argument(
        "run_id",
        type=str,
    )

    resolve_parser = subparsers.add_parser(
        "resolve",
        help="разрешить неопределённое действие",
        add_help=False,
    )

    resolve_parser.add_argument(
        "run_id",
        type=str,
    )

    resolve_parser.add_argument(
        "action",
        choices=["retry", "skip", "abort"],
    )

    status_parser = subparsers.add_parser(
        "status",
        help="посмотреть состояние запуска",
        add_help=False,
    )

    status_parser.add_argument(
        "run_id",
        type=str,
    )

    approve_parser = subparsers.add_parser(
        "approve",
        help="подтвердить ожидающее действие агента",
        add_help=False,
    )

    approve_parser.add_argument(
        "run_id",
        type=str,
    )

    reject_parser = subparsers.add_parser(
        "reject",
        help="отклонить ожидающее действие агента",
        add_help=False,
    )

    reject_parser.add_argument(
        "run_id",
        type=str,
    )

    for command, help_text in (
        ("plan", "посмотреть сохранённый план"),
        ("approve-plan", "утвердить план и начать исполнение"),
        ("reject-plan", "отклонить план без исполнения"),
    ):
        plan_parser = subparsers.add_parser(command, help=help_text, add_help=False)
        plan_parser.add_argument("run_id", type=str)

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
                "отвечай пользователю по-русски "
                "начинай сообщения со строчной буквы "
                "не ставь точку в конце сообщения "
                "не утверждай, что выполнил инструмент, "
                "пока не получишь его результат"
            ),
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    settings = LLMSettings.from_env()
    client = OpenAICompatibleClient(settings)

    response = client.complete(
        messages,
        tools=tools,
    )

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
        print(
            "ответ модели: "
            + format_model_text(response.content or "")
        )

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


def build_engine(
    max_steps: int = 20,
    mode: str = "safe",
    verify_tests: bool = True,
) -> AgentEngine:
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

    store = RunStore(STATE_DB)

    llm = OpenAICompatibleClient(settings)

    return AgentEngine(
        llm=llm,
        registry=registry,
        executor=executor,
        store=store,
        max_steps=max_steps,
        mode=mode,
        verify_tests=verify_tests,
        on_tool_result=print_tool_result,
    )

def print_plan(plan: dict) -> None:
    print("цель: " + format_model_text(plan["goal"]))
    print("шаги:")
    for number, step in enumerate(plan["steps"], 1):
        print(f"{number}) " + format_model_text(step["description"]))
        print("   инструменты: " + ", ".join(step["tools"]))
    for key, label in (("risks", "риски"), ("assumptions", "допущения")):
        print(label + ": " + (
            "; ".join(format_model_text(item) for item in plan[key]) or "не указаны"
        ))
    print("критерий готовности: " + format_model_text(plan["done_when"]))


def print_run_result(
    engine: AgentEngine,
    result,
) -> int:
    print(f"идентификатор запуска: {result.run_id}")
    print(f"состояние запуска: {result.status}")
    print(f"обращений к модели: {result.model_calls}")
    print(f"вызовов инструментов: {result.tool_calls}")

    if result.status == "waiting_plan":
        print_plan(engine.store.get_plan(result.run_id))
        print("агент ожидает утверждения плана")
        print(f"для утверждения: python -m agent_my approve-plan {result.run_id}")
        print(f"для отказа: python -m agent_my reject-plan {result.run_id}")
        return 3

    if result.status == "plan_rejected":
        print("план отклонён, исполнение не начато")
        return 0

    if result.status == "completed":
        print(
            "ответ агента: "
            + format_model_text(
                result.final_answer or ""
            )
        )
        return 0

    if result.status == "limit_reached":
        print(
            "достигнут лимит обращений к модели",
            file=sys.stderr,
        )
        return 2

    if result.status == "failed":
        print(
            result.final_answer or "запуск завершён с ошибкой",
            file=sys.stderr,
        )
        return 1

    if result.status == "waiting_approval":
        approval = engine.store.get_pending_approval(
            result.run_id
        )

        print("агент ожидает подтверждения действия")

        if approval is not None:
            print(
                f"инструмент: {approval['name']}"
            )

            arguments = json.loads(
                approval["arguments"]
            )

            if "path" in arguments:
                print(
                    f"путь: {arguments['path']}"
                )

            if "command" in arguments:
                print(
                    f"команда: {arguments['command']}"
                )

            if "cwd" in arguments:
                print(
                    f"рабочая директория: {arguments['cwd']}"
                )

            if approval["name"] == "write_file":
                content = arguments.get("content", "")
                print(
                    f"размер нового содержимого: "
                    f"{len(content.encode('utf-8'))} байт"
                )

        print(
            "для подтверждения: "
            f"python -m agent_my approve {result.run_id}"
        )

        print(
            "для отказа: "
            f"python -m agent_my reject {result.run_id}"
        )

        return 3

    if result.status == "waiting_resolution":
        call = engine.store.get_uncertain(
            result.run_id
        )

        print(
            "обнаружено действие с неопределённым результатом"
        )

        if call is not None:
            print(f"инструмент: {call['name']}")
            print(f"идентификатор вызова: {call['call_id']}")

            if call["name"] in (
                "read_file",
                "list_files",
                "submit_plan",
            ):
                print(
                    "доступные решения: retry, skip, abort"
                )
            else:
                print(
                    "повторение этого инструмента небезопасно"
                )
                print(
                    "доступные решения: skip, abort"
                )

        print(
            f"для продолжения используй команду "
            f"python -m agent_my resolve {result.run_id} <решение>"
        )

        return 3

    return 0


def run_agent(args: argparse.Namespace) -> int:
    engine = build_engine(
        max_steps=args.max_steps,
        mode=args.mode,
        verify_tests=not args.no_verify_tests,
    )

    run_id = str(uuid.uuid4())

    print("агент запущен")
    print(f"идентификатор запуска: {run_id}")

    result = engine.run(
        args.prompt,
        run_id=run_id,
        plan=args.plan,
    )

    return print_run_result(
        engine,
        result,
    )


def resume_agent(run_id: str) -> int:
    engine = build_engine()

    result = engine.resume(run_id)

    return print_run_result(
        engine,
        result,
    )


def resolve_action(
    run_id: str,
    action: str,
) -> int:
    engine = build_engine()

    result = engine.resolve(
        run_id,
        action,
    )

    return print_run_result(
        engine,
        result,
    )

def approve_action(run_id: str) -> int:
    engine = build_engine()

    result = engine.approve(run_id)

    return print_run_result(
        engine,
        result,
    )


def reject_action(run_id: str) -> int:
    engine = build_engine()

    result = engine.reject(run_id)

    return print_run_result(
        engine,
        result,
    )


def show_plan(run_id: str) -> int:
    store = RunStore(STATE_DB)
    try:
        plan = store.get_plan(run_id)
        if plan is None:
            raise StoreError("для этого запуска ещё нет сохранённого плана")
        print_plan(plan)
        return 0
    finally:
        store.close()


def decide_plan(run_id: str, approve: bool) -> int:
    engine = build_engine()
    result = engine.approve_plan(run_id) if approve else engine.reject_plan(run_id)
    return print_run_result(engine, result)


def show_status(run_id: str) -> int:
    store = RunStore(STATE_DB)

    try:
        run = store.get_run(run_id)

        print(f"идентификатор запуска: {run_id}")
        print(f"состояние запуска: {run['status']}")
        print(
            f"обращений к модели: {run['model_calls']}"
        )
        print(
            f"вызовов инструментов: {run['tool_calls']}"
        )

        print(f"режим выполнения: {run['mode']}")
        print("фаза: " + ("планирование" if run["phase"] == "planning" else "исполнение"))

        if run["status"] == "waiting_plan":
            print("агент ожидает утверждения плана")
            print(f"для просмотра: python -m agent_my plan {run_id}")

        if run["status"] == "waiting_approval":
            approval = store.get_pending_approval(run_id)

            if approval is not None:
                print(
                    f"ожидает подтверждения: {approval['name']}"
                )

        if run["status"] == "waiting_resolution":
            call = store.get_uncertain(run_id)

            if call is not None:
                print(
                    f"неопределённый инструмент: {call['name']}"
                )

        return 0

    finally:
        store.close()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "model-test":
            return run_model_test(args)

        if args.command == "run":
            return run_agent(args)

        if args.command == "plan":
            return show_plan(args.run_id)

        if args.command == "approve-plan":
            return decide_plan(args.run_id, approve=True)

        if args.command == "reject-plan":
            return decide_plan(args.run_id, approve=False)

        if args.command == "resume":
            return resume_agent(args.run_id)

        if args.command == "resolve":
            return resolve_action(
                args.run_id,
                args.action,
            )

        if args.command == "status":
            return show_status(args.run_id)

        if args.command == "approve":
            return approve_action(args.run_id)

        if args.command == "reject":
            return reject_action(args.run_id)

        parser.print_help()
        return 0

    except (
        LLMError,
        AgentError,
        StoreError,
        ValueError,
        OSError,
    ) as exc:
        print(
            f"ошибка: {exc}",
            file=sys.stderr,
        )
        return 1
