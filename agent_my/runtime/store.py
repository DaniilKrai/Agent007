import json
import sqlite3

from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence
from jsonschema import Draft202012Validator

from agent_my.models import LLMResponse, ToolCall, ToolResult
from .planning import PLAN_SCHEMA
from .budgets import BudgetLimits, MIN_RESULT_BYTES, MAX_RESULT_BYTES, SIDE_EFFECT_TOOLS, limit_result, result_size
from .tasks import TaskManager


class StoreError(Exception):
    pass


class RunStore:
    def __init__(self, path: Path | str):
        self.path = str(path)

        if self.path != ":memory:":
            Path(self.path).parent.mkdir(
                parents=True,
                exist_ok=True,
            )

        self.db = sqlite3.connect(
            self.path,
            timeout=30,
        )

        self.db.row_factory = sqlite3.Row
        self.tasks = TaskManager(self.db)

        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.execute("PRAGMA busy_timeout = 30000")

        if self.path != ":memory:":
            self.db.execute("PRAGMA journal_mode = WAL")

        self.db.execute("PRAGMA synchronous = FULL")

        self._create_tables()

    @contextmanager
    def _transaction(self):
        if self.db.in_transaction:
            yield
            return
        self.db.execute("BEGIN IMMEDIATE")
        with self.db:
            yield

    def _create_tables(self) -> None:
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                max_steps INTEGER NOT NULL,
                model_calls INTEGER NOT NULL DEFAULT 0,
                tool_calls INTEGER NOT NULL DEFAULT 0,
                final_answer TEXT,
                mode TEXT NOT NULL DEFAULT 'auto'
            );

            CREATE TABLE IF NOT EXISTS budgets (
                run_id TEXT PRIMARY KEY REFERENCES runs(run_id),
                tool_calls_limit INTEGER NOT NULL,
                side_effects_limit INTEGER NOT NULL,
                result_bytes_limit INTEGER NOT NULL,
                tool_calls_used INTEGER NOT NULL DEFAULT 0,
                side_effects_used INTEGER NOT NULL DEFAULT 0,
                result_bytes_used INTEGER NOT NULL DEFAULT 0,
                result_bytes_reserved INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS messages (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                payload TEXT NOT NULL,
                FOREIGN KEY (run_id)
                    REFERENCES runs(run_id)
            );

            CREATE TABLE IF NOT EXISTS tool_calls (
                run_id TEXT NOT NULL,
                call_id TEXT NOT NULL,
                name TEXT NOT NULL,
                arguments TEXT NOT NULL,
                status TEXT NOT NULL,
                attempt_count INTEGER NOT NULL DEFAULT 0,
                result TEXT,
                PRIMARY KEY (run_id, call_id),
                FOREIGN KEY (run_id)
                    REFERENCES runs(run_id)
            );

            CREATE TABLE IF NOT EXISTS pending_calls (
                run_id TEXT NOT NULL,
                position INTEGER NOT NULL,
                call_id TEXT NOT NULL,
                override_result TEXT,
                delivered INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (run_id, position),
                FOREIGN KEY (run_id, call_id)
                    REFERENCES tool_calls(run_id, call_id)
            );

            CREATE TABLE IF NOT EXISTS tool_attempts (
                run_id TEXT NOT NULL,
                call_id TEXT NOT NULL,
                attempt_no INTEGER NOT NULL,
                status TEXT NOT NULL,
                result TEXT,
                PRIMARY KEY (run_id, call_id, attempt_no),
                FOREIGN KEY (run_id, call_id)
                    REFERENCES tool_calls(run_id, call_id)
            );

            CREATE TABLE IF NOT EXISTS decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                call_id TEXT NOT NULL,
                action TEXT NOT NULL,
                FOREIGN KEY (run_id)
                    REFERENCES runs(run_id)
            );

            CREATE TABLE IF NOT EXISTS approvals (
                run_id TEXT NOT NULL,
                call_id TEXT NOT NULL,
                status TEXT NOT NULL,
                PRIMARY KEY (run_id, call_id),
                FOREIGN KEY (run_id, call_id)
                    REFERENCES tool_calls(run_id, call_id)
            );
        """)

        self.tasks.create_schema()
        with self._transaction():
            for table in ("tool_calls", "pending_calls"):
                names = {row["name"] for row in self.db.execute(f"PRAGMA table_info({table})")}
                if "task_id" not in names:
                    self.db.execute(f"ALTER TABLE {table} ADD COLUMN task_id TEXT")

        columns = {
            row["name"]
            for row in self.db.execute(
                "PRAGMA table_info(runs)"
            )
        }

        if "mode" not in columns:
            with self._transaction():
                self.db.execute(
                    """
                    ALTER TABLE runs
                    ADD COLUMN mode TEXT NOT NULL DEFAULT 'auto'
                    """
                )

        with self._transaction():
            for name, definition in (
                ("phase", "TEXT NOT NULL DEFAULT 'execution'"),
                ("plan", "TEXT"),
                ("plan_call_id", "TEXT"),
                ("model_started", "INTEGER NOT NULL DEFAULT 0"),
            ):
                if name not in columns:
                    self.db.execute(f"ALTER TABLE runs ADD COLUMN {name} {definition}")

        with self._transaction():
            pending_columns = {r["name"] for r in self.db.execute("PRAGMA table_info(pending_calls)")}
            for name in ("result_limit", "result_charged"):
                if name not in pending_columns:
                    self.db.execute(f"ALTER TABLE pending_calls ADD COLUMN {name} INTEGER NOT NULL DEFAULT 0")
            old_runs = self.db.execute(
                "SELECT run_id FROM runs WHERE run_id NOT IN (SELECT run_id FROM budgets)"
            ).fetchall()
            for row in old_runs:
                run_id = row["run_id"]
                history = self.get_messages(run_id)
                calls = [c for m in history for c in m.get("tool_calls", [])]
                used_tools = max(len(calls), self.get_run(run_id)["tool_calls"])
                used_effects = sum(c["function"]["name"] in SIDE_EFFECT_TOOLS for c in calls)
                used_bytes = sum(len(m["content"].encode("utf-8")) for m in history if m["role"] == "tool")
                defaults = BudgetLimits()
                self._create_budget(run_id, BudgetLimits(
                    max(defaults.tool_calls, used_tools),
                    max(defaults.side_effects, used_effects),
                    max(defaults.result_bytes, used_bytes),
                ))
                self.db.execute(
                    "UPDATE budgets SET tool_calls_used = ?, side_effects_used = ?, result_bytes_used = ? WHERE run_id = ?",
                    (used_tools, used_effects, used_bytes, run_id),
                )
                pending = self.db.execute(
                    "SELECT position FROM pending_calls WHERE run_id = ? AND delivered = 0", (run_id,),
                ).fetchall()
                if pending:
                    available = max(defaults.result_bytes, used_bytes) - used_bytes
                    allowance = min(MAX_RESULT_BYTES, available // len(pending))
                    if allowance < MIN_RESULT_BYTES:
                        self._budget_stop(run_id, "результатов инструментов")
                    else:
                        self.db.execute(
                            "UPDATE pending_calls SET result_limit = ? WHERE run_id = ? AND delivered = 0",
                            (allowance, run_id),
                        )
                        self.db.execute(
                            "UPDATE budgets SET result_bytes_reserved = ? WHERE run_id = ?",
                            (allowance * len(pending), run_id),
                        )

    def _create_budget(self, run_id: str, limits: BudgetLimits) -> None:
        self.db.execute(
            "INSERT INTO budgets (run_id, tool_calls_limit, side_effects_limit, result_bytes_limit) VALUES (?, ?, ?, ?)",
            (run_id, limits.tool_calls, limits.side_effects, limits.result_bytes),
        )

    def get_budget(self, run_id: str) -> dict:
        self.get_run(run_id)
        return dict(self.db.execute("SELECT * FROM budgets WHERE run_id = ?", (run_id,)).fetchone())

    def prepare_tasks(self, run_id: str):
        with self._transaction():
            run = self.get_run(run_id)
            if run["phase"] != "execution" or run["status"] != "running":
                return
            pending = self.next_pending(run_id)
            if pending is not None and not self.tasks.get_all(run_id):
                self.tasks.ensure_implicit(run_id)
                self.db.execute(
                    "UPDATE pending_calls SET task_id = 'task_1' WHERE run_id = ? AND delivered = 0", (run_id,),
                )
                self.db.execute(
                    "UPDATE tool_calls SET task_id = 'task_1' WHERE run_id = ? AND call_id IN "
                    "(SELECT call_id FROM pending_calls WHERE run_id = ? AND delivered = 0)", (run_id, run_id),
                )
            self.tasks.current(run_id)

    def _budget_stop(self, run_id: str, label: str) -> None:
        self.tasks.fail_current(run_id)
        self.db.execute(
            "UPDATE runs SET status = 'limit_reached', final_answer = ? WHERE run_id = ?",
            ("исчерпан бюджет " + label, run_id),
        )

    def begin_model_request(self, run_id: str) -> bool:
        with self._transaction():
            run = self.get_run(run_id)
            if run["status"] != "running" or self.next_pending(run_id) is not None:
                raise StoreError("нельзя начать запрос к модели в текущем состоянии")
            if run["model_calls"] >= run["max_steps"]:
                self._budget_stop(run_id, "обращений к модели")
                return False
            self.db.execute(
                "UPDATE runs SET model_calls = model_calls + 1, model_started = 1 WHERE run_id = ?", (run_id,),
            )
            return True

    def _admit_batch(self, run_id: str, calls: Sequence[ToolCall]) -> int | None:
        budget = self.get_budget(run_id)
        count = len(calls)
        effects = sum(call.name in SIDE_EFFECT_TOOLS for call in calls)
        for kind, needed, label in (
            ("tool_calls", count, "вызовов инструментов"),
            ("side_effects", effects, "действий с побочным эффектом"),
        ):
            if budget[kind + "_used"] + needed > budget[kind + "_limit"]:
                self._budget_stop(run_id, label)
                return None
        available = budget["result_bytes_limit"] - budget["result_bytes_used"] - budget["result_bytes_reserved"]
        allowance = min(MAX_RESULT_BYTES, available // count)
        if allowance < MIN_RESULT_BYTES:
            self._budget_stop(run_id, "результатов инструментов")
            return None
        self.db.execute(
            "UPDATE budgets SET tool_calls_used = tool_calls_used + ?, side_effects_used = side_effects_used + ?, "
            "result_bytes_reserved = result_bytes_reserved + ? WHERE run_id = ?",
            (count, effects, allowance * count, run_id),
        )
        return allowance

    def _settle_result(self, run_id: str, position: int, result: ToolResult) -> ToolResult:
        pending = self.db.execute(
            "SELECT * FROM pending_calls WHERE run_id = ? AND position = ?", (run_id, position),
        ).fetchone()
        if pending is None or pending["delivered"]:
            raise StoreError("результат уже сохранён или вызов не найден")
        limited = limit_result(result, pending["result_limit"])
        if not pending["result_charged"]:
            self.db.execute(
                "UPDATE budgets SET result_bytes_reserved = result_bytes_reserved - ?, "
                "result_bytes_used = result_bytes_used + ? WHERE run_id = ?",
                (pending["result_limit"], result_size(limited), run_id),
            )
            self.db.execute(
                "UPDATE pending_calls SET result_charged = 1 WHERE run_id = ? AND position = ?", (run_id, position),
            )
        return limited

    def _charge_uncertain_result(self, run_id: str, call_id: str) -> None:
        pending = self.db.execute(
            "SELECT position, result_limit FROM pending_calls WHERE run_id = ? AND call_id = ? "
            "AND delivered = 0 AND result_charged = 0", (run_id, call_id),
        ).fetchall()
        for row in pending:
            self.db.execute(
                "UPDATE budgets SET result_bytes_reserved = result_bytes_reserved - ?, "
                "result_bytes_used = result_bytes_used + ? WHERE run_id = ?",
                (row["result_limit"], row["result_limit"], run_id),
            )
            self.db.execute(
                "UPDATE pending_calls SET result_charged = 1 WHERE run_id = ? AND position = ?",
                (run_id, row["position"]),
            )

    @staticmethod
    def _json(value: Any) -> str:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
        )

    def _append_message(
        self,
        run_id: str,
        message: dict,
    ) -> None:
        self.db.execute(
            """
            INSERT INTO messages (run_id, payload)
            VALUES (?, ?)
            """,
            (run_id, self._json(message)),
        )

    def create_run(
        self,
        run_id: str,
        prompt: str,
        system_prompt: str,
        max_steps: int,
        mode: str = "auto",
        phase: str = "execution",
        budgets: BudgetLimits | None = None,
    ) -> None:
        if mode not in ("safe", "auto"):
            raise StoreError(
                "неизвестный режим выполнения"
            )

        if phase not in ("planning", "execution"):
            raise StoreError("неизвестная фаза запуска")

        with self._transaction():
            self.db.execute(
                """
                INSERT INTO runs (
                    run_id, status, max_steps, mode, phase
                )
                VALUES (?, 'running', ?, ?, ?)
                """,
                (run_id, max_steps, mode, phase),
            )

            self._create_budget(run_id, budgets or BudgetLimits())

            self._append_message(
                run_id,
                {
                    "role": "system",
                    "content": system_prompt,
                },
            )

            self._append_message(
                run_id,
                {
                    "role": "user",
                    "content": prompt,
                },
            )

    def get_run(self, run_id: str) -> dict:
        row = self.db.execute(
            """
            SELECT *
            FROM runs
            WHERE run_id = ?
            """,
            (run_id,),
        ).fetchone()

        if row is None:
            raise StoreError(
                f"запуск не найден: {run_id}"
            )

        return dict(row)

    def get_messages(self, run_id: str) -> list[dict]:
        self.get_run(run_id)

        rows = self.db.execute(
            """
            SELECT payload
            FROM messages
            WHERE run_id = ?
            ORDER BY seq
            """,
            (run_id,),
        ).fetchall()

        return [
            json.loads(row["payload"])
            for row in rows
        ]

    def record_response(
        self,
        run_id: str,
        response: LLMResponse,
        retry_message: str | None = None,
        error: str | None = None,
    ) -> None:
        if (
            (retry_message is not None and error is not None)
            or (
                response.tool_calls
                and (retry_message is not None or error is not None)
            )
        ):
            raise StoreError("некорректное решение по ответу модели")

        with self._transaction():
            run = self.get_run(run_id)

            if run["status"] != "running":
                raise StoreError(
                    "нельзя сохранить ответ для остановленного запуска"
                )

            unfinished = self.db.execute(
                """
                SELECT 1
                FROM pending_calls
                WHERE run_id = ?
                  AND delivered = 0
                LIMIT 1
                """,
                (run_id,),
            ).fetchone()

            if unfinished is not None:
                raise StoreError(
                    "предыдущие вызовы инструментов ещё не завершены"
                )

            self.db.execute(
                """
                DELETE FROM pending_calls
                WHERE run_id = ?
                """,
                (run_id,),
            )

            self._append_message(
                run_id,
                response.to_assistant_message(),
            )

            if not run["model_started"]:
                raise StoreError("запрос к модели не был списан из бюджета")
            self.db.execute("UPDATE runs SET model_started = 0 WHERE run_id = ?", (run_id,))

            if not response.tool_calls:
                if retry_message is not None:
                    self._append_message(
                        run_id,
                        {"role": "system", "content": retry_message},
                    )
                    return

                if error is None and self.tasks.pending(run_id):
                    raise StoreError("нельзя завершить запуск с незавершёнными задачами")
                if error is not None:
                    self.tasks.fail_current(run_id)
                else:
                    self.tasks.finish_implicit(run_id)

                self.db.execute(
                    """
                    UPDATE runs
                    SET status = ?,
                        final_answer = ?
                    WHERE run_id = ?
                    """,
                    (
                        "failed" if error is not None else "completed",
                        error if error is not None else response.content,
                        run_id,
                    ),
                )

                return

            allowance = self._admit_batch(run_id, response.tool_calls)
            if allowance is None:
                return

            task_id = None
            if run["phase"] == "execution":
                if not self.tasks.get_all(run_id):
                    self.tasks.ensure_implicit(run_id)
                task = self.tasks.current(run_id)
                if task is not None:
                    task_id = task["task_id"]

            for position, call in enumerate(
                response.tool_calls
            ):
                arguments = self._json(call.arguments)

                existing = self.db.execute(
                    """
                    SELECT *
                    FROM tool_calls
                    WHERE run_id = ? AND call_id = ?
                    """,
                    (run_id, call.id),
                ).fetchone()

                override = None

                if existing is None:
                    self.db.execute(
                        """
                        INSERT INTO tool_calls (
                            run_id,
                            call_id,
                            name,
                            arguments,
                            status,
                            task_id
                        )
                        VALUES (?, ?, ?, ?, 'queued', ?)
                        """,
                        (
                            run_id,
                            call.id,
                            call.name,
                            arguments,
                            task_id,
                        ),
                    )

                elif (
                    existing["name"] != call.name
                    or existing["arguments"] != arguments
                    or existing["task_id"] != task_id
                ):
                    error = ToolResult(
                        tool_call_id=call.id,
                        name=call.name,
                        success=False,
                        error=(
                            "ошибка протокола: повторный tool_call_id "
                            "с другим именем, аргументами или задачей"
                        ),
                    )

                    override = self._json(
                        asdict(error)
                    )

                self.db.execute(
                    """
                    INSERT INTO pending_calls (
                        run_id,
                        position,
                        call_id,
                        override_result,
                        result_limit,
                        task_id
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        position,
                        call.id,
                        override,
                        allowance,
                        task_id,
                    ),
                )

    def next_pending(
        self,
        run_id: str,
    ) -> dict | None:
        row = self.db.execute(
            """
            SELECT
                p.position,
                p.call_id,
                p.override_result,
                p.result_limit,
                p.task_id,
                c.name,
                c.arguments,
                c.status,
                c.result
            FROM pending_calls p
            JOIN tool_calls c
              ON c.run_id = p.run_id
             AND c.call_id = p.call_id
            WHERE p.run_id = ?
              AND p.delivered = 0
            ORDER BY p.position
            LIMIT 1
            """,
            (run_id,),
        ).fetchone()

        return dict(row) if row else None

    def begin_attempt(
        self,
        run_id: str,
        call_id: str,
    ) -> None:
        with self._transaction():
            run = self.get_run(run_id)
            pending = self.next_pending(run_id)
            if run["phase"] == "execution":
                if pending is None or pending["call_id"] != call_id:
                    raise StoreError("вызов не относится к текущей задаче")
                error = self.tasks.check_call(run_id, pending["task_id"], pending["name"])
                if error:
                    raise StoreError(error)
            row = self.db.execute(
                """
                SELECT status, attempt_count
                FROM tool_calls
                WHERE run_id = ? AND call_id = ?
                """,
                (run_id, call_id),
            ).fetchone()

            if row is None or row["status"] != "queued":
                raise StoreError(
                    "невозможно начать выполнение инструмента"
                )

            attempt_no = row["attempt_count"] + 1

            self.db.execute(
                """
                UPDATE tool_calls
                SET status = 'started',
                    attempt_count = ?
                WHERE run_id = ? AND call_id = ?
                """,
                (attempt_no, run_id, call_id),
            )

            self.db.execute(
                """
                INSERT INTO tool_attempts (
                    run_id,
                    call_id,
                    attempt_no,
                    status
                )
                VALUES (?, ?, ?, 'started')
                """,
                (run_id, call_id, attempt_no),
            )

            self.db.execute(
                """
                UPDATE runs
                SET tool_calls = tool_calls + 1
                WHERE run_id = ?
                """,
                (run_id,),
            )

    def batch_size(self, run_id: str) -> int:
        return self.db.execute(
            "SELECT COUNT(*) FROM pending_calls WHERE run_id = ?", (run_id,),
        ).fetchone()[0]

    def get_plan(self, run_id: str) -> dict | None:
        plan = self.get_run(run_id)["plan"]
        return json.loads(plan) if plan is not None else None

    def decide_plan(self, run_id: str, action: str) -> None:
        if action not in ("approve", "reject"):
            raise StoreError("неизвестное решение по плану")

        with self._transaction():
            run = self.get_run(run_id)
            if run["status"] != "waiting_plan" or run["phase"] != "planning":
                raise StoreError("запуск не ожидает утверждения плана")
            if run["plan"] is None or run["plan_call_id"] is None:
                raise StoreError("сохранённый план не найден")

            self.db.execute(
                "INSERT INTO decisions (run_id, call_id, action) VALUES (?, ?, ?)",
                (run_id, run["plan_call_id"], action + "_plan"),
            )
            if action == "approve":
                self.tasks.create(run_id, json.loads(run["plan"])["steps"])
                self.db.execute(
                    "UPDATE runs SET phase = 'execution', status = 'running' WHERE run_id = ?",
                    (run_id,),
                )
                message = (
                    "пользователь утвердил план, режим планирования завершён\n"
                    "теперь выполняй утверждённый план, изменяй файлы и проверяй результат через Bash\n"
                    "утверждение плана не отменяет policy и подтверждения отдельных действий"
                )
            else:
                self.db.execute(
                    "UPDATE runs SET status = 'plan_rejected' WHERE run_id = ?", (run_id,),
                )
                message = "пользователь отклонил план, исполнение не начато"

            self._append_message(run_id, {"role": "system", "content": message})

    def deliver_result(
        self,
        run_id: str,
        position: int,
        result: ToolResult,
        executed: bool,
    ) -> ToolResult:
        plan = (result.data or {}).get("plan")
        completed_task = result.data if result.name == "complete_task" and result.success else None
        with self._transaction():
            result = self._settle_result(run_id, position, result)
            encoded = self._json(asdict(result))
            pending = self.db.execute(
                """
                SELECT call_id, delivered
                FROM pending_calls
                WHERE run_id = ? AND position = ?
                """,
                (run_id, position),
            ).fetchone()

            if (
                pending is None
                or pending["delivered"]
                or pending["call_id"] != result.tool_call_id
            ):
                raise StoreError(
                    "некорректное состояние вызова инструмента"
                )

            if executed:
                row = self.db.execute(
                    """
                    SELECT status, attempt_count, name
                    FROM tool_calls
                    WHERE run_id = ? AND call_id = ?
                    """,
                    (run_id, result.tool_call_id),
                ).fetchone()

                if row is None or row["status"] != "started":
                    raise StoreError(
                        "результат не соответствует выполняемому действию"
                    )

                if completed_task is not None:
                    self.tasks.finish(run_id, completed_task["task_id"], completed_task["evidence"])

                if row["name"] == "submit_plan" and result.success:
                    run = self.get_run(run_id)
                    if (
                        run["phase"] != "planning"
                        or self.batch_size(run_id) != 1
                        or not Draft202012Validator(PLAN_SCHEMA).is_valid(plan)
                    ):
                        raise StoreError("нельзя сохранить план в текущем состоянии")
                    self.db.execute(
                        "UPDATE runs SET plan = ?, plan_call_id = ?, status = 'waiting_plan' WHERE run_id = ?",
                        (self._json(plan), result.tool_call_id, run_id),
                    )

                self.db.execute(
                    """
                    UPDATE tool_calls
                    SET status = 'completed',
                        result = ?
                    WHERE run_id = ? AND call_id = ?
                    """,
                    (
                        encoded,
                        run_id,
                        result.tool_call_id,
                    ),
                )

                self.db.execute(
                    """
                    UPDATE tool_attempts
                    SET status = 'completed',
                        result = ?
                    WHERE run_id = ?
                      AND call_id = ?
                      AND attempt_no = ?
                    """,
                    (
                        encoded,
                        run_id,
                        result.tool_call_id,
                        row["attempt_count"],
                    ),
                )

            self.db.execute(
                """
                UPDATE pending_calls
                SET delivered = 1
                WHERE run_id = ? AND position = ?
                """,
                (run_id, position),
            )

            self._append_message(
                run_id,
                result.to_message(),
            )

        return result

    def complete_without_execution(
        self,
        run_id: str,
        position: int,
        result: ToolResult,
    ) -> ToolResult:
        with self._transaction():
            result = self._settle_result(run_id, position, result)
            encoded = self._json(asdict(result))
            pending = self.next_pending(run_id)

            if (
                pending is None
                or pending["position"] != position
                or pending["call_id"] != result.tool_call_id
                or pending["status"] != "queued"
            ):
                raise StoreError(
                    "некорректное состояние вызова инструмента"
                )

            self.db.execute(
                """
                UPDATE tool_calls
                SET status = 'completed',
                    result = ?
                WHERE run_id = ? AND call_id = ?
                """,
                (
                    encoded,
                    run_id,
                    result.tool_call_id,
                ),
            )

            self.db.execute(
                """
                UPDATE pending_calls
                SET delivered = 1
                WHERE run_id = ?
                  AND position = ?
                """,
                (run_id, position),
            )

            self.db.execute(
                """
                UPDATE runs
                SET tool_calls = tool_calls + 1
                WHERE run_id = ?
                """,
                (run_id,),
            )

            self._append_message(
                run_id,
                result.to_message(),
            )

        return result

    def get_approval(
        self,
        run_id: str,
        call_id: str,
    ) -> str | None:
        row = self.db.execute(
            """
            SELECT status
            FROM approvals
            WHERE run_id = ? AND call_id = ?
            """,
            (run_id, call_id),
        ).fetchone()

        return row["status"] if row else None

    def get_pending_approval(
        self,
        run_id: str,
    ) -> dict | None:
        row = self.db.execute(
            """
            SELECT
                a.call_id,
                c.name,
                c.arguments
            FROM approvals a
            JOIN tool_calls c
              ON c.run_id = a.run_id
             AND c.call_id = a.call_id
            WHERE a.run_id = ?
              AND a.status = 'pending'
            LIMIT 1
            """,
            (run_id,),
        ).fetchone()

        return dict(row) if row else None

    def request_approval(
        self,
        run_id: str,
        call_id: str,
    ) -> None:
        with self._transaction():
            run = self.get_run(run_id)

            if run["status"] != "running":
                raise StoreError(
                    "запуск не находится в состоянии выполнения"
                )

            pending = self.next_pending(run_id)

            if pending is None or pending["call_id"] != call_id:
                raise StoreError(
                    "ожидающий вызов инструмента не найден"
                )

            if pending["status"] != "queued":
                raise StoreError(
                    "инструмент уже начал выполнение"
                )

            approval = self.get_approval(
                run_id,
                call_id,
            )

            if approval is not None:
                raise StoreError(
                    "подтверждение уже запрашивалось"
                )

            self.db.execute(
                """
                INSERT INTO approvals (
                    run_id,
                    call_id,
                    status
                )
                VALUES (?, ?, 'pending')
                """,
                (run_id, call_id),
            )

            self.db.execute(
                """
                UPDATE runs
                SET status = 'waiting_approval'
                WHERE run_id = ?
                """,
                (run_id,),
            )

    def decide_approval(
        self,
        run_id: str,
        action: str,
    ) -> ToolResult | None:
        if action not in ("approve", "reject"):
            raise StoreError(
                "неизвестное решение пользователя"
            )

        with self._transaction():
            run = self.get_run(run_id)

            if run["status"] != "waiting_approval":
                raise StoreError(
                    "запуск не ожидает подтверждения действия"
                )

            approval = self.get_pending_approval(run_id)

            if approval is None:
                raise StoreError(
                    "ожидающее подтверждение не найдено"
                )

            call_id = approval["call_id"]

            pending = self.next_pending(run_id)

            if (
                pending is None
                or pending["call_id"] != call_id
                or pending["status"] != "queued"
            ):
                raise StoreError(
                    "некорректное состояние подтверждения"
                )

            new_status = (
                "approved"
                if action == "approve"
                else "rejected"
            )

            self.db.execute(
                """
                UPDATE approvals
                SET status = ?
                WHERE run_id = ? AND call_id = ?
                """,
                (
                    new_status,
                    run_id,
                    call_id,
                ),
            )

            self.db.execute(
                """
                INSERT INTO decisions (
                    run_id,
                    call_id,
                    action
                )
                VALUES (?, ?, ?)
                """,
                (
                    run_id,
                    call_id,
                    action,
                ),
            )

            result = None

            if action == "reject":
                result = ToolResult(
                    tool_call_id=call_id,
                    name=approval["name"],
                    success=False,
                    error=(
                        "пользователь отклонил выполнение инструмента"
                    ),
                )

                result = self._settle_result(run_id, pending["position"], result)
                encoded = self._json(asdict(result))

                self.db.execute(
                    """
                    UPDATE tool_calls
                    SET status = 'completed',
                        result = ?
                    WHERE run_id = ? AND call_id = ?
                    """,
                    (
                        encoded,
                        run_id,
                        call_id,
                    ),
                )

                self.db.execute(
                    """
                    UPDATE pending_calls
                    SET delivered = 1
                    WHERE run_id = ?
                      AND position = ?
                    """,
                    (
                        run_id,
                        pending["position"],
                    ),
                )

                self.db.execute(
                    """
                    UPDATE runs
                    SET tool_calls = tool_calls + 1
                    WHERE run_id = ?
                    """,
                    (run_id,),
                )

                self._append_message(
                    run_id,
                    result.to_message(),
                )

            self.db.execute(
                """
                UPDATE runs
                SET status = 'running'
                WHERE run_id = ?
                """,
                (run_id,),
            )

            return result

    def mark_uncertain(
        self,
        run_id: str,
    ) -> bool:
        with self._transaction():
            rows = self.db.execute(
                """
                SELECT call_id, attempt_count
                FROM tool_calls
                WHERE run_id = ?
                  AND status = 'started'
                """,
                (run_id,),
            ).fetchall()

            for row in rows:
                self._charge_uncertain_result(run_id, row["call_id"])
                self.db.execute(
                    """
                    UPDATE tool_calls
                    SET status = 'uncertain'
                    WHERE run_id = ? AND call_id = ?
                    """,
                    (run_id, row["call_id"]),
                )

                self.db.execute(
                    """
                    UPDATE tool_attempts
                    SET status = 'uncertain'
                    WHERE run_id = ?
                      AND call_id = ?
                      AND attempt_no = ?
                    """,
                    (
                        run_id,
                        row["call_id"],
                        row["attempt_count"],
                    ),
                )

            if rows:
                self.db.execute(
                    """
                    UPDATE runs
                    SET status = 'waiting_resolution'
                    WHERE run_id = ?
                    """,
                    (run_id,),
                )

            return bool(rows)

    def get_uncertain(
        self,
        run_id: str,
    ) -> dict | None:
        row = self.db.execute(
            """
            SELECT
                call_id,
                name,
                arguments,
                attempt_count
            FROM tool_calls
            WHERE run_id = ?
              AND status = 'uncertain'
            ORDER BY rowid
            LIMIT 1
            """,
            (run_id,),
        ).fetchone()

        return dict(row) if row else None

    def resolve(
        self,
        run_id: str,
        action: str,
    ) -> None:
        if action not in ("retry", "skip", "abort"):
            raise StoreError(
                "неизвестное решение пользователя"
            )

        with self._transaction():
            run = self.get_run(run_id)

            if run["status"] != "waiting_resolution":
                raise StoreError(
                    "запуск не ожидает решения пользователя"
                )

            call = self.get_uncertain(run_id)

            if call is None:
                raise StoreError(
                    "неопределённый вызов не найден"
                )

            call_id = call["call_id"]

            if action == "retry" and call["name"] not in (
                "read_file",
                "list_files",
                "submit_plan",
                "complete_task",
            ):
                raise StoreError(
                    "повторение этого инструмента небезопасно"
                )

            self.db.execute(
                """
                INSERT INTO decisions (
                    run_id,
                    call_id,
                    action
                )
                VALUES (?, ?, ?)
                """,
                (run_id, call_id, action),
            )

            if action == "retry":
                allowance = self._admit_batch(run_id, [ToolCall(call_id, call["name"], {})])
                if allowance is None:
                    return
                self.db.execute(
                    "UPDATE pending_calls SET result_limit = ?, result_charged = 0 "
                    "WHERE run_id = ? AND call_id = ? AND delivered = 0",
                    (allowance, run_id, call_id),
                )
                self.db.execute(
                    """
                    UPDATE tool_calls
                    SET status = 'queued'
                    WHERE run_id = ? AND call_id = ?
                    """,
                    (run_id, call_id),
                )

                self.db.execute(
                    """
                    UPDATE runs
                    SET status = 'running'
                    WHERE run_id = ?
                    """,
                    (run_id,),
                )

            elif action == "skip":
                result = ToolResult(
                    tool_call_id=call_id,
                    name=call["name"],
                    success=False,
                    error=(
                        "действие не подтверждено после сбоя, "
                        "пользователь выбрал пропустить вызов"
                    ),
                )

                pending_result = self.next_pending(run_id)
                result = self._settle_result(run_id, pending_result["position"], result)
                encoded = self._json(asdict(result))

                self.db.execute(
                    """
                    UPDATE tool_calls
                    SET status = 'completed',
                        result = ?
                    WHERE run_id = ? AND call_id = ?
                    """,
                    (encoded, run_id, call_id),
                )

                pending = self.db.execute(
                    """
                    SELECT position
                    FROM pending_calls
                    WHERE run_id = ?
                      AND call_id = ?
                      AND delivered = 0
                    ORDER BY position
                    LIMIT 1
                    """,
                    (run_id, call_id),
                ).fetchone()

                if pending is None:
                    raise StoreError(
                        "ожидающий вызов не найден"
                    )

                self.db.execute(
                    """
                    UPDATE pending_calls
                    SET delivered = 1
                    WHERE run_id = ? AND position = ?
                    """,
                    (run_id, pending["position"]),
                )

                self._append_message(
                    run_id,
                    result.to_message(),
                )

                self.db.execute(
                    """
                    UPDATE runs
                    SET status = 'running'
                    WHERE run_id = ?
                    """,
                    (run_id,),
                )

            else:
                self.tasks.fail_current(run_id)
                self.db.execute(
                    """
                    UPDATE runs
                    SET status = 'failed'
                    WHERE run_id = ?
                    """,
                    (run_id,),
                )

    def mark_limit(self, run_id: str) -> None:
        with self._transaction():
            self.tasks.fail_current(run_id)
            self.db.execute(
                """
                UPDATE runs
                SET status = 'limit_reached'
                WHERE run_id = ? AND status = 'running'
                """,
                (run_id,),
            )

    def get_tool_call(
        self,
        run_id: str,
        call_id: str,
    ) -> dict | None:
        row = self.db.execute(
            """
            SELECT *
            FROM tool_calls
            WHERE run_id = ? AND call_id = ?
            """,
            (run_id, call_id),
        ).fetchone()

        return dict(row) if row else None

    def get_attempts(
        self,
        run_id: str,
        call_id: str,
    ) -> list[dict]:
        rows = self.db.execute(
            """
            SELECT *
            FROM tool_attempts
            WHERE run_id = ? AND call_id = ?
            ORDER BY attempt_no
            """,
            (run_id, call_id),
        ).fetchall()

        return [
            dict(row)
            for row in rows
        ]

    def get_decisions(
        self,
        run_id: str,
    ) -> list[dict]:
        rows = self.db.execute(
            """
            SELECT *
            FROM decisions
            WHERE run_id = ?
            ORDER BY id
            """,
            (run_id,),
        ).fetchall()

        return [
            dict(row)
            for row in rows
        ]

    def close(self) -> None:
        self.db.close()
