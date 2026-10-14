import json


COMPLETE_TASK_SCHEMA = {
    "type": "object",
    "properties": {
        "task_id": {"type": "string", "minLength": 1},
        "evidence": {
            "type": "array", "minItems": 1, "uniqueItems": True,
            "items": {"type": "string", "minLength": 1},
        },
    },
    "required": ["task_id", "evidence"],
    "additionalProperties": False,
}

TASK_RETRY_MESSAGE = (
    "работа по задачам ещё не завершена\n"
    "выполни текущую задачу и вызови complete_task с её task_id и идентификаторами "
    "успешных вызовов инструментов из этой задачи\n"
    "текстовое заявление не подтверждает выполнение"
)


def complete_task(task_id: str, evidence: list[str], result_limit: int | None = None) -> dict:
    return {"task_id": task_id, "evidence": evidence}


class TaskManager:
    def __init__(self, db):
        self.db = db

    def create_schema(self):
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS tasks (
                run_id TEXT NOT NULL REFERENCES runs(run_id),
                task_id TEXT NOT NULL,
                position INTEGER NOT NULL,
                description TEXT NOT NULL,
                dependencies TEXT NOT NULL,
                tools TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                evidence TEXT NOT NULL DEFAULT '[]',
                implicit INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (run_id, task_id)
            );
        """)

    def get_all(self, run_id):
        rows = self.db.execute(
            "SELECT * FROM tasks WHERE run_id = ? ORDER BY position", (run_id,),
        ).fetchall()
        result = []
        for row in rows:
            task = dict(row)
            for name in ("dependencies", "tools", "evidence"):
                task[name] = json.loads(task[name])
            task["expected_result"] = (
                "успешные необрезанные результаты: " + ", ".join(task["tools"])
                if task["tools"] else "хотя бы один успешный необрезанный результат инструмента"
            )
            if "Bash" in task["tools"]:
                task["expected_result"] += "; Bash с кодом 0 без превышения времени"
            result.append(task)
        return result

    def create(self, run_id, steps, implicit=False):
        if self.get_all(run_id):
            return
        for index, step in enumerate(steps, 1):
            self.db.execute(
                "INSERT INTO tasks (run_id, task_id, position, description, dependencies, tools, implicit) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (run_id, f"task_{index}", index, step["description"],
                 json.dumps([f"task_{index - 1}"] if index > 1 else []),
                 json.dumps(step["tools"]), int(implicit)),
            )

    def ensure_implicit(self, run_id):
        self.create(run_id, [{"description": "выполнить запрос и проверить результаты инструментов", "tools": []}], implicit=True)

    def current(self, run_id):
        tasks = self.get_all(run_id)
        completed = {task["task_id"] for task in tasks if task["status"] == "completed"}
        for task in tasks:
            if task["status"] == "running":
                return task
            if task["status"] == "pending" and set(task["dependencies"]) <= completed:
                self.db.execute(
                    "UPDATE tasks SET status = 'running' WHERE run_id = ? AND task_id = ?",
                    (run_id, task["task_id"]),
                )
                task["status"] = "running"
                return task
        return None

    def pending(self, run_id):
        return any(task["status"] != "completed" and not task["implicit"] for task in self.get_all(run_id))

    def check_call(self, run_id, task_id, name):
        tasks = self.get_all(run_id)
        task = next((item for item in tasks if item["task_id"] == task_id), None)
        if task is None or task["status"] != "running":
            return "вызов не относится к текущей задаче"
        completed = {item["task_id"] for item in tasks if item["status"] == "completed"}
        if not set(task["dependencies"]) <= completed:
            return "зависимости задачи ещё не выполнены"
        if not task["implicit"] and name not in self.allowed_tools(run_id, task):
            return "инструмент не относится к текущей задаче, сначала заверши её"
        return None

    def allowed_tools(self, run_id, task):
        allowed = {*task["tools"], "read_file", "list_files", "complete_task"}
        if "Bash" in task["tools"]:
            bash_calls = [call for call in self.calls(run_id, task["task_id"]) if call["name"] == "Bash" and call["attempt_count"]]
            if bash_calls and bash_calls[-1]["status"] == "completed" and not self.successful(bash_calls[-1]):
                allowed.add("write_file")
        return allowed

    @staticmethod
    def successful(call):
        if call["status"] != "completed" or call["attempt_count"] == 0 or not call["result"]:
            return False
        result = json.loads(call["result"])
        data = result.get("data") or {}
        if not result.get("success") or result.get("truncated") or data.get("truncated"):
            return False
        if call["name"] == "Bash":
            return data.get("exit_code") == 0 and not data.get("timed_out")
        return call["name"] != "complete_task"

    def calls(self, run_id, task_id):
        return [dict(row) for row in self.db.execute(
            "SELECT rowid AS sequence, * FROM tool_calls WHERE run_id = ? AND task_id = ? ORDER BY rowid",
            (run_id, task_id),
        ).fetchall()]

    def completion_error(self, run_id, task_id, evidence):
        error = self.check_call(run_id, task_id, "complete_task")
        if error:
            return error
        task = next(task for task in self.get_all(run_id) if task["task_id"] == task_id)
        calls = {call["call_id"]: call for call in self.calls(run_id, task_id)}
        if not evidence or any(call_id not in calls or not self.successful(calls[call_id]) for call_id in evidence):
            return "нужны успешные необрезанные результаты инструментов текущей задачи"
        proved = {calls[call_id]["name"] for call_id in evidence}
        if not set(task["tools"]) <= proved:
            return "нет подтверждений для всех инструментов задачи: " + ", ".join(sorted(set(task["tools"]) - proved))
        for name in task["tools"]:
            latest = next(call for call in reversed(list(calls.values())) if call["name"] == name)
            if latest["call_id"] not in evidence or not self.successful(latest):
                return "подтверждение устарело, нужен последний успешный результат инструмента " + name
        bash = [calls[key] for key in evidence if calls[key]["name"] == "Bash"]
        changes = [call for call in calls.values() if call["name"] in ("write_file", "delete_file") and call["attempt_count"]]
        if bash and changes and max(call["sequence"] for call in bash) < max(call["sequence"] for call in changes):
            return "проверка Bash должна выполняться после последнего изменения"
        return None

    def finish(self, run_id, task_id, evidence):
        error = self.completion_error(run_id, task_id, evidence)
        if error:
            raise ValueError(error)
        self.db.execute(
            "UPDATE tasks SET status = 'completed', evidence = ? WHERE run_id = ? AND task_id = ?",
            (json.dumps(evidence), run_id, task_id),
        )

    def finish_implicit(self, run_id):
        for task in self.get_all(run_id):
            if task["implicit"] and task["status"] != "completed":
                evidence = [call["call_id"] for call in self.calls(run_id, task["task_id"]) if self.successful(call)]
                self.db.execute(
                    "UPDATE tasks SET status = ?, evidence = ? WHERE run_id = ? AND task_id = ?",
                    ("completed" if evidence else "failed", json.dumps(evidence), run_id, task["task_id"]),
                )

    def fail_current(self, run_id):
        self.db.execute("UPDATE tasks SET status = 'failed' WHERE run_id = ? AND status = 'running'", (run_id,))

    def prompt(self, run_id):
        task = next((task for task in self.get_all(run_id) if task["status"] == "running" and not task["implicit"]), None)
        if task is None:
            return None
        evidence = [
            {"call_id": call["call_id"], "tool": call["name"]}
            for call in self.calls(run_id, task["task_id"]) if self.successful(call)
        ]
        return (
            "выполняй только текущую задачу " + task["task_id"] + ": " + task["description"] + "\n"
            "для завершения нужны успешные результаты каждого инструмента: " + ", ".join(task["tools"]) + "\n"
            "правило о запуске тестов относится к завершению всего запуска, а не каждого шага\n"
            "после выполнения вызови complete_task отдельно от остальных инструментов, передай task_id и evidence\n"
            "evidence — идентификаторы настоящих успешных вызовов из этой задачи, не придумывай их\n"
            "доступные подтверждения: " + json.dumps(evidence, ensure_ascii=False) + "\n"
            "если подтверждений нет, сначала выполни инструменты; при ошибке исправь причину и повтори действие\n"
            "после неудачной проверки Bash разрешено исправить исходник через write_file в этой же задаче "
            "и повторить Bash; тесты менять нельзя\n"
            "следующую задачу начнёт runtime после проверки подтверждений"
        )
