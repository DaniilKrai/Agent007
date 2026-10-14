PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "goal": {"type": "string", "pattern": "\\S"},
        "steps": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "description": {"type": "string", "pattern": "\\S"},
                    "tools": {
                        "type": "array",
                        "minItems": 1,
                        "uniqueItems": True,
                        "items": {
                            "type": "string",
                            "enum": ["list_files", "read_file", "write_file", "delete_file", "Bash"],
                        },
                    },
                },
                "required": ["description", "tools"],
                "additionalProperties": False,
            },
        },
        "risks": {"type": "array", "items": {"type": "string", "pattern": "\\S"}},
        "assumptions": {"type": "array", "items": {"type": "string", "pattern": "\\S"}},
        "done_when": {"type": "string", "pattern": "\\S"},
    },
    "required": ["goal", "steps", "risks", "assumptions", "done_when"],
    "additionalProperties": False,
}

PLANNING_PROMPT = (
    "сейчас действует режим планирования\n"
    "изучай проект только через list_files и read_file\n"
    "не изменяй и не удаляй файлы, не вызывай Bash, даже для просмотра или тестов\n"
    "правила об исправлении и запуске тестов начнут действовать после утверждения плана\n"
    "когда изучишь проект, вызови submit_plan отдельным вызовом без других инструментов\n"
    "укажи цель, шаги по порядку с нужными инструментами, риски, допущения и критерий готовности\n"
    "если код уже исправлен, всё равно предложи план проверки через Bash после утверждения\n"
    "после отправки плана дождись явного решения пользователя"
)

PLAN_RETRY_MESSAGE = (
    "теперь верни только JSON-объект плана по заданной схеме, без markdown и вызовов инструментов\n"
    "используй уже прочитанные файлы и результаты инструментов\n"
    "обязательные поля: goal, steps, risks, assumptions, done_when\n"
    "каждый шаг содержит description и непустой список tools\n"
    "если изменений не требуется, запланируй проверку тестов через Bash после утверждения\n"
    "для проверки тестов используй python3 -m unittest -v в директории .\n"
    "не утверждай, что тесты прошли, пока они не запускались\n"
    "этот JSON будет сохранён как план, исполнение начнётся только после решения пользователя"
)


def submit_plan(**plan) -> dict:
    return {"plan": plan}
