import os
import selectors
import signal
import subprocess
import time

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_my.runtime.policy import Policy


class BashTool:
    def __init__(
        self,
        policy: "Policy",
        timeout_seconds: float = 10.0,
        max_output_bytes: int = 65536,
    ):
        if timeout_seconds <= 0:
            raise ValueError(
                "время выполнения должно быть положительным"
            )

        if max_output_bytes <= 0:
            raise ValueError(
                "лимит вывода должен быть положительным"
            )

        self.policy = policy
        self.timeout_seconds = timeout_seconds
        self.max_output_bytes = max_output_bytes

    @staticmethod
    def _kill_process(process: subprocess.Popen) -> None:
        if process.poll() is not None:
            return

        try:
            os.killpg(process.pid, signal.SIGKILL)
            return
        except (ProcessLookupError, PermissionError):
            pass

        if process.poll() is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass

    def run(self, command: str, cwd: str) -> dict:
        directory = self.policy.check_bash(command, cwd)

        environment = os.environ.copy()
        environment.pop("BASH_ENV", None)
        environment.pop("ENV", None)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"

        try:
            process = subprocess.Popen(
                [
                    "bash",
                    "--noprofile",
                    "--norc",
                    "-c",
                    command,
                ],
                cwd=directory,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
                env=environment,
            )

        except FileNotFoundError as exc:
            raise ValueError(
                "не удалось запустить bash"
            ) from exc

        output = {
            "stdout": bytearray(),
            "stderr": bytearray(),
        }

        total_bytes = 0
        timed_out = False
        truncated = False

        deadline = time.monotonic() + self.timeout_seconds

        try:
            with selectors.DefaultSelector() as selector:
                selector.register(
                    process.stdout,
                    selectors.EVENT_READ,
                    "stdout",
                )

                selector.register(
                    process.stderr,
                    selectors.EVENT_READ,
                    "stderr",
                )

                while selector.get_map():
                    remaining_time = deadline - time.monotonic()

                    if remaining_time <= 0:
                        timed_out = True
                        break

                    events = selector.select(remaining_time)

                    for key, _ in events:
                        chunk = os.read(
                            key.fileobj.fileno(),
                            8192,
                        )

                        if not chunk:
                            selector.unregister(key.fileobj)
                            key.fileobj.close()
                            continue

                        available = (
                            self.max_output_bytes - total_bytes
                        )

                        if len(chunk) > available:
                            output[key.data].extend(
                                chunk[:available]
                            )

                            total_bytes += available
                            truncated = True
                            break

                        output[key.data].extend(chunk)
                        total_bytes += len(chunk)

                    if truncated:
                        break

            if not timed_out and not truncated:
                remaining_time = max(
                    0,
                    deadline - time.monotonic(),
                )

                try:
                    process.wait(timeout=remaining_time)
                except subprocess.TimeoutExpired:
                    timed_out = True

        finally:
            if timed_out or truncated:
                self._kill_process(process)

            process.wait()

            if process.stdout and not process.stdout.closed:
                process.stdout.close()

            if process.stderr and not process.stderr.closed:
                process.stderr.close()

        return {
            "stdout": output["stdout"].decode(
                "utf-8",
                errors="replace",
            ),
            "stderr": output["stderr"].decode(
                "utf-8",
                errors="replace",
            ),
            "exit_code": process.returncode,
            "timed_out": timed_out,
            "truncated": truncated,
        }