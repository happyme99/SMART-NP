"""Small llama.cpp OpenAI-server client. No OpenAI SDK needed."""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path
from urllib import request


THINK = re.compile(r"<think>.*?</think>|<think>.*$", flags=re.I | re.S)
THINKING_BUDGET_TOKENS = 6096
FINAL_ANSWER_TOKENS = 128


def _chat_payload(system: str, user: str, thinking: bool) -> dict[str, object]:
    payload: dict[str, object] = {
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": 0,
        "max_tokens": FINAL_ANSWER_TOKENS + (THINKING_BUDGET_TOKENS if thinking else 0),
        "chat_template_kwargs": {"enable_thinking": thinking},
    }
    if thinking:
        payload["thinking_budget_tokens"] = THINKING_BUDGET_TOKENS
        payload["reasoning_budget_message"] = "Provide the final class label now."
    return payload


class LlamaServer:
    def __init__(self, binary: Path, model: Path, port: int, *, ctx: int = 8191, threads: int = 32, ngl: str = "all", cuda_device: int | None = None):
        self.binary, self.model, self.port = Path(binary), Path(model), int(port)
        self.ctx, self.threads, self.ngl, self.cuda_device, self.process = int(ctx), int(threads), str(ngl), cuda_device, None

    def __enter__(self):
        if not self.binary.is_file() or not self.model.is_file():
            raise FileNotFoundError("Run scripts/build_model.sh first")
        environment = os.environ.copy()
        if self.cuda_device is not None:
            environment["CUDA_VISIBLE_DEVICES"] = str(self.cuda_device)
        environment["LD_LIBRARY_PATH"] = f"{self.binary.parent}:{environment.get('LD_LIBRARY_PATH', '')}".rstrip(":")
        command = [str(self.binary), "-m", str(self.model), "--jinja", "--reasoning", "auto", "-t", str(self.threads), "-ngl", self.ngl, "-c", str(self.ctx), "--host", "127.0.0.1", "--port", str(self.port)]
        self.process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, env=environment)
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"llama-server exited early ({self.process.returncode})")
            if _ready(self.port):
                return self
            time.sleep(1)
        self.__exit__(None, None, None)
        raise TimeoutError("llama-server was not ready after 300 seconds")

    def __exit__(self, *_):
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()

    def chat(self, system: str, user: str, thinking: bool = True) -> tuple[str, str]:
        payload = _chat_payload(system, user, thinking)
        req = request.Request(f"http://127.0.0.1:{self.port}/v1/chat/completions", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST")
        with request.urlopen(req, timeout=600) as response:
            data = json.loads(response.read().decode())
        choice = data["choices"][0]
        return str(choice["message"].get("content", "")).strip(), str(choice.get("finish_reason") or "")


def _ready(port: int) -> bool:
    try:
        with request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as response:
            return response.status == 200
    except OSError:
        return False


def extract_label(output: str, labels: list[str]) -> str:
    text = THINK.sub(" ", str(output)).replace("</think>", " ")
    normalized = [(label, label.casefold()) for label in labels]
    for line in reversed(text.splitlines()):
        candidate = line.strip().strip("`\"' ").rstrip(".")
        candidate = re.sub(r"^(class name|final answer|answer)\s*:\s*", "", candidate, flags=re.I).strip()
        for label, folded in normalized:
            if candidate.casefold() == folded:
                return label
    for label, folded in normalized:
        if re.search(rf"(?<![\w]){re.escape(folded)}(?![\w])", text.casefold()):
            return label
    return "INVALID"
