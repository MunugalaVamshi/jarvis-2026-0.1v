"""Single supported language model: Ollama on the local machine only."""
from __future__ import annotations
import json
from typing import AsyncIterator
import httpx
from .base import DEFAULT_CAPS, FunctionCall

DEFAULT_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "llama3.2"

class OllamaLLM:
    def __init__(self, cfg: dict | None = None):
        cfg = cfg or {}
        self.base_url = str(cfg.get("base_url") or DEFAULT_URL).strip().rstrip("/")
        if self.base_url.endswith("/v1"):
            self.base_url = self.base_url[:-3]
        self.model = str(cfg.get("model") or DEFAULT_MODEL).strip()
        self.timeout = float(cfg.get("timeout") or 180)
        self.caps = dict(DEFAULT_CAPS)
        self.caps.update({"tools": True, "tool_format": "openai", "vision": False,
                          "max_tools": 12, "max_tool_chars": 12000, "local_only": True})

    async def chat(self, messages: list[dict], tools: list[dict] | None = None,
                   *, stream: bool = True) -> AsyncIterator[dict]:
        payload = {
            "model": self.model, "messages": messages, "stream": bool(stream),
            "keep_alive": "10m",
        }
        if tools:
            payload["tools"] = tools
        # Cold local CPU inference may take minutes before its first token.
        # Bound connection/upload time, but never abort a healthy stream for
        # waiting on the first local token.
        timeout = httpx.Timeout(connect=5, read=None, write=60, pool=10)
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("POST", f"{self.base_url}/api/chat", json=payload) as response:
                response.raise_for_status()
                tool_calls = []
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    message = item.get("message") or {}
                    content = message.get("content") or ""
                    if content:
                        yield {"text": content}
                    for call in message.get("tool_calls") or []:
                        fn = call.get("function") or {}
                        args = fn.get("arguments") or {}
                        if isinstance(args, str):
                            try: args = json.loads(args)
                            except json.JSONDecodeError: args = {}
                        tool_name = str(fn.get("name") or "")
                        tool_calls.append(FunctionCall(id=f"call_{tool_name}", name=tool_name, args=args))
                if tool_calls:
                    yield {"tool_calls": tool_calls}

    async def warmup(self):
        async with httpx.AsyncClient(timeout=httpx.Timeout(connect=3, read=15, write=10, pool=5)) as client:
            response = await client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
