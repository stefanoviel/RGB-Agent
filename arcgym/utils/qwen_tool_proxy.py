from __future__ import annotations

import json
import socket
import threading
import uuid
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


_TOOL_CALL_START = "<tool_call>"
_TOOL_CALL_END = "</tool_call>"
_DEBUG_LOG = Path("/tmp/qwen_tool_proxy_debug.log")


def _extract_tool_calls(content: str) -> tuple[str | None, list[dict[str, Any]]]:
    text = str(content or "")
    tool_calls: list[dict[str, Any]] = []
    cleaned_parts: list[str] = []
    cursor = 0
    while True:
        start = text.find(_TOOL_CALL_START, cursor)
        if start < 0:
            cleaned_parts.append(text[cursor:])
            break
        cleaned_parts.append(text[cursor:start])
        end = text.find(_TOOL_CALL_END, start)
        if end < 0:
            cleaned_parts.append(text[start:])
            break
        payload = text[start + len(_TOOL_CALL_START):end].strip()
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            cleaned_parts.append(text[start:end + len(_TOOL_CALL_END)])
            cursor = end + len(_TOOL_CALL_END)
            continue

        name = data.get("name")
        arguments = data.get("arguments", {})
        if not isinstance(name, str) or not name:
            cleaned_parts.append(text[start:end + len(_TOOL_CALL_END)])
            cursor = end + len(_TOOL_CALL_END)
            continue

        tool_calls.append(
            {
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments, separators=(",", ":")),
                },
            }
        )
        cursor = end + len(_TOOL_CALL_END)

    cleaned = "".join(cleaned_parts).strip() or None
    return cleaned, tool_calls


def _normalize_chat_completion(payload: dict[str, Any]) -> dict[str, Any]:
    choices = payload.get("choices")
    if not isinstance(choices, list):
        return payload

    changed = False
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if not isinstance(message, dict):
            continue
        existing = message.get("tool_calls")
        if isinstance(existing, list) and existing:
            continue
        content = message.get("content")
        if not isinstance(content, str) or _TOOL_CALL_START not in content:
            continue
        cleaned, tool_calls = _extract_tool_calls(content)
        if not tool_calls:
            continue
        message["content"] = cleaned
        message["tool_calls"] = tool_calls
        choice["finish_reason"] = "tool_calls"
        changed = True
    return payload if changed else payload


def _as_sse_payloads(payload: dict[str, Any]) -> list[str]:
    chunks: list[str] = []
    base = {
        "id": payload.get("id", f"chatcmpl-{uuid.uuid4().hex[:16]}"),
        "object": "chat.completion.chunk",
        "created": payload.get("created"),
        "model": payload.get("model"),
    }
    choices = payload.get("choices") or []
    if not isinstance(choices, list) or not choices:
        return ['data: [DONE]\n\n']

    choice = choices[0] if isinstance(choices[0], dict) else {}
    message = choice.get("message") if isinstance(choice, dict) else {}
    if not isinstance(message, dict):
        message = {}

    chunks.append(
        "data: "
        + json.dumps(
            {
                **base,
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
            },
            separators=(",", ":"),
        )
        + "\n\n"
    )

    content = message.get("content")
    if isinstance(content, str) and content:
        chunks.append(
            "data: "
            + json.dumps(
                {
                    **base,
                    "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
                },
                separators=(",", ":"),
            )
            + "\n\n"
        )

    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        for index, tool_call in enumerate(tool_calls):
            if not isinstance(tool_call, dict):
                continue
            chunks.append(
                "data: "
                + json.dumps(
                    {
                        **base,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": index,
                                            "id": tool_call.get("id", f"call_{uuid.uuid4().hex[:24]}"),
                                            "type": "function",
                                            "function": tool_call.get("function", {}),
                                        }
                                    ]
                                },
                                "finish_reason": None,
                            }
                        ],
                    },
                    separators=(",", ":"),
                )
                + "\n\n"
            )

    finish_reason = choice.get("finish_reason")
    if tool_calls:
        finish_reason = "tool_calls"
    chunks.append(
        "data: "
        + json.dumps(
            {
                **base,
                "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason or "stop"}],
            },
            separators=(",", ":"),
        )
        + "\n\n"
    )
    return chunks


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@dataclass(frozen=True)
class ProxyHandle:
    base_url: str
    server: ThreadingHTTPServer
    thread: threading.Thread

    def close(self) -> None:
        self.server.shutdown()
        self.thread.join(timeout=5)


def start_proxy(*, upstream_base_url: str, api_key: str) -> ProxyHandle:
    upstream = upstream_base_url.rstrip("/")
    auth_value = api_key or "EMPTY"

    class Handler(BaseHTTPRequestHandler):
        def _forward(self) -> None:
            length = int(self.headers.get("Content-Length", "0") or "0")
            body = self.rfile.read(length) if length > 0 else b""
            body_json: dict[str, Any] | None = None
            body_text = body.decode("utf-8", errors="ignore")
            wants_stream = '"stream"' in body_text and "true" in body_text.lower()
            content_type = self.headers.get("Content-Type", "application/json")
            if "application/json" in content_type and body:
                try:
                    body_json = json.loads(body_text)
                    wants_stream = wants_stream or bool(body_json.get("stream"))
                except Exception:
                    body_json = None
            if self.path.endswith("/chat/completions"):
                try:
                    _DEBUG_LOG.write_text(
                        json.dumps(
                            {
                                "path": self.path,
                                "content_type": content_type,
                                "wants_stream": wants_stream,
                                "body_text": body_text[:4000],
                            },
                            indent=2,
                        )
                    )
                except Exception:
                    pass

            upstream_path = self.path
            if upstream.endswith("/v1") and upstream_path.startswith("/v1/"):
                upstream_path = upstream_path[3:]
            if body_json is not None and self.path.endswith("/chat/completions") and wants_stream:
                body_json = dict(body_json)
                body_json["stream"] = False
                body = json.dumps(body_json).encode("utf-8")
            req = Request(
                f"{upstream}{upstream_path}",
                data=body,
                method=self.command,
                headers={
                    "Content-Type": content_type,
                    "Authorization": f"Bearer {auth_value}",
                },
            )
            try:
                with urlopen(req, timeout=300) as resp:
                    raw = resp.read()
                    status = resp.status
                    headers = {str(k).lower(): v for k, v in resp.headers.items()}
            except HTTPError as exc:
                raw = exc.read()
                status = exc.code
                headers = {str(k).lower(): v for k, v in exc.headers.items()}
            except URLError as exc:
                payload = json.dumps({"error": {"message": str(exc)}}).encode("utf-8")
                self.send_response(502)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return

            upstream_content_type = headers.get("content-type", "")
            if self.path.endswith("/chat/completions") and "application/json" in upstream_content_type:
                try:
                    payload = json.loads(raw.decode("utf-8"))
                    payload = _normalize_chat_completion(payload)
                    if wants_stream:
                        sse = "".join(_as_sse_payloads(payload)).encode("utf-8")
                        self.send_response(status)
                        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                        self.send_header("X-Qwen-Proxy", "stream")
                        self.send_header("Cache-Control", "no-cache")
                        self.send_header("Connection", "keep-alive")
                        self.send_header("Content-Length", str(len(sse)))
                        self.end_headers()
                        self.wfile.write(sse)
                        return
                    raw = json.dumps(payload).encode("utf-8")
                except Exception as exc:
                    error_payload = json.dumps({"proxy_error": str(exc)}).encode("utf-8")
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(error_payload)))
                    self.end_headers()
                    self.wfile.write(error_payload)
                    return

            self.send_response(status)
            self.send_header("Content-Type", headers.get("content-type", "application/json"))
            self.send_header("X-Qwen-Proxy", "plain")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:  # noqa: N802
            self._forward()

        def do_POST(self) -> None:  # noqa: N802
            self._forward()

        def log_message(self, format: str, *args: Any) -> None:
            return

    port = _pick_free_port()
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return ProxyHandle(base_url=f"http://127.0.0.1:{port}/v1", server=server, thread=thread)
