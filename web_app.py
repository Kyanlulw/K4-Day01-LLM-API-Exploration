"""
Tiny local web UI for comparing two OpenAI-compatible chat models.

Run:
    python web_app.py
"""

from __future__ import annotations

import argparse
import json
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import template


ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "web"


def _read_json(handler: SimpleHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length", "0"))
    raw = handler.rfile.read(length) if length else b"{}"
    return json.loads(raw.decode("utf-8"))


def _write_json(
    handler: SimpleHTTPRequestHandler,
    payload: dict[str, Any],
    status: int = 200,
) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _cost_from_usage(
    prompt: str,
    response_text: str,
    model: str,
    usage: Any | None,
) -> dict[str, Any]:
    pricing = template.PRICING_PER_1K_TOKENS.get(
        model,
        template.PRICING_PER_1K_TOKENS["gpt-4o"],
    )

    if usage and getattr(usage, "prompt_tokens", None) is not None:
        prompt_tokens = int(usage.prompt_tokens)
        completion_tokens = int(usage.completion_tokens or 0)
        prompt_cost = prompt_tokens / 1000 * pricing["input"]
        completion_cost = completion_tokens / 1000 * pricing["output"]
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "prompt_cost": prompt_cost,
            "completion_cost": completion_cost,
            "total_cost": prompt_cost + completion_cost,
        }

    return template.estimate_cost(prompt, response_text, model)


def _complete_once(
    *,
    model: str,
    prompt: str,
    system_prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> dict[str, Any]:
    client = template.make_openai_client()
    messages = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    messages.append({"role": "user", "content": prompt})

    start = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stream=False,
    )
    latency = max(time.perf_counter() - start, 1e-9)
    text = response.choices[0].message.content or ""
    billed_prompt = "\n".join(m["content"] for m in messages)
    cost = _cost_from_usage(billed_prompt, text, model, getattr(response, "usage", None))

    return {
        "model": model,
        "answer": text,
        "latency": latency,
        "cost": cost,
    }


def compare(payload: dict[str, Any]) -> dict[str, Any]:
    prompt = str(payload.get("prompt", "")).strip()
    if not prompt:
        raise ValueError("Prompt is required.")

    system_prompt = str(payload.get("system_prompt", "")).strip()
    temperature = float(payload.get("temperature", 0.7))
    top_p = float(payload.get("top_p", 0.9))
    max_tokens = int(payload.get("max_tokens", 256))
    models = payload.get("models") or [template.OPENAI_MODEL, template.OPENAI_MINI_MODEL]
    if len(models) != 2:
        raise ValueError("Exactly two models are required.")

    temperature = min(max(temperature, 0.0), 2.0)
    top_p = min(max(top_p, 0.0), 1.0)
    max_tokens = min(max(max_tokens, 1), 4096)

    results = [
        _complete_once(
            model=str(model),
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        for model in models
    ]

    totals = [item["cost"]["total_cost"] for item in results]
    latencies = [item["latency"] for item in results]
    return {
        "results": results,
        "winner": {
            "cheaper": results[totals.index(min(totals))]["model"],
            "faster": results[latencies.index(min(latencies))]["model"],
        },
        "settings": {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        },
    }


class WebHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def do_GET(self) -> None:
        if self.path == "/api/config":
            _write_json(
                self,
                {
                    "models": [template.OPENAI_MODEL, template.OPENAI_MINI_MODEL],
                    "pricing_per_1k_tokens": template.PRICING_PER_1K_TOKENS,
                },
            )
            return
        super().do_GET()

    def do_POST(self) -> None:
        if self.path != "/api/compare":
            _write_json(self, {"error": "Not found"}, status=404)
            return
        try:
            _write_json(self, compare(_read_json(self)))
        except Exception as exc:
            _write_json(self, {"error": str(exc)}, status=400)


def run(host: str, port: int) -> None:
    for candidate in range(port, port + 20):
        try:
            server = ThreadingHTTPServer((host, candidate), WebHandler)
            break
        except OSError:
            continue
    else:
        raise OSError(f"No free port found from {port} to {port + 19}.")

    url = f"http://{host}:{server.server_port}"
    print(f"Web UI running at {url}")
    print("Press Ctrl+C to stop.")
    server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    args = parser.parse_args()
    run(args.host, args.port)
