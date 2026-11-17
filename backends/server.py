"""Loopback inference service backed by a loaded native runtime."""
import argparse
import json
from pathlib import Path
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from backends import load_runtime
from experiments.config import load_config
from experiments.paper_protocol import validate_paper_config


def make_server(runtime, host, port, export_dir):
    if host not in {"127.0.0.1", "localhost"}:
        raise ValueError("The controlled inference service binds to loopback")
    root = Path(export_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    seen, guard = set(), threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            # Runtime exports carry request diagnostics; suppress HTTP access logs.
            return None

        def respond(self, status, payload):
            content = json.dumps(payload, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self):
            if self.path != "/health":
                return self.respond(404, {"error": "Unknown endpoint"})
            self.respond(200, dict(status="ready", system=runtime.config["system"],
                                  concurrent_requests=bool(getattr(runtime, "concurrent_requests", False))))

        def do_POST(self):
            started, request_id = None, None
            def failure(status, exc):
                payload = dict(error=type(exc).__name__, detail=str(exc))
                if started is not None:
                    payload.update(getattr(exc, "native_export_receipt", {}))
                    payload.update(request_id=request_id, system=runtime.config["system"],
                                   provenance="measured", execution_status="failed",
                                   execution_start_monotonic_ms=started,
                                   execution_end_monotonic_ms=time.perf_counter() * 1000)
                self.respond(status, payload)

            try:
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < length <= 16 * 1024 * 1024:
                    raise ValueError("Invalid request size")
                body = json.loads(self.rfile.read(length))
                prompt = body["prompt"]
                if not isinstance(prompt, str) or not prompt:
                    raise ValueError("A nonempty prompt is required")
                maximum = body.get("max_new_tokens")
                if self.path == "/describe":
                    description = runtime.describe_request(prompt, max_new_tokens=maximum)
                    description["concurrent_requests"] = bool(getattr(runtime, "concurrent_requests", False))
                    return self.respond(200, description)
                if self.path != "/generate":
                    return self.respond(404, {"error": "Unknown endpoint"})
                request_id = self.headers.get("X-KvTrail-Request-ID", "")
                if not re.fullmatch(r"[A-Za-z0-9_-]{1,160}", request_id):
                    raise ValueError("Request ID must be an opaque filename-safe identifier")
                with guard:
                    if request_id in seen:
                        raise ValueError("Request IDs cannot be reused")
                    seen.add(request_id)
                started = time.perf_counter() * 1000
                receipt = runtime.generate(prompt, request_id, root / request_id,
                                           max_new_tokens=maximum)
                receipt.update(request_id=request_id, provenance="measured",
                               execution_start_monotonic_ms=started,
                               execution_end_monotonic_ms=time.perf_counter() * 1000)
                # Generation is performed; collection receives its receipt and
                # operation exports, without copying prompt/generated text.
                for key in ("prompt", "text", "generated_text", "output_text", "token_ids", "output_token_ids"):
                    receipt.pop(key, None)
                (root / request_id / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
                self.respond(200, receipt)
            except (ValueError, KeyError, TypeError) as exc:
                failure(400, exc)
            except Exception as exc:
                failure(500, exc)

    return ThreadingHTTPServer((host, port), Handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--model-path")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--ready-file")
    parser.add_argument("--sweep", action="append")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    validate_paper_config(config, allowed_sweep=args.sweep)
    runtime = load_runtime(config, args.source_dir, args.model_path, args.device)
    server = make_server(runtime, args.host, args.port, args.export_dir)
    if args.ready_file:
        destination = Path(args.ready_file)
        if destination.exists():
            raise FileExistsError(destination)
        pending = destination.with_suffix(destination.suffix + ".pending")
        with pending.open("x", encoding="utf-8") as stream:
            json.dump(dict(endpoint=f"http://{args.host}:{server.server_port}/generate",
                           system=config["system"]), stream)
        pending.replace(destination)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        if hasattr(runtime, "close"):
            runtime.close()


if __name__ == "__main__":
    main()
