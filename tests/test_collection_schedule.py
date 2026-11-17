"""Exercise real local HTTP concurrency and measured completion bookkeeping."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import time
import pytest
from experiments.collection import run_schedule


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        time.sleep(payload["delay"])
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"done":true}')

    def log_message(self, *args):
        return


def test_real_async_requests_and_target_anchor(tmp_path):
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        requests = [dict(sample_id=str(i), payload=dict(delay=.08)) for i in range(4)]
        result = run_schedule(f"http://127.0.0.1:{server.server_port}/infer", requests, 4,
                              [0, 5, 10, 15], target_id="0", anchor_path=tmp_path/"anchor.json")
        assert all(row["status"] == "completed" for row in result)
        assert max(r["active_at_submission"] for r in result) > 1
        assert result[1]["submission_ms"] < result[0]["completion_ms"]
        assert all(r["latency_ms"] >= 70 for r in result)
        assert not any("payload" in r for r in result)
        anchor = json.loads((tmp_path/"anchor.json").read_text())
        assert anchor["submission_monotonic_ms"] == result[0]["submission_monotonic_ms"]
    finally:
        server.shutdown()
        server.server_close()


def test_remote_endpoint_and_nonpaper_concurrency_rejected():
    with pytest.raises(ValueError): run_schedule("https://example.com", [], 1, [])
    with pytest.raises(ValueError): run_schedule("http://localhost", [dict(sample_id="a",payload={})], 2, [0])
