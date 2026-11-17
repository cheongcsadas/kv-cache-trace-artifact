from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from experiments.collection import collect, run_schedule


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        time.sleep(payload.get("delay", 0))
        self.send_response(payload.get("status", 200))
        self.end_headers()
        self.wfile.write(b"{}")

    def log_message(self, *args): return


class CollectionRevisionTests(unittest.TestCase):
    def setUp(self):
        self.server = ThreadingHTTPServer(("127.0.0.1",0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.endpoint = f"http://127.0.0.1:{self.server.server_port}/infer"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def test_http_errors_are_retained_attempts_and_never_follow_redirects(self):
        rows = run_schedule(self.endpoint,
            [dict(sample_id="a",payload=dict(status=503)),dict(sample_id="b",payload=dict(status=302))],
            1,[0,1])
        self.assertEqual([r["status"] for r in rows],["failed","failed"])
        self.assertEqual([r["http_status"] for r in rows],[503,302])
        self.assertTrue(all(r["latency_ms"]>0 for r in rows))

    def test_anchor_is_exported_while_response_is_still_running(self):
        with tempfile.TemporaryDirectory() as tmp:
            anchor = Path(tmp)/"anchor.json"
            result = []
            worker = threading.Thread(target=lambda: result.extend(run_schedule(self.endpoint,
                [dict(sample_id="a",payload=dict(delay=.3))],1,[0],target_id="a",anchor_path=anchor)))
            worker.start()
            deadline = time.monotonic()+2
            while not anchor.exists() and time.monotonic()<deadline: time.sleep(.002)
            self.assertTrue(anchor.exists())
            self.assertTrue(worker.is_alive())
            written = json.loads(anchor.read_text())
            worker.join()
            self.assertEqual(written["submission_monotonic_ms"],result[0]["submission_monotonic_ms"])
            self.assertIn("submission_uncertainty_ms", written)

    def test_metadata_request_and_collection_failures_are_saved(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            manifest = dict(config=str(Path(__file__).resolve().parents[1]/"configs/quest.yaml"),
                channel="meta",endpoint=self.endpoint,target_id="a",concurrency=1,
                requests=[dict(sample_id="a",source_id="source-a",domain="medical",payload=dict(status=503))],
                arrival_offsets_ms=[0],metadata_exports=["empty.jsonl"])
            (base/"empty.jsonl").write_text("",encoding="utf-8")
            (base/"manifest.json").write_text(json.dumps(manifest),encoding="utf-8")
            attempt = collect(base/"manifest.json",base/"out")
            self.assertEqual(attempt["request_status"],"failed")
            self.assertEqual(attempt["acquisition_status"],"failed")
            self.assertNotIn("observation_path",attempt)
            self.assertEqual(len((base/"out/attempt.jsonl").read_text().splitlines()),1)
            self.assertGreaterEqual(attempt["collection_ms"],0)
            self.assertIn("setup_ms",attempt)


if __name__ == "__main__": unittest.main()
