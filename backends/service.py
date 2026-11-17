"""Start and stop an isolated native-model Python environment on this host."""
import json
from pathlib import Path
import subprocess
import sys
import time


class NativeService:
    def __init__(self, config_path, source_dir, model_path=None, *, python=sys.executable,
                 device="cuda:0", output, sweep=None, start_timeout_s=1800):
        self.config_path = Path(config_path).resolve()
        self.source_dir = Path(source_dir).resolve()
        self.model_path = model_path
        self.python, self.device = str(python), device
        self.output = Path(output).resolve()
        self.sweep, self.start_timeout_s = sweep, start_timeout_s
        self.process = None

    def __enter__(self):
        self.output.mkdir(parents=True, exist_ok=False)
        ready = self.output / "ready.json"
        command = [self.python, "-m", "backends.server", "--config", str(self.config_path),
                   "--source-dir", str(self.source_dir), "--device", self.device,
                   "--port", "0", "--export-dir", str(self.output / "exports"),
                   "--ready-file", str(ready)]
        if self.model_path:
            command.extend(["--model-path", str(self.model_path)])
        if self.sweep:
            for value in ([self.sweep] if isinstance(self.sweep, str) else self.sweep):
                command.extend(["--sweep", value])
        self._log = (self.output / "backend.log").open("x", encoding="utf-8")
        try:
            self.process = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[1],
                                            stdout=self._log, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + self.start_timeout_s
            while not ready.exists():
                if self.process.poll() is not None:
                    raise RuntimeError(f"Native backend exited; inspect {self.output / 'backend.log'}")
                if time.monotonic() >= deadline:
                    raise TimeoutError("Native backend startup timed out")
                time.sleep(.1)
            self.endpoint = json.loads(ready.read_text(encoding="utf-8"))["endpoint"]
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def describe(self, prompt, max_new_tokens=None):
        from experiments.collection import native_describe
        return native_describe(self.endpoint, dict(prompt=prompt, max_new_tokens=max_new_tokens))

    def __exit__(self, *args):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=15)
        if hasattr(self, "_log"):
            self._log.close()
