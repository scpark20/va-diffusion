"""Read-only VA-Diffusion dashboard server using the established PRX layout."""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import re
import subprocess
import time
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parent
LIVE_PATTERN = re.compile(r"live_[0-9]{2}\.png\Z")
SAMPLE_PATTERN = re.compile(
    r"epoch_[0-9]{4}_nfe_(1|3|5|10)_temp_(0p0|0p3|0p6)_[0-9]{2}\.png\Z"
)


def read_json(path, default):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return default


def gpu_status():
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        )
        return [
            {
                "index": int(parts[0]),
                "utilization": int(parts[1]),
                "memory_used": int(parts[2]),
                "memory_total": int(parts[3]),
            }
            for line in result.stdout.splitlines()
            if len(parts := [part.strip() for part in line.split(",")]) == 4
        ]
    except (OSError, subprocess.SubprocessError, ValueError):
        return []


def snapshot(run_dir):
    status = read_json(run_dir / "status.json", {"state": "prepared", "step": 0})
    history = read_json(run_dir / "history.json", {"loss": []})
    live_batch = [
        {"url": f"/live-batch/{path.name}?v={path.stat().st_mtime_ns}", "label": f"Live {index + 1}"}
        for index, path in enumerate(sorted((run_dir / "live_batch").glob("live_*.png")))
        if LIVE_PATTERN.fullmatch(path.name)
    ]
    samples = []
    for path in sorted((run_dir / "samples").glob("epoch_*.png")):
        match = SAMPLE_PATTERN.fullmatch(path.name)
        if not match:
            continue
        parts = path.stem.split("_")
        samples.append({
            "url": f"/samples/{path.name}?v={path.stat().st_mtime_ns}",
            "label": f"Epoch {int(parts[1])} · #{int(parts[-1]) + 1}",
            "nfe": int(match.group(1)),
            "temperature": float(match.group(2).replace("p", ".")),
        })
    return {
        "status": status,
        "history": history,
        "live_batch": live_batch,
        "samples": samples,
        "gpus": gpu_status(),
        "server_time": time.time(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            route = urlparse(self.path).path
            if route == "/api/status":
                payload = json.dumps(snapshot(run_dir), allow_nan=False).encode()
                return self.send_payload(payload, "application/json")
            if route in ("/", "/index.html"):
                return self.send_file(ROOT / "index.html")
            if route == "/dashboard.js":
                return self.send_file(ROOT / "dashboard.js")
            if route.startswith("/live-batch/"):
                name = route.removeprefix("/live-batch/")
                if LIVE_PATTERN.fullmatch(name):
                    return self.send_file(run_dir / "live_batch" / name)
            if route.startswith("/samples/"):
                name = route.removeprefix("/samples/")
                if SAMPLE_PATTERN.fullmatch(name):
                    return self.send_file(run_dir / "samples" / name)
            self.send_error(404)

        def send_file(self, path):
            if not path.is_file():
                return self.send_error(404)
            self.send_payload(path.read_bytes(), mimetypes.guess_type(path.name)[0] or "application/octet-stream")

        def send_payload(self, payload, content_type):
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
