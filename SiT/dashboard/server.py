"""Read-only status and PNG endpoints for the VA-Diffusion run."""
import argparse
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import subprocess
from threading import Lock, Thread
import time
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent
SAMPLE = re.compile(r"/samples/(step_[0-9]{8,})/(nfe[135]_t(?:00|03|06)_[0-3]\.png)")
LIVE = re.compile(r"/live/(image[0-3]\.png)")


def read_json(path, default):
    try:
        return json.loads(path.read_bytes())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def recent_history(run):
    try:
        with (run / "history.jsonl").open("rb") as stream:
            size = stream.seek(0, 2)
            stream.seek(max(0, size - 800000))
            if size > 800000:
                stream.readline()
            return [json.loads(line) for line in stream if line.endswith(b"\n")][-2000:]
    except FileNotFoundError:
        return []


class Hardware:
    def __init__(self):
        self.lock = Lock()
        self.cpu, self.gpus, self.history = {}, [], deque(maxlen=201)
        Thread(target=self.collect, daemon=True).start()

    def collect(self):
        last = None
        while True:
            ticks = list(map(int, Path("/proc/stat").read_text().splitlines()[0].split()[1:9]))
            total, idle = sum(ticks), ticks[3] + ticks[4]
            cpu = {"logical_cpus": os.cpu_count(), "load_average": os.getloadavg()[0],
                   "usage_percent": None if last is None else 100 * (1 - (idle - last[1]) / max(1, total - last[0]))}
            last = total, idle
            gpus = []
            try:
                fields = "name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"
                line = subprocess.check_output(["nvidia-smi", "-i", "1", "--query-gpu=" + fields,
                                                "--format=csv,noheader,nounits"], text=True, timeout=2)
                name, util, used, capacity, temp, power = [part.strip() for part in line.split(",")]
                gpus = [dict(index=1, name=name, utilization_percent=float(util), memory_used_mib=float(used),
                             memory_total_mib=float(capacity), temperature_c=float(temp), power_w=float(power))]
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
            with self.lock:
                self.cpu, self.gpus = cpu, gpus
                self.history.append({"timestamp": time.time(), "utilization": {"1": gpus[0]["utilization_percent"]} if gpus else {},
                                     "gap": not bool(gpus)})
            time.sleep(3)

    def snapshot(self):
        with self.lock:
            return {"cpu": self.cpu, "gpus": self.gpus, "gpu_history": {"samples": list(self.history)}}


def serve(run, host, port):
    hardware = Hardware()
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            route = urlsplit(self.path).path
            if route == "/api/status":
                s = read_json(run / "status.json", {"state":"prepared", "step":0})
                steps = sorted(int(f.parent.name[5:]) for f in (run / "samples").glob("step_*/grid.json")
                               if re.fullmatch(r"step_[0-9]{8,}", f.parent.name))
                live = read_json(run / "live" / "batch.json", {})
                version = live.get("step", 0)
                payload = {"server_time": time.time(), "status": s, "steps": steps, "live_step": version,
                           "live": [f"/live/image{i}.png?v={version}" for i in range(live.get("count",0))],
                           "history": recent_history(run), **hardware.snapshot()}
                return self.reply(json.dumps(payload, allow_nan=False).encode(), "application/json")
            if route in ("/", "/index.html", "/dashboard.js"):
                file = ROOT / ("dashboard.js" if route.endswith(".js") else "index.html")
                return self.reply(file.read_bytes(), "application/javascript" if route.endswith(".js") else "text/html; charset=utf-8")
            m = SAMPLE.fullmatch(route)
            if m:
                file = run / "samples" / m[1] / m[2]
            else:
                m = LIVE.fullmatch(route)
                if not m:
                    return self.send_error(404)
                file = run / "live" / m[1]
            try:
                return self.reply(file.read_bytes(), "image/png")
            except FileNotFoundError:
                self.send_error(404)

        def reply(self, body, mime):
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (ConnectionResetError, BrokenPipeError):
                pass

        def log_message(self, *args):
            pass

    ThreadingHTTPServer((host, port), Handler).serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--host", default="192.168.0.67")
    parser.add_argument("--port", type=int, default=8805)
    args = parser.parse_args()
    serve(args.run, args.host, args.port)
