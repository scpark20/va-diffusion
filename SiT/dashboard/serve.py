"""Read-only, allowlisted VA-Diffusion dashboard on server 67."""
import argparse
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import re
import subprocess
import threading
import time
from urllib.parse import urlsplit

from dashboard.prx import build
from experiment import RUN, UNIT, BATCH, LR, STEPS, TEMPS, CLASSES, beta, read_json

HERE = Path(__file__).resolve().parent
IMAGE = re.compile(r"/(?:live/[0-9]+/[0-3]\.png|samples/[0-9]+/n[135]_t(?:00|03|06)_[0-3]\.png)")


def actual_phase(status, active, pid):
    if active == "deactivating":
        return "stopping"
    if active in ("active","activating") and pid > 0:
        if status.get("pid") == pid:
            return status.get("phase","starting")
        return "starting"
    return "failed" if active == "failed" or status.get("phase") == "failed" else "stopped"


def telemetry():
    fields = "name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"
    data = subprocess.check_output(["nvidia-smi", "-i","1","--query-gpu="+fields,
                                    "--format=csv,noheader,nounits"],text=True,timeout=2)
    name, util, used, total, temp, power = [x.strip() for x in data.split(",")]
    return [dict(index=1,name=name,utilization_percent=float(util),memory_used_mib=float(used),
                 memory_total_mib=float(total),temperature_c=float(temp),power_w=float(power))]


def progress_check(run, active, pid, previous):
    status = read_json(run / "status.json", {})
    phase = actual_phase(status, active, pid)
    step = status.get("step", 0)
    issues = []
    if phase == "training":
        if previous is not None and previous == (pid, step):
            issues.append("step_stalled")
        m = status.get("latest", {})
        if not all(math.isfinite(m.get(k, math.nan)) for k in ("step", "e4", "e8", "e16", "kl", "loss", "beta", "data_mse")):
            issues.append("missing_or_nonfinite_metric")
        else:
            if not math.isclose(m["kl"], m["e4"] + m["e8"] + m["e16"], rel_tol=2e-6, abs_tol=1e-7):
                issues.append("kl_sum")
            if m["beta"] != beta(m["step"] - 1):
                issues.append("beta_schedule")
            if not math.isclose(m["loss"], m["data_mse"] + m["beta"] * m["kl"], rel_tol=2e-6, abs_tol=1e-7):
                issues.append("loss_sum")
        due = status.get("next_checkpoint_at")
        if due is not None and time.time() > due + 60:
            issues.append("checkpoint_overdue")
    saved = status.get("last_checkpoint_step")
    if saved is not None and phase == "training" and not (run / "samples" / str(saved) / "manifest.json").is_file():
        issues.append("saved_samples_missing")
    return {"phase": phase, "step": step, "saved": saved, "issues": issues}


class State:
    def __init__(self, run):
        self.run = run
        self.lock = threading.Lock()
        self.samples = deque(maxlen=200)
        self.cpu, self.gpus, self.active, self.pid = {}, [], "unknown", 0
        threading.Thread(target=self.poll, daemon=True).start()

    def poll(self):
        previous = None
        previous_progress = None
        next_check = time.monotonic()
        while True:
            tick = list(map(int,Path("/proc/stat").read_text().splitlines()[0].split()[1:9]))
            total, idle = sum(tick), tick[3]+tick[4]
            util = None if previous is None else 100*(1-(idle-previous[1])/max(total-previous[0],1))
            previous = total, idle
            cpu = dict(logical_cpus=os.cpu_count(),load_average=os.getloadavg()[0],usage_percent=util)
            try:
                gpus = telemetry()
            except (OSError,ValueError,subprocess.SubprocessError):
                gpus = []
            try:
                result = subprocess.run(["systemctl","--user","show",UNIT,"-p","ActiveState","-p","MainPID"],
                                        text=True,capture_output=True,timeout=2)
                fields = dict(s.split("=",1) for s in result.stdout.splitlines() if "=" in s)
            except (OSError,subprocess.SubprocessError):
                fields = {}
            with self.lock:
                self.cpu,self.gpus = cpu,gpus
                self.active,self.pid = fields.get("ActiveState","unknown"), int(fields.get("MainPID",0))
                self.samples.append({"timestamp":time.time(),"gap":not bool(gpus),
                                     "utilization":{"1":gpus[0]["utilization_percent"]} if gpus else {}})
            if time.monotonic() >= next_check:
                check = progress_check(self.run, self.active, self.pid, previous_progress)
                print(json.dumps({"event": "progress_check", **check}), flush=True)
                previous_progress = (self.pid, check["step"]) if check["phase"] == "training" else None
                next_check += 300
            time.sleep(3)

    def snapshot(self, run):
        status = read_json(run/"status.json", {})
        with self.lock:
            phase = actual_phase(status,self.active,self.pid)
            system = dict(cpu=self.cpu,gpus=self.gpus,gpu_history={"samples":list(self.samples)})
        status["phase"] = phase
        if phase not in ("baseline","starting","training","sampling"):
            status["next_checkpoint_at"] = None
        points = []
        try:
            with (run/"history.jsonl").open("rb") as stream:
                size = stream.seek(0,2)
                stream.seek(max(0,size-1100000))
                if size > 1100000:
                    stream.readline()
                points = [json.loads(s) for s in stream if s.endswith(b"\n")][-3000:]
        except FileNotFoundError:
            pass
        manifests = sorted(int(p.parent.name) for p in (run/"samples").glob("*/manifest.json")
                           if p.parent.name.isdigit())
        config = {"batch":BATCH,"lr":LR,"nfes":STEPS,"temps":TEMPS,"classes":CLASSES,
                  "schedule":[[s,beta(s)] for s in (0,21,22,5042)],"plateau":5042}
        return {**system,"config":config,"status":status,"history":points,"sample_steps":manifests,
                "live":read_json(run/"live"/"manifest.json",{}),"server_time":time.time()}


def serve(run, prx_html, bind):
    page, prx_js = build(prx_html, (HERE/"index.html").read_text())
    static = {"/":(page,"text/html; charset=utf-8"), "/prx.js":(prx_js,"application/javascript"),
              "/view.js":((HERE/"view.js").read_bytes(),"application/javascript")}
    state = State(run)
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = urlsplit(self.path).path
            if path == "/api/status":
                return self.reply(json.dumps(state.snapshot(run),allow_nan=False).encode(),"application/json")
            if path in static:
                return self.reply(*static[path])
            if IMAGE.fullmatch(path):
                try:
                    return self.reply((run/path[1:]).read_bytes(),"image/png")
                except FileNotFoundError:
                    pass
            return self.send_error(404)

        def reply(self, payload, mime):
            self.send_response(200)
            self.send_header("Content-Type",mime)
            self.send_header("Content-Length",str(len(payload)))
            self.send_header("Cache-Control","no-store")
            self.send_header("X-Content-Type-Options","nosniff")
            self.end_headers()
            try:
                self.wfile.write(payload)
            except (BrokenPipeError,ConnectionResetError):
                pass

        def log_message(self,*args):
            pass
    ThreadingHTTPServer((bind,8805),Handler).serve_forever()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--prx-html",type=Path,required=True)
    p.add_argument("--run",type=Path,default=RUN)
    p.add_argument("--bind", required=True, help="LAN address of the dashboard host")
    args = p.parse_args()
    serve(args.run,args.prx_html,args.bind)
