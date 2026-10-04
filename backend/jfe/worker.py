"""Single numerical worker: dispatcher and executor in one loop, concurrency 1.

Selection policy is enforced here (not by queue order): REHEARSAL/PRESENTATION dispatch operator runs only;
OPEN_DEMO and DRAINING dispatch everyone (operator first, then FIFO; one outstanding run per session is
enforced at admission); CLOSED dispatches nothing."""
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback

from . import db, exports
from .engine import load_pack, run_experiment
from .schemas import Scenario

log = logging.getLogger("jfe.worker")
QUEUE_LIFETIME = os.environ.get("JFE_QUEUE_LIFETIME", "5 minutes")
LEASE = "2 minutes"
GPU_MIN_DRAWS = int(os.environ.get("JFE_GPU_MIN_DRAWS", "1024"))
BACKEND = os.environ.get("JFE_SIM_BACKEND", "auto")  # auto | cpu | gpu


def gpu_available():
    try:
        import cupy
        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        log.exception("GPU unavailable; using the NumPy backend")
        return False


HAS_GPU = BACKEND != "cpu" and gpu_available()


def choose_backend(sc: Scenario):
    if BACKEND == "gpu" and HAS_GPU:
        return "gpu"
    if BACKEND == "auto" and HAS_GPU and sc.uncertainty.draws >= GPU_MIN_DRAWS:
        return "gpu"
    return "cpu"


def reconcile():
    """On startup: a run whose lease expired was abandoned by a dead worker. Requeue once, then fail."""
    db.q("update runs set status='queued', lease_until=null where status in ('running','cancel_requested') and lease_until < now() and attempts < 2")
    db.q("update runs set status='failed', error='Worker stopped twice while running this job.', finished=now() where status in ('running','cancel_requested') and lease_until < now()")


def expire():
    db.q(f"update runs set status='expired', finished=now(), error='Waited longer than {QUEUE_LIFETIME} in the queue. Resubmitting does not use extra quota.' where status='queued' and created < now() - interval '{QUEUE_LIFETIME}'")


def claim():
    with db.pool().connection() as c, c.transaction():
        mode = c.execute("select value from settings where key='mode'").fetchone()["value"]
        if mode == "CLOSED":
            return None
        who = "and is_operator" if mode in ("REHEARSAL", "PRESENTATION") else ""
        row = c.execute(f"select * from runs where status='queued' {who} order by is_operator desc, created limit 1 for update skip locked").fetchone()
        if row:
            c.execute(f"update runs set status='running', started=now(), lease_until=now() + interval '{LEASE}', attempts=attempts+1, queue_seconds=extract(epoch from now()-created) where id=%s", (row["id"],))
        return row


def execute(run):
    sc = Scenario.model_validate(run["scenario"])
    cached = db.one("select result from results where experiment_hash=%s", run["experiment_hash"])
    if cached:  # numbers are shared by experiment hash; the scenario text (title, notes, parent) is always this run's own
        result, cache_status = cached["result"] | {"scenario": run["scenario"]}, "Reused calculation"
    else:
        backend = choose_backend(sc)
        try:
            result = run_experiment(sc, load_pack(), backend)
        except ValueError:
            raise
        except Exception:
            if backend != "gpu":
                raise
            log.exception("GPU execution failed; retrying on CPU")
            result = run_experiment(sc, load_pack(), "cpu")
            result["warnings"].append("GPU execution failed; this result was computed on the CPU.")
        cache_status = "Fresh calculation"
        db.q("insert into results (experiment_hash, result) values (%s, %s) on conflict do nothing", run["experiment_hash"], db.J(result))
    if db.one("select status from runs where id=%s", run["id"])["status"] == "cancel_requested":
        db.q("update runs set status='cancelled', finished=now() where id=%s", run["id"])
        return
    # numerical results are published immediately; exports are rendered by the separate report worker
    db.q("update runs set status='complete', finished=now(), cache_status=%s, backend=%s, compute_seconds=%s, artifacts_status='pending', lease_until=null where id=%s",
         cache_status, result["receipt"]["backend"], result["receipt"]["compute_seconds"], run["id"])


def report_loop():
    """Report worker (CPU only): renders exports for completed runs; a failure never affects numerical results."""
    log.info("report worker ready")
    while True:
        with db.pool().connection() as c, c.transaction():
            run = c.execute("select * from runs where status='complete' and artifacts_status in ('pending') order by finished limit 1 for update skip locked").fetchone()
            if run:
                c.execute("update runs set artifacts_status='rendering' where id=%s", (run["id"],))
        if not run:
            time.sleep(0.2)
            continue
        try:
            result = db.one("select result from results where experiment_hash=%s", run["experiment_hash"])["result"] | {"scenario": run["scenario"]}
            exports.build(run, result, run["cache_status"])
            db.q("update runs set artifacts_status='ready' where id=%s", run["id"])
            db.event("export", "ready")
        except Exception:
            log.exception("export build failed for %s", run["id"])
            db.q("update runs set artifacts_status='failed' where id=%s", run["id"])
            db.event("export", "failed")


def system_stats():
    out = {}
    try:
        la = open("/proc/loadavg").read().split()
        out["load_1m"], out["load_5m"] = float(la[0]), float(la[1])
        out["cpu_count"] = os.cpu_count()
        mem = dict(line.split(":", 1) for line in open("/proc/meminfo"))
        kb = lambda k: int(mem[k].split()[0])  # noqa: E731
        out["ram_total_gb"], out["ram_available_gb"] = round(kb("MemTotal") / 2**20, 1), round(kb("MemAvailable") / 2**20, 1)
    except Exception:
        pass
    if shutil.which("nvidia-smi"):
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit",
                                "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
            n, u, mu, mt, t, pd, pl = [x.strip() for x in r.stdout.strip().splitlines()[0].split(",")]
            out["gpu"] = {"name": n, "util_pct": float(u), "mem_used_mb": float(mu), "mem_total_mb": float(mt), "temp_c": float(t), "power_w": float(pd), "power_limit_w": float(pl)}
        except Exception as e:
            out["gpu_error"] = str(e)[:200]
    out["backend"], out["gpu_available"], out["gpu_min_draws"] = BACKEND, HAS_GPU, GPU_MIN_DRAWS
    return out


def heartbeat():
    db.q("insert into heartbeat (component, at, data) values ('worker', now(), %s) on conflict (component) do update set at=now(), data=excluded.data", db.J(system_stats()))


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))  # PID 1 in the container; leases cover in-flight work
    db.init()
    if "--reports" in sys.argv:
        db.q("update runs set artifacts_status='pending' where artifacts_status='rendering'")  # reconcile after a crash
        load_pack()
        return report_loop()
    load_pack()
    reconcile()
    log.info("worker ready; backend=%s gpu=%s gpu_min_draws=%s", BACKEND, HAS_GPU, GPU_MIN_DRAWS)
    if HAS_GPU:  # warm CUDA context and kernels so the first audience run is not a cold start
        from .schemas import Uncertainty, default_scenario
        run_experiment(default_scenario(uncertainty=Uncertainty(mode="process", draws=64)), load_pack(), "gpu")
    last_expire = 0.0
    while True:
        if time.time() - last_expire > 5:
            expire()
            heartbeat()
            last_expire = time.time()
        run = claim()
        if not run:
            time.sleep(0.2)
            continue
        try:
            execute(run)
            db.event("run", "complete")
        except ValueError as e:  # semantic / feasibility failure: never retried
            db.q("update runs set status='failed', error=%s, finished=now(), lease_until=null where id=%s", str(e), run["id"])
            db.event("run", "failed_validation")
        except Exception:
            log.error("run %s failed: %s", run["id"], traceback.format_exc())
            db.q("update runs set status='failed', error='Internal error while computing this run.', finished=now(), lease_until=null where id=%s", run["id"])
            db.event("run", "failed_internal")


if __name__ == "__main__":
    main()
