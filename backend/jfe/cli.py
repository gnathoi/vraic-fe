"""Replay an exported scenario and compare with its archived result.

python -m jfe.cli run --scenario scenario.json --output ./reproduced [--compare result.json]
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .engine import load_pack, run_experiment
from .schemas import Scenario


def main():
    ap = argparse.ArgumentParser(prog="jfe.cli")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--scenario", required=True, type=Path)
    r.add_argument("--output", required=True, type=Path)
    r.add_argument("--compare", type=Path, help="archived result.json to compare against")
    r.add_argument("--backend", default=None, choices=["cpu", "gpu"], help="default: the archived run's backend (GPU and CPU draws differ), else cpu")
    a = ap.parse_args()
    sc = Scenario.model_validate_json(a.scenario.read_text())
    backend = a.backend or ("gpu" if a.compare and "cuda" in json.loads(a.compare.read_text())["receipt"]["backend"] else "cpu")
    res = run_experiment(sc, load_pack(), backend)
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / "result.json").write_text(json.dumps(res))
    print(f"experiment {res['experiment_hash'][:16]} pack {res['pack_id']} backend {res['receipt']['backend']}")
    if a.compare:
        old = json.loads(a.compare.read_text())
        if old["pack_id"] != res["pack_id"]:
            print(f"WARNING: archived data pack {old['pack_id']} differs from this image's {res['pack_id']}")
        worst = 0.0
        for series in ("scenario", "comparator", "difference"):
            for m, blk in res["metrics"][series].items():
                for agg, vals in blk.items():
                    pairs = [(a, b) for a, b in zip(vals, old["metrics"][series][m][agg]) if a is not None and b is not None]
                    if pairs:
                        worst = max(worst, max(abs(a - b) for a, b in pairs))
        ok = worst < 1e-6
        print(f"max absolute difference vs archive: {worst:.3g} -> {'REPRODUCED' if ok else 'DIFFERS'}")
        raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
