#!/usr/bin/env python3
"""
Pulls compressed `/data/logs/*.jsonl.gz` from `vol://jlens-data/logs` into `./logs/`,
rebuilds `runs/manifest.jsonl`, and prints cumulative MI325X GPU-hour budget usage & val_ce leaderboard.
"""
import gzip
import json
import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR  = REPO_ROOT / "logs"
MANIFEST  = REPO_ROOT / "runs/manifest.jsonl"
FLS_BIN   = os.environ.get("FEATHERLESS_BIN", str(Path.home() / ".local/bin/featherless-cloud"))


def main():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [FLS_BIN, "volume", "sync", "vol://jlens-data/logs", str(LOGS_DIR), "--non-interactive"],
        cwd=REPO_ROOT, check=False,
    )

    records = []
    tot_bf16_gpu_h  = 0.0
    tot_fp8_gpu_h   = 0.0
    tot_jlens_gpu_h = 0.0
    tot_other_h     = 0.0

    for gz_path in sorted(LOGS_DIR.glob("*.jsonl.gz")):
        meta, summary = {}, {}
        last_step = {}
        try:
            with gzip.open(gz_path, "rt", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    tp = obj.get("type")
                    if tp == "meta":
                        meta = obj
                    elif tp == "step":
                        last_step = obj
                    elif tp == "summary":
                        summary = obj
        except Exception as e:
            print(f"[warn] Could not parse {gz_path.name}: {e}")
            continue

        gpu_s = float(summary.get("elapsed_s", last_step.get("elapsed_s", 0.0)))
        gpu_h = gpu_s / 3600.0
        lmode = meta.get("linear_mode", summary.get("linear_mode", "bench"))
        jmode = meta.get("jlens_mode", summary.get("jlens_mode", "none"))
        kind  = meta.get("kind", summary.get("kind", "train"))

        if jmode and jmode != "none":
            tot_jlens_gpu_h += gpu_h
        elif kind == "kernel_bench" and ("fp8" in gz_path.name or "02" in gz_path.name):
            tot_fp8_gpu_h += gpu_h
        elif kind == "kernel_bench":
            tot_bf16_gpu_h += gpu_h
        elif "fp8" in str(lmode) or "fp8" in gz_path.name:
            tot_fp8_gpu_h += gpu_h
        elif lmode == "bf16" or "bf16" in gz_path.name:
            tot_bf16_gpu_h += gpu_h
        else:
            tot_other_h += gpu_h

        rec = {
            "log_file": gz_path.name,
            "run_id": meta.get("run_id", gz_path.stem),
            "commit_sha": meta.get("commit_sha", "unknown"),
            "script": meta.get("script", "unknown"),
            "kind": kind,
            "linear_mode": lmode,
            "jlens_mode": jmode,
            "jlens_weight": meta.get("jlens_weight", 0.0),
            "opt": meta.get("opt", ""),
            "lr": meta.get("lr", None),
            "wd": meta.get("wd", None),
            "toks_per_step": meta.get("toks_per_step", None),
            "steps_done": summary.get("steps", last_step.get("step", 0)),
            "val_ce": summary.get("val_ce", last_step.get("val_ce", None)),
            "tok_per_s": summary.get("tok_per_s", last_step.get("tok_per_s", None)),
            "ms_per_step": summary.get("ms_per_step", last_step.get("ms_per_step", None)),
            "gpu_hours": round(gpu_h, 4),
        }
        records.append(rec)

    with open(MANIFEST, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print("=" * 124)
    print(f" MI325X BUDGET TRACKER | Stage1 BF16: {tot_bf16_gpu_h:.3f}/3.00h | Stage2 FP8: {tot_fp8_gpu_h:.3f}/3.00h (min 1.0h) | Phase3 J-Lens: {tot_jlens_gpu_h:.3f}h | Misc: {tot_other_h:.3f}h")
    print("=" * 124)
    print(f"{'Run ID':32s} | {'SHA':7s} | {'Mode':18s} | {'Steps':>5s} | {'Val CE':>8s} | {'ms/st':>7s} | {'tok/s':>8s} | {'GPU-h':>6s}")
    print("-" * 116)
    for r in records:
        v_s  = f"{r['val_ce']:8.4f}" if isinstance(r["val_ce"], (int, float)) else "       -"
        ms_s = f"{r['ms_per_step']:7.1f}" if isinstance(r["ms_per_step"], (int, float)) else "      -"
        tk_s = f"{r['tok_per_s']/1e3:7.1f}k" if isinstance(r["tok_per_s"], (int, float)) else "       -"
        print(f"{str(r['run_id'])[:32]:32s} | {str(r['commit_sha'])[:7]:7s} | {str(r['linear_mode'])[:18]:18s} | "
              f"{int(r['steps_done'] or 0):5d} | {v_s} | {ms_s} | {tk_s} | {r['gpu_hours']:6.3f}")
    print("=" * 116)


if __name__ == "__main__":
    main()
