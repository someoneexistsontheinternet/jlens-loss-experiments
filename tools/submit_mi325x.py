#!/usr/bin/env python3
"""
Commit-pinned Featherless MI325X job submitter.
  1. Verifies zero secrets via tools/secret_scan.py.
  2. Verifies git working tree is clean & pushed to origin/main so exact COMMIT_SHA is reproducible on GitHub.
  3. Submits a 1x MI325X Run Job (`--gpu-type mi325x --gpu-count 1 --volume jlens-data:/data`) with ZERO local env secrets.
"""
import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REPO_URL  = "https://github.com/someoneexistsontheinternet/jlens-loss-experiments.git"
FLS_BIN   = os.environ.get("FEATHERLESS_BIN", str(Path.home() / ".local/bin/featherless-cloud"))


def sh(cmd: list[str], check: bool = True) -> str:
    r = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=check)
    return r.stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("script", help="Relative path to script, e.g. scripts/001_mi325x_kernel_bench.sh")
    ap.add_argument("--name", default=None, help="Optional Featherless job name")
    ap.add_argument("--volume", default="jlens-data:/data")
    ap.add_argument("--template", default="rocm-pytorch")
    ap.add_argument("--timeout", type=int, default=14400, help="Job timeout seconds (default 4h)")
    ap.add_argument("--wait", action="store_true", help="Stream job attached instead of --detach")
    ap.add_argument("--auto-commit", action="store_true", help="Auto commit & push script changes before submitting")
    args = ap.parse_args()

    # 1. Secret scan
    rc = subprocess.call([sys.executable, str(REPO_ROOT / "tools/secret_scan.py")], cwd=REPO_ROOT)
    if rc != 0:
        sys.exit(rc)

    script_rel = str(Path(args.script))
    if not (REPO_ROOT / script_rel).is_file():
        sys.exit(f"Script not found: {script_rel}")

    status = sh(["git", "status", "--porcelain"])
    if status:
        if args.auto_commit:
            sh(["git", "add", "-A"])
            if subprocess.call([sys.executable, str(REPO_ROOT / "tools/secret_scan.py")], cwd=REPO_ROOT) != 0:
                sys.exit(1)
            sh(["git", "commit", "-m", f"exp: prepare {script_rel}"])
            sh(["git", "push", "origin", "main"])
        else:
            sys.exit(f"Uncommitted changes present! Commit & push first (or pass --auto-commit):\n{status}")
    else:
        # Ensure local HEAD is pushed to origin/main
        sh(["git", "push", "origin", "main"])

    commit_sha = sh(["git", "rev-parse", "HEAD"])
    sha7       = commit_sha[:7]
    stem       = Path(script_rel).stem
    run_id     = args.name or f"{stem}-{sha7}-{int(time.time()) % 100000}"

    remote_Bootstrap = (
        f"set -euo pipefail; "
        f"export RUN_ID={shlex.quote(run_id)} GIT_COMMIT_SHA={shlex.quote(commit_sha)} ATTEMPT_SCRIPT={shlex.quote(script_rel)} DATA_ROOT=/data; "
        f"mkdir -p /data/logs /data/checkpoints /tmp/work; "
        f"cd /tmp/work && rm -rf repo && git clone --quiet {shlex.quote(REPO_URL)} repo && cd repo; "
        f"git checkout --quiet {shlex.quote(commit_sha)}; "
        f"test \"$(git rev-parse HEAD)\" = {shlex.quote(commit_sha)}; "
        f"bash {shlex.quote(script_rel)}"
    )

    cmd = [
        FLS_BIN, "exec",
        "--name", run_id,
        "--template", args.template,
        "--gpu-type", "mi325x",
        "--gpu-count", "1",
        "--volume", args.volume,
        "--timeout", str(args.timeout),
        "--non-interactive",
    ]
    if not args.wait:
        cmd.append("--detach")
    cmd.extend(["--", "bash", "-lc", remote_Bootstrap])

    print(f"[submit] Launching {run_id} @ commit {commit_sha} -> {script_rel}")
    out = sh(cmd)
    print(f"[submit] Featherless output: {out}")

    # Record launch entry locally
    launch_rec = {
        "event": "submitted",
        "run_id": run_id,
        "job_output": out.splitlines()[-1] if out else "",
        "commit_sha": commit_sha,
        "script": script_rel,
        "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(REPO_ROOT / "runs/submissions.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(launch_rec) + "\n")


if __name__ == "__main__":
    main()
