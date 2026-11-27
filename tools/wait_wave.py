#!/usr/bin/env python3
"""Waits until all non-terminal Featherless Run Jobs complete, printing concise progress."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

FLS_BIN = os.environ.get("FEATHERLESS_BIN", str(Path.home() / ".local/bin/featherless-cloud"))
TERMINAL = {"succeeded", "failed", "cancelled", "canceled", "timed_out", "expired", "error"}


def main():
    poll_s = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    t0 = time.time()
    while True:
        try:
            out = subprocess.check_output([FLS_BIN, "runjob", "list", "--json"], text=True)
            items = json.loads(out).get("items", [])
        except Exception as e:
            print(f"[wait_wave] transient list error: {e}", flush=True)
            time.sleep(poll_s)
            continue

        active = [it for it in items if str(it.get("state", "")).lower() not in TERMINAL]
        elapsed = int(time.time() - t0)
        if not active:
            recent = [(it.get("name"), it.get("state"), it.get("exitCode")) for it in items[:8]]
            print(f"[wait_wave] All jobs finished at +{elapsed}s! Recent: {recent}", flush=True)
            break
        summary = ", ".join(f"{it.get('name')[:22]}:{it.get('state')}" for it in active)
        print(f"[wait_wave +{elapsed:4d}s] {len(active)} active -> {summary}", flush=True)
        time.sleep(poll_s)


if __name__ == "__main__":
    main()
