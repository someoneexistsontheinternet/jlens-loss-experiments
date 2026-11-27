"""Compact gzip-compressed JSONL logger for reproducible GitHub run artifacts."""
import gzip
import json
import os
import subprocess
import time
from pathlib import Path
import torch


class CompressedJSONLLogger:
    def __init__(self, out_path: Path, meta: dict):
        self.out_path = Path(out_path)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.t0 = time.time()
        self._fh = gzip.open(self.out_path, "wt", encoding="utf-8")

        sha = os.environ.get("GIT_COMMIT_SHA", "")
        if not sha:
            try:
                sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            except Exception:
                sha = "unknown"

        dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        arch     = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "cuda") if torch.cuda.is_available() else "cpu"
        header = {
            "type": "meta",
            "run_id": os.environ.get("RUN_ID", self.out_path.stem),
            "commit_sha": sha,
            "script": os.environ.get("ATTEMPT_SCRIPT", ""),
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "torch": torch.__version__,
            "hip": getattr(torch.version, "hip", None),
            "device": dev_name,
            "arch": arch,
            **meta,
        }
        self.write(header)

    def write(self, record: dict):
        record["elapsed_s"] = round(time.time() - self.t0, 3)
        self._fh.write(json.dumps(record, separators=(",", ":")) + "\n")
        self._fh.flush()

    def close(self, summary: dict = None):
        if summary is not None:
            self.write({"type": "summary", **summary})
        self._fh.close()
