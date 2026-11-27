#!/usr/bin/env python3
"""Pre-commit / pre-push scanner ensuring zero environment secrets or API keys exist in tracked/staged files."""
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

PATTERNS = [
    (re.compile(r"gh[pousr]_[A-Za-z0-9_]{20,}"), "GitHub token"),
    (re.compile(r"github_pat_[A-Za-z0-9_]{20,}"), "GitHub fine-grained PAT"),
    (re.compile(r"sk-[A-Za-z0-9\-_]{20,}"), "OpenAI/Anthropic/API sk- key"),
    (re.compile(r"hf_[A-Za-z0-9]{20,}"), "HuggingFace token"),
    (re.compile(r"fls_[A-Za-z0-9_\-]{16,}"), "Featherless token"),
    (re.compile(r"AIza[0-9A-Za-z\-_]{30,}"), "Google API key"),
]


def get_sensitive_env_values() -> list[tuple[str, str]]:
    out = []
    for k, v in os.environ.items():
        ku = k.upper()
        if any(tag in ku for tag in ("KEY", "TOKEN", "SECRET", "PASSWORD", "CRED", "AUTH")):
            val = v.strip()
            if len(val) >= 12 and val not in ("true", "false"):
                out.append((k, val))
    return out


def scan() -> int:
    res = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    )
    env_secrets = get_sensitive_env_values()
    bad = []

    for rel in res.stdout.splitlines():
        p = REPO_ROOT / rel
        if not p.is_file() or rel.endswith(".gz"):
            continue
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for rx, label in PATTERNS:
            if rx.search(txt):
                bad.append(f"{rel}: matched {label} pattern")
        for env_k, env_v in env_secrets:
            if env_v in txt:
                bad.append(f"{rel}: contains literal value of local env var {env_k}")

    if bad:
        print("SECRET SCAN FAILED:", file=sys.stderr)
        for item in bad:
            print(f"  - {item}", file=sys.stderr)
        return 1
    print("Secret scan clean (0 tokens / env secrets found).")
    return 0


if __name__ == "__main__":
    sys.exit(scan())
