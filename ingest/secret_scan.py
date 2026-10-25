"""Find secrets with gitleaks and redact them before files are chunked.

Requires the gitleaks binary on PATH (https://github.com/gitleaks/gitleaks), or its path in
$GITLEAKS. Secret values are never logged.
"""

import json
import os
import shutil
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

MAX_TARGET_MEGABYTES = 10


class GitleaksNotFound(RuntimeError):
    pass


@dataclass(frozen=True)
class Finding:
    path: str  # relative to the repo root
    rule: str
    line: int
    secret: str

    def __repr__(self):
        return f"Finding({self.path}:{self.line} {self.rule})"


def gitleaks_binary() -> str:
    binary = os.environ.get("GITLEAKS") or shutil.which("gitleaks")
    if not binary:
        raise GitleaksNotFound(
            "gitleaks not found. Install it (macOS: `brew install gitleaks`; Linux: download the release "
            "binary from https://github.com/gitleaks/gitleaks/releases onto PATH) or set $GITLEAKS."
        )
    return binary


def scan(repo_dir: Path) -> dict[str, list[Finding]]:
    """Scan the working tree (not git history) and group findings by file."""
    result = subprocess.run(
        [
            gitleaks_binary(), "dir", ".",
            "--report-format", "json", "--report-path", "-",
            "--exit-code", "0", "--no-banner", "--log-level", "error",
            "--max-target-megabytes", str(MAX_TARGET_MEGABYTES),
        ],
        cwd=repo_dir,  # so reported paths are relative to the repo root
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"gitleaks failed ({result.returncode}): {result.stderr.strip()}")
    findings = defaultdict(list)
    for item in json.loads(result.stdout or "[]"):
        finding = Finding(Path(item["File"]).as_posix().removeprefix("./"), item["RuleID"], item["StartLine"], item["Secret"])
        findings[finding.path].append(finding)
    return dict(findings)


def redact(raw: bytes, findings: list[Finding]) -> bytes:
    """Replace each secret, keeping its newlines so line numbers don't move."""
    for finding in sorted(findings, key=lambda f: len(f.secret), reverse=True):
        if not finding.secret:
            continue
        secret = finding.secret.encode()
        replacement = f"[REDACTED:{finding.rule}]".encode() + b"\n" * secret.count(b"\n")
        raw = raw.replace(secret, replacement)
    return raw
