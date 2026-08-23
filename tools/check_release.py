#!/usr/bin/env python3
"""Fail when a release contains machine paths, credentials, or large artifacts."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
TEXT_SUFFIXES = {
    "",
    ".env",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
MAX_FILE_BYTES = 10 * 1024 * 1024

PATTERNS = {
    "machine-specific absolute path": re.compile(
        r"(?:/remote" r"-home/|/ho" r"me/[A-Za-z0-9._-]+/|/ro" r"ot/)"
    ),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "GitHub token": re.compile(r"\b(?:ghp|github_pat)_[A-Za-z0-9_]{20,}\b"),
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "hard-coded CUDA device": re.compile(r"CUDA_VISIBLE_DEVICES\s*=\s*[0-9]"),
}


def iter_files():
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in SKIP_DIRS for part in path.parts):
            continue
        yield path


def main() -> int:
    problems: list[str] = []
    for path in iter_files():
        relative = path.relative_to(ROOT)
        size = path.stat().st_size
        if size > MAX_FILE_BYTES:
            problems.append(f"large file ({size} bytes): {relative}")
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in {
            ".gitignore",
            "config.example.env",
        }:
            problems.append(f"unexpected artifact type: {relative}")
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            problems.append(f"binary file: {relative}")
            continue
        for label, pattern in PATTERNS.items():
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                problems.append(f"{label}: {relative}:{line}")

    if problems:
        print("Release check failed:")
        for problem in problems:
            print(f"- {problem}")
        return 1

    print("Release check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
