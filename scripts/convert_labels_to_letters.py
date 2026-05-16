#!/usr/bin/env python3
"""
Convert classification assistant labels from full text to single option letters.

Example:
    python scripts/convert_labels_to_letters.py partition-alpha0.5-clt10 test.json --dry-run
    python scripts/convert_labels_to_letters.py partition-alpha0.5-clt10 test.json

Reverse conversion:
    python scripts/convert_letters_to_labels.py partition-alpha0.5-clt10 test.json --dry-run
    python scripts/convert_letters_to_labels.py partition-alpha0.5-clt10 test.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


OPTION_LINE_RE = re.compile(r"^\s*\(([A-Z])\)\s*(.+?)\s*$")
LEADING_OPTION_RE = re.compile(r"^\s*\(?([A-Z])\)?(?:\s|$|[\.\):\-])")


def normalize_text(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_options(user_content: str) -> Dict[str, str]:
    options: Dict[str, str] = {}
    for line in (user_content or "").splitlines():
        m = OPTION_LINE_RE.match(line)
        if m:
            options[m.group(1)] = m.group(2).strip()
    return options


def extract_letter(label_text: str, options: Dict[str, str]) -> str | None:
    text = (label_text or "").strip()
    if not text:
        return None

    # Fast path: "(B) xxx", "B", "B)"...
    m = LEADING_OPTION_RE.match(text.upper())
    if m:
        letter = m.group(1)
        if not options or letter in options:
            return letter

    if not options:
        return None

    # Fallback: match by option name text.
    text_norm = normalize_text(text)
    for letter, option_name in options.items():
        option_norm = normalize_text(option_name)
        if not option_norm:
            continue
        if text_norm == option_norm:
            return letter
        if option_norm in text_norm:
            return letter

    return None


def iter_target_files(targets: Iterable[str]) -> List[Path]:
    files: List[Path] = []
    for raw in targets:
        p = Path(raw)
        if p.is_file() and p.suffix.lower() == ".json":
            files.append(p)
            continue
        if p.is_dir():
            files.extend(sorted(p.glob("client_*.json")))
            continue
        raise FileNotFoundError(f"Target not found or unsupported: {raw}")
    # De-duplicate while preserving order.
    dedup: List[Path] = []
    seen = set()
    for f in files:
        key = str(f.resolve())
        if key not in seen:
            dedup.append(f)
            seen.add(key)
    return dedup


def convert_json_file(path: Path) -> Tuple[int, int]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        return 0, 0

    converted = 0
    for sample in data:
        if not isinstance(sample, dict):
            continue
        conversations = sample.get("conversations", [])
        if not isinstance(conversations, list):
            continue

        user_content = ""
        for turn in conversations:
            if isinstance(turn, dict) and turn.get("role") == "user":
                user_content = str(turn.get("content", ""))
                break
        options = parse_options(user_content)
        if not options:
            continue

        for turn in conversations:
            if not isinstance(turn, dict) or turn.get("role") != "assistant":
                continue
            old = str(turn.get("content", "")).strip()
            new_letter = extract_letter(old, options)
            if new_letter and old != new_letter:
                turn["content"] = new_letter
                converted += 1

    return converted, len(data)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert assistant labels to single uppercase letters."
    )
    parser.add_argument(
        "targets",
        nargs="+",
        help="JSON files and/or directories (directories will process client_*.json).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without writing files.",
    )
    args = parser.parse_args()

    files = iter_target_files(args.targets)
    if not files:
        print("No files found.")
        return 0

    total_converted = 0
    total_samples = 0
    touched = []

    for path in files:
        converted, sample_count = convert_json_file(path)
        total_converted += converted
        total_samples += sample_count
        if converted > 0:
            touched.append((path, converted, sample_count))

    print(f"Scanned files: {len(files)}")
    print(f"Scanned samples: {total_samples}")
    print(f"Convertible assistant labels: {total_converted}")

    if touched:
        print("Files with changes:")
        for path, converted, sample_count in touched:
            print(f"  {path}  ({converted} labels / {sample_count} samples)")

    if args.dry_run:
        print("Dry-run mode: no files were modified.")
        return 0

    # Re-run conversion and persist.
    written = 0
    for path in files:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            continue

        changed = False
        for sample in data:
            if not isinstance(sample, dict):
                continue
            conversations = sample.get("conversations", [])
            if not isinstance(conversations, list):
                continue

            user_content = ""
            for turn in conversations:
                if isinstance(turn, dict) and turn.get("role") == "user":
                    user_content = str(turn.get("content", ""))
                    break
            options = parse_options(user_content)
            if not options:
                continue

            for turn in conversations:
                if not isinstance(turn, dict) or turn.get("role") != "assistant":
                    continue
                old = str(turn.get("content", "")).strip()
                new_letter = extract_letter(old, options)
                if new_letter and old != new_letter:
                    turn["content"] = new_letter
                    changed = True

        if changed:
            with path.open("w", encoding="utf-8", newline="\n") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
                f.write("\n")
            written += 1

    print(f"Updated files: {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
