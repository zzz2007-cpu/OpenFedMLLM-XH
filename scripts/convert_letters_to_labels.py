#!/usr/bin/env python3
"""
Restore classification labels from option letters to full option text.

This script is the reverse of scripts/convert_labels_to_letters.py:
1) assistant content: normalize to "(A) original_option_text"
2) user prompt: remove "single-letter answer" instructions

Examples:
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

# Remove only answer-format constraints, keep the question/options themselves.
PROMPT_REMOVE_PATTERNS = [
    re.compile(
        r"Answer with the option's letter from the given choices directly and only give the best option\.?",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"Respond with ONLY one uppercase option letter from\s*\{?[A-Z,\s]+\}?\.",
        flags=re.IGNORECASE,
    ),
    re.compile(r"用单个字母回答这句话[。.!！?？]?", flags=0),
    re.compile(r"请只用单个字母回答[。.!！?？]?", flags=0),
]


def parse_options(user_content: str) -> Dict[str, str]:
    options: Dict[str, str] = {}
    for line in (user_content or "").splitlines():
        m = OPTION_LINE_RE.match(line)
        if m:
            options[m.group(1)] = m.group(2).strip()
    return options


def extract_letter(answer_text: str) -> str | None:
    text = (answer_text or "").strip()
    if not text:
        return None
    m = LEADING_OPTION_RE.match(text.upper())
    if m:
        return m.group(1)
    if len(text) == 1 and text.isalpha():
        return text.upper()
    return None


def normalize_text(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def infer_letter_from_answer(answer_text: str, options: Dict[str, str]) -> str | None:
    # 1) Direct letter forms: "A", "(A)", "A) ..."
    letter = extract_letter(answer_text)
    if letter and letter in options:
        return letter

    # 2) Match by option text: "not_humanitarian" / "(H) not_humanitarian"
    ans_norm = normalize_text(answer_text)
    if not ans_norm:
        return None
    for opt_letter, opt_name in options.items():
        opt_norm = normalize_text(opt_name)
        if not opt_norm:
            continue
        if ans_norm == opt_norm:
            return opt_letter
        if opt_norm in ans_norm:
            return opt_letter
    return None


def clean_user_prompt(content: str) -> Tuple[str, bool]:
    original = content or ""
    cleaned = original
    for pattern in PROMPT_REMOVE_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    # Light cleanup after deletions.
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n[ \t]+", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = cleaned.strip()
    return cleaned, cleaned != original


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


def convert_json_data(data: object) -> Tuple[object, int, int]:
    if not isinstance(data, list):
        return data, 0, 0

    restored_labels = 0
    cleaned_prompts = 0

    for sample in data:
        if not isinstance(sample, dict):
            continue
        conversations = sample.get("conversations", [])
        if not isinstance(conversations, list):
            continue

        user_content = ""
        user_turn = None
        for turn in conversations:
            if isinstance(turn, dict) and turn.get("role") == "user":
                user_content = str(turn.get("content", ""))
                user_turn = turn
                break
        options = parse_options(user_content)

        if user_turn is not None:
            new_user, changed = clean_user_prompt(user_content)
            if changed:
                user_turn["content"] = new_user
                cleaned_prompts += 1

        if not options:
            continue

        for turn in conversations:
            if not isinstance(turn, dict) or turn.get("role") != "assistant":
                continue
            old_answer = str(turn.get("content", "")).strip()
            letter = infer_letter_from_answer(old_answer, options)
            if not letter:
                continue
            if letter not in options:
                continue
            new_answer = f"({letter}) {options[letter]}"
            if old_answer != new_answer:
                turn["content"] = new_answer
                restored_labels += 1

    return data, restored_labels, cleaned_prompts


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Restore assistant labels from option letters to full option text."
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

    total_restored = 0
    total_cleaned = 0
    touched = []

    for path in files:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        new_data, restored, cleaned = convert_json_data(data)
        _ = new_data
        total_restored += restored
        total_cleaned += cleaned
        if restored > 0 or cleaned > 0:
            touched.append((path, restored, cleaned))

    print(f"Scanned files: {len(files)}")
    print(f"Restored assistant labels: {total_restored}")
    print(f"Cleaned user prompts: {total_cleaned}")
    if touched:
        print("Files with changes:")
        for path, restored, cleaned in touched:
            print(f"  {path}  (labels={restored}, prompts={cleaned})")

    if args.dry_run:
        print("Dry-run mode: no files were modified.")
        return 0

    written = 0
    for path in files:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        new_data, restored, cleaned = convert_json_data(data)
        if restored > 0 or cleaned > 0:
            with path.open("w", encoding="utf-8", newline="\n") as f:
                json.dump(new_data, f, ensure_ascii=False, indent=4)
                f.write("\n")
            written += 1

    print(f"Updated files: {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
