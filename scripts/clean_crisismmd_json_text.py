#!/usr/bin/env python3
"""
Clean noisy CrisisMMD JSON conversations for train/test consistency.

What this script cleans in user tweet text:
1) URL links (http/https/www)
2) Retweet prefixes like: RT @user: / RT @user
3) Hashtag symbol '#' (keeps the token text)
4) Common mojibake / encoding-noise fragments

The script only edits the tweet-text segment inside user content and preserves
the prompt template/question/options/assistant labels.

Examples:
    python scripts/clean_crisismmd_json_text.py test.json --dry-run
    python scripts/clean_crisismmd_json_text.py test.json
    python scripts/clean_crisismmd_json_text.py partition-alpha0.5-clt10 test.json
"""

from __future__ import annotations

import argparse
import html
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple


URL_RE = re.compile(r"(?:https?://|www\.)\S+", flags=re.IGNORECASE)
RT_PREFIX_RE = re.compile(r"^\s*(?:RT\s+@[A-Za-z0-9_]{1,20}:?\s*)+", flags=re.IGNORECASE)
HASHTAG_SIGN_RE = re.compile(r"#")
QUESTION_LINE_RE = re.compile(
    r"^(What is .*?\?|Is this image and text informative about the crisis event\?)$"
)
GREEK_EXTENDED_RE = re.compile(r"[\u1F00-\u1FFF]+")
CONTROL_CHAR_RE = re.compile(r"[\u0000-\u0008\u000B-\u001F\u007F-\u009F]+")

# Common mojibake fragments observed in this corpus.
MOJIBAKE_MAP = {
    "â€™": "'",
    "â€˜": "'",
    "â€œ": '"',
    "â€": '"',
    "â€“": "-",
    "â€”": "-",
    "â€¦": "...",
    "‚Ä¶": "...",
    "‚Äù": '"',
    "‚Äú": '"',
    "‚Äô": "'",
    "Â": "",
}


@dataclass
class CleanStats:
    files_scanned: int = 0
    files_updated: int = 0
    samples_scanned: int = 0
    samples_changed: int = 0
    urls_removed: int = 0
    rt_prefix_removed: int = 0
    hashtag_signs_removed: int = 0
    mojibake_fixes: int = 0
    parse_skipped: int = 0


def iter_target_files(targets: Iterable[str]) -> List[Path]:
    files: List[Path] = []
    for raw in targets:
        p = Path(raw)
        if p.is_file() and p.suffix.lower() == ".json":
            files.append(p)
            continue
        if p.is_dir():
            files.extend(sorted(p.glob("client_*.json")))
            files.extend(sorted(p.glob("test*.json")))
            continue
        raise FileNotFoundError(f"Target not found or unsupported: {raw}")

    dedup: List[Path] = []
    seen = set()
    for f in files:
        key = str(f.resolve())
        if key not in seen:
            dedup.append(f)
            seen.add(key)
    return dedup


def _maybe_fix_mojibake(text: str) -> Tuple[str, int]:
    original = text
    changed = 0

    text = html.unescape(text)

    for bad, good in MOJIBAKE_MAP.items():
        if bad in text:
            text = text.replace(bad, good)
            changed += 1

    # Recover common "UTF-8 bytes decoded as cp1252" artifacts.
    if re.search(r"[âÃÂ‚á¼á½]", text):
        try:
            repaired = text.encode("cp1252", errors="ignore").decode("utf-8", errors="ignore")
        except Exception:
            repaired = text
        if repaired and repaired != text:
            text = repaired
            changed += 1

    # Remove leftover Greek-extended garbage often produced by broken decode.
    if GREEK_EXTENDED_RE.search(text):
        text = GREEK_EXTENDED_RE.sub("", text)
        changed += 1

    if CONTROL_CHAR_RE.search(text):
        text = CONTROL_CHAR_RE.sub(" ", text)
        changed += 1

    if text != original:
        return text, changed
    return text, 0


def clean_tweet_text(tweet_text: str) -> Tuple[str, dict]:
    text = tweet_text or ""
    meta = {
        "urls_removed": 0,
        "rt_prefix_removed": 0,
        "hashtag_signs_removed": 0,
        "mojibake_fixes": 0,
    }

    text, mfix = _maybe_fix_mojibake(text)
    meta["mojibake_fixes"] += mfix

    url_hits = URL_RE.findall(text)
    if url_hits:
        meta["urls_removed"] += len(url_hits)
        text = URL_RE.sub("", text)

    rt_hit = RT_PREFIX_RE.search(text)
    if rt_hit:
        text = RT_PREFIX_RE.sub("", text)
        meta["rt_prefix_removed"] += 1

    hash_hits = HASHTAG_SIGN_RE.findall(text)
    if hash_hits:
        meta["hashtag_signs_removed"] += len(hash_hits)
        text = HASHTAG_SIGN_RE.sub("", text)

    # Final polish.
    text = re.sub(r"\s+", " ", text).strip()
    return text, meta


def clean_user_content(user_content: str) -> Tuple[str, dict, bool]:
    lines = (user_content or "").splitlines()
    meta = {
        "urls_removed": 0,
        "rt_prefix_removed": 0,
        "hashtag_signs_removed": 0,
        "mojibake_fixes": 0,
        "parse_skipped": 0,
    }

    # Expected format:
    # line0: <image>
    # line1: Select the best answer...
    # line2..k-1: tweet text
    # linek: question line
    if len(lines) < 4 or lines[0].strip() != "<image>":
        meta["parse_skipped"] = 1
        return user_content, meta, False

    question_idx = -1
    for idx in range(2, len(lines)):
        if QUESTION_LINE_RE.match(lines[idx].strip()):
            question_idx = idx
            break

    if question_idx <= 2:
        meta["parse_skipped"] = 1
        return user_content, meta, False

    tweet_text = "\n".join(lines[2:question_idx]).strip()
    cleaned_tweet, cmeta = clean_tweet_text(tweet_text)
    for key in ("urls_removed", "rt_prefix_removed", "hashtag_signs_removed", "mojibake_fixes"):
        meta[key] += cmeta[key]

    new_lines = lines[:2] + [cleaned_tweet] + lines[question_idx:]
    new_content = "\n".join(new_lines)
    changed = new_content != user_content
    return new_content, meta, changed


def process_file(path: Path) -> Tuple[object, bool, CleanStats]:
    local = CleanStats(files_scanned=1)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        return data, False, local

    changed_any = False
    for sample in data:
        if not isinstance(sample, dict):
            continue
        local.samples_scanned += 1
        conversations = sample.get("conversations", [])
        if not isinstance(conversations, list):
            continue
        for turn in conversations:
            if not isinstance(turn, dict) or turn.get("role") != "user":
                continue
            old = str(turn.get("content", ""))
            new, meta, changed = clean_user_content(old)
            local.urls_removed += meta["urls_removed"]
            local.rt_prefix_removed += meta["rt_prefix_removed"]
            local.hashtag_signs_removed += meta["hashtag_signs_removed"]
            local.mojibake_fixes += meta["mojibake_fixes"]
            local.parse_skipped += meta["parse_skipped"]
            if changed:
                turn["content"] = new
                changed_any = True
                local.samples_changed += 1
            break

    return data, changed_any, local


def add_stats(dst: CleanStats, src: CleanStats) -> None:
    dst.files_scanned += src.files_scanned
    dst.files_updated += src.files_updated
    dst.samples_scanned += src.samples_scanned
    dst.samples_changed += src.samples_changed
    dst.urls_removed += src.urls_removed
    dst.rt_prefix_removed += src.rt_prefix_removed
    dst.hashtag_signs_removed += src.hashtag_signs_removed
    dst.mojibake_fixes += src.mojibake_fixes
    dst.parse_skipped += src.parse_skipped


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Clean CrisisMMD JSON tweet text: URLs / RT prefixes / hashtag signs / mojibake."
    )
    parser.add_argument(
        "targets",
        nargs="+",
        help="JSON files and/or directories (directory: client_*.json + test*.json).",
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

    total = CleanStats()
    touched: List[Tuple[Path, CleanStats]] = []

    for path in files:
        new_data, changed, local = process_file(path)
        if changed:
            local.files_updated = 1
            touched.append((path, local))
        add_stats(total, local)

        if changed and not args.dry_run:
            with path.open("w", encoding="utf-8", newline="\n") as f:
                json.dump(new_data, f, ensure_ascii=False, indent=4)
                f.write("\n")

    print(f"Scanned files: {total.files_scanned}")
    print(f"Scanned samples: {total.samples_scanned}")
    print(f"Changed samples: {total.samples_changed}")
    print(f"URL removals: {total.urls_removed}")
    print(f"RT-prefix removals: {total.rt_prefix_removed}")
    print(f"Hashtag-sign removals: {total.hashtag_signs_removed}")
    print(f"Mojibake fixes: {total.mojibake_fixes}")
    if total.parse_skipped:
        print(f"Template-parse skipped: {total.parse_skipped}")

    if touched:
        print("Files with changes:")
        for p, s in touched:
            print(
                f"  {p}  (samples={s.samples_changed}, urls={s.urls_removed}, "
                f"rt={s.rt_prefix_removed}, #={s.hashtag_signs_removed}, mojibake={s.mojibake_fixes})"
            )

    if args.dry_run:
        print("Dry-run mode: no files were modified.")
    else:
        print(f"Updated files: {total.files_updated}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
