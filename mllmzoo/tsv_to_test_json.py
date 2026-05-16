#!/usr/bin/env python3
"""
Convert crisis-mmd official TSV split files to the JSON conversation format
used by client_0.json / client_1.json in partition-alpha0.5-clt10/.

The output format exactly matches the training JSON:
  {
    "id": "<image_id>",
    "image": "../../data/crisis-mmd/raw_data/data_image/...",
    "conversations": [
      {"role": "user",      "content": "<image>\nSelect the best answer...\nOptions:\n..."},
      {"role": "assistant", "content": "E"}
    ]
  }

Usage (run from the project root):
  # Humanitarian task (default, 8-class):
  python mllmzoo/tsv_to_test_json.py \\
      --tsv  crisismmd_datasplit_all/task_humanitarian_text_img_test.tsv \\
      --out  data1/crisis-mmd/minicpmv_data/test.json

  # Damage task (3-class):
  python mllmzoo/tsv_to_test_json.py \\
      --tsv  crisismmd_datasplit_all/task_damage_text_img_test.tsv \\
      --out  data1/crisis-mmd/minicpmv_data/test_damage.json \\
      --task damage

  # Informative task (2-class):
  python mllmzoo/tsv_to_test_json.py \\
      --tsv  crisismmd_datasplit_all/task_informative_text_img_test.tsv \\
      --out  data1/crisis-mmd/minicpmv_data/test_informative.json \\
      --task informative

  Then set in quick_mllm_fed_config.py:
    eval_data_path="./data1/crisis-mmd/minicpmv_data/test.json"
"""

import argparse
import csv
import html
import json
import os
import re


# ─── Task definitions ────────────────────────────────────────────────────────

TASK_CONFIGS = {
    "humanitarian": {
        "question": "What is the humanitarian category based on the image and text?",
        "options": [
            ("A", "affected_individuals"),
            ("B", "infrastructure_and_utility_damage"),
            ("C", "injured_or_dead_people"),
            ("D", "missing_or_found_people"),
            ("E", "rescue_volunteering_or_donation_effort"),
            ("F", "vehicle_damage"),
            ("G", "other_relevant_information"),
            ("H", "not_humanitarian"),
        ],
        # 'label' = randomly selected from text/image label (matches client_X.json)
        # Readme says: "label: randomly selected labels from text and image labels"
        "label_col": "label",
    },
    "damage": {
        "question": "What is the damage severity based on the image and text?",
        "options": [
            ("A", "severe_damage"),
            ("B", "mild_damage"),
            ("C", "little_or_no_damage"),
        ],
        "label_col": "label_image",   # damage task only has label_image
    },
    "informative": {
        "question": "Is this image and text informative about the crisis event?",
        "options": [
            ("A", "informative"),
            ("B", "not_informative"),
        ],
        "label_col": "label",
    },
}

URL_RE = re.compile(r"(?:https?://|www\.)\S+", flags=re.IGNORECASE)
RT_PREFIX_RE = re.compile(r"^\s*(?:RT\s+@[A-Za-z0-9_]{1,20}:?\s*)+", flags=re.IGNORECASE)
HASHTAG_SIGN_RE = re.compile(r"#")
GREEK_EXTENDED_RE = re.compile(r"[\u1F00-\u1FFF]+")
CONTROL_CHAR_RE = re.compile(r"[\u0000-\u0008\u000B-\u001F\u007F-\u009F]+")

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


def clean_tweet_text(tweet_text: str) -> str:
    text = html.unescape(tweet_text or "")

    for bad, good in MOJIBAKE_MAP.items():
        if bad in text:
            text = text.replace(bad, good)

    if re.search(r"[âÃÂ‚á¼á½]", text):
        try:
            repaired = text.encode("cp1252", errors="ignore").decode("utf-8", errors="ignore")
            if repaired:
                text = repaired
        except Exception:
            pass

    text = GREEK_EXTENDED_RE.sub("", text)
    text = CONTROL_CHAR_RE.sub(" ", text)
    text = URL_RE.sub("", text)
    text = RT_PREFIX_RE.sub("", text)
    text = HASHTAG_SIGN_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_user_content(tweet_text: str, cfg: dict) -> str:
    """Build the multiple-choice question prompt matching the training format."""
    options_str = "\n".join(f"({letter}) {label}" for letter, label in cfg["options"])
    return (
        f"<image>\n"
        f"Select the best answer to the following multiple-choice question "
        f"based on the text and image.\n"
        f"{tweet_text}\n"
        f"{cfg['question']}\n"
        f"Options:\n{options_str}\n"
        f"Answer with the option's letter from the given choices directly "
        f"and only give the best option. The best answer is: "
    )


def build_assistant_content(label: str, cfg: dict) -> str:
    """Return the single-letter answer expected by the current training split."""
    label = label.strip()
    for letter, name in cfg["options"]:
        if name == label:
            return letter
    # Fallback: return raw label if not found in option list
    return label


def convert(tsv_path: str, out_path: str, task: str, img_prefix: str, clean_text: bool = True) -> None:
    cfg = TASK_CONFIGS[task]
    label_col = cfg["label_col"]

    records = []
    skipped = 0

    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            raw_label = row.get(label_col, "").strip()
            if not raw_label:
                skipped += 1
                continue

            # Build image path (same relative prefix as training data)
            img_rel = row["image"].strip()          # e.g. data_image/hurricane_harvey/...
            img_path = os.path.join(img_prefix, img_rel).replace("\\", "/")

            tweet_text = row["tweet_text"].strip()
            if clean_text:
                tweet_text = clean_tweet_text(tweet_text)
            user_content  = build_user_content(tweet_text, cfg)
            asst_content  = build_assistant_content(raw_label, cfg)

            records.append({
                "id": row["image_id"].strip(),
                "image": img_path,
                "conversations": [
                    {"role": "user",      "content": user_content},
                    {"role": "assistant", "content": asst_content},
                ],
            })

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=4)

    print(f"✓ Converted {len(records)} samples → {out_path}")
    if skipped:
        print(f"  (skipped {skipped} rows with empty '{label_col}' column)")

    # Print label distribution
    from collections import Counter
    labels = [r["conversations"][1]["content"] for r in records]
    print("  Label distribution:")
    for label, cnt in sorted(Counter(labels).items(), key=lambda x: -x[1]):
        print(f"    {label}: {cnt}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert crisis-mmd TSV split to client_X.json conversation format"
    )
    parser.add_argument("--tsv",  required=True,
                        help="Input TSV file (e.g. crisismmd_datasplit_all/task_humanitarian_text_img_test.tsv)")
    parser.add_argument("--out",  required=True,
                        help="Output JSON path (e.g. data1/crisis-mmd/minicpmv_data/test.json)")
    parser.add_argument("--task", default="humanitarian",
                        choices=list(TASK_CONFIGS.keys()),
                        help="Which crisis-mmd task (default: humanitarian)")
    parser.add_argument("--img_prefix", default="../../data//crisis-mmd/raw_data",
                        help="Path prefix prepended to the image column value. "
                             "Should match what's in client_X.json. "
                             "Default: '../../data//crisis-mmd/raw_data'")
    parser.add_argument(
        "--no_clean_text",
        action="store_true",
        help="Disable tweet-text cleaning (URL/RT-prefix/#/mojibake).",
    )
    args = parser.parse_args()

    if not os.path.exists(args.tsv):
        print(f"ERROR: TSV file not found: {args.tsv}")
        return

    convert(args.tsv, args.out, args.task, args.img_prefix, clean_text=(not args.no_clean_text))


if __name__ == "__main__":
    main()
