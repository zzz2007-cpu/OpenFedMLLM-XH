"""Temp script: convert TSV to test.json"""
import csv, json, os, sys

tsv_path = "crisismmd_datasplit_all/task_humanitarian_text_img_test.tsv"
out_path = "mllmzoo/test.json"
img_prefix = "../../data//crisis-mmd/raw_data"

OPTIONS = [
    ("A", "affected_individuals"),
    ("B", "infrastructure_and_utility_damage"),
    ("C", "injured_or_dead_people"),
    ("D", "missing_or_found_people"),
    ("E", "rescue_volunteering_or_donation_effort"),
    ("F", "vehicle_damage"),
    ("G", "other_relevant_information"),
    ("H", "not_humanitarian"),
]
QUESTION = "What is the humanitarian category based on the image and text?"
OPTIONS_STR = "\n".join(f"({l}) {n}" for l, n in OPTIONS)
PROMPT_SUFFIX = (
    f"\n{QUESTION}\nOptions:\n{OPTIONS_STR}\n"
    "Answer with the option's letter from the given choices directly "
    "and only give the best option. The best answer is: "
)

def to_asst(label):
    label = label.strip()
    for l, n in OPTIONS:
        if n == label:
            return l
    return label

records = []
with open(tsv_path, encoding="utf-8") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        raw = row.get("label_text_image", "").strip()
        if not raw:
            continue
        img = img_prefix + "/" + row["image"].strip()
        user = "<image>\nSelect the best answer to the following multiple-choice question based on the text and image.\n"
        user += row["tweet_text"].strip() + PROMPT_SUFFIX
        records.append({
            "id": row["image_id"].strip(),
            "image": img,
            "conversations": [
                {"role": "user",      "content": user},
                {"role": "assistant", "content": to_asst(raw)},
            ]
        })

os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

from collections import Counter
labels = [r["conversations"][1]["content"] for r in records]
print(f"Done: {len(records)} samples -> {out_path}")
for label, cnt in sorted(Counter(labels).items(), key=lambda x: -x[1]):
    print(f"  {label}: {cnt}")
