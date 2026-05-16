import re
from collections import defaultdict
from typing import Dict, List, Optional


_ARTICLES = {"a", "an", "the"}
_NUMBER_MAP = {
    "none": "0",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}
_CONTRACTIONS = {
    "aint": "ain't",
    "arent": "aren't",
    "cant": "can't",
    "couldve": "could've",
    "couldnt": "couldn't",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "dont": "don't",
    "hadnt": "hadn't",
    "hasnt": "hasn't",
    "havent": "haven't",
    "hes": "he's",
    "im": "i'm",
    "isnt": "isn't",
    "itll": "it'll",
    "its": "it's",
    "ive": "i've",
    "lets": "let's",
    "mightnt": "mightn't",
    "mustnt": "mustn't",
    "shes": "she's",
    "shouldnt": "shouldn't",
    "thats": "that's",
    "theres": "there's",
    "theyre": "they're",
    "wasnt": "wasn't",
    "werent": "weren't",
    "whats": "what's",
    "wheres": "where's",
    "whos": "who's",
    "wont": "won't",
    "wouldnt": "wouldn't",
    "youre": "you're",
}
_IRREGULAR_PLURAL_MAP = {
    "men": "man",
    "women": "woman",
    "children": "child",
    "people": "person",
    "mice": "mouse",
    "geese": "goose",
    "teeth": "tooth",
    "feet": "foot",
}

_COMMA_BETWEEN_DIGITS_RE = re.compile(r"(?<=\d),(?=\d)")
_PERIOD_NOT_DECIMAL_RE = re.compile(r"(?<!\d)\.(?!\d)")
_SPACE_RE = re.compile(r"\s+")


def _normalize_plural_token(token: str) -> str:
    if not token:
        return token
    if token in _IRREGULAR_PLURAL_MAP:
        return _IRREGULAR_PLURAL_MAP[token]
    if token.isdigit() or len(token) <= 3:
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("ses") and len(token) > 4:
        return token[:-2]
    if token.endswith("xes") or token.endswith("zes") or token.endswith("ches") or token.endswith("shes"):
        return token[:-2]
    if token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def normalize_vqa_answer(text: Optional[str]) -> str:
    value = "" if text is None else str(text)
    value = value.strip().lower()
    value = value.replace("’", "'").replace("`", "'")
    value = _COMMA_BETWEEN_DIGITS_RE.sub("", value)
    value = _PERIOD_NOT_DECIMAL_RE.sub(" ", value)
    value = re.sub(r"[^\w\s']", " ", value)

    tokens = []
    for token in value.split():
        token = _CONTRACTIONS.get(token, token)
        token = _NUMBER_MAP.get(token, token)
        if token in _ARTICLES:
            continue
        token = _normalize_plural_token(token)
        if token:
            tokens.append(token)

    return _SPACE_RE.sub(" ", " ".join(tokens)).strip()


def compute_normalized_exact_match(prediction: Optional[str], references: List[str]) -> float:
    pred_norm = normalize_vqa_answer(prediction)
    if not pred_norm:
        return 0.0
    ref_norms = [normalize_vqa_answer(r) for r in references if str(r).strip()]
    if not ref_norms:
        return 0.0
    return 1.0 if pred_norm in set(ref_norms) else 0.0


def compute_vqa_soft_score(prediction: Optional[str], references: List[str]) -> float:
    """
    Approximate VQAv2 official score with multi-reference consensus:
    score = min(1, (#matched_human_answers) / 3).
    """
    pred_norm = normalize_vqa_answer(prediction)
    if not pred_norm:
        return 0.0
    ref_norms = [normalize_vqa_answer(r) for r in references if str(r).strip()]
    if not ref_norms:
        return 0.0
    match_count = sum(1 for ref in ref_norms if ref == pred_norm)
    return min(1.0, float(match_count) / 3.0)


def score_vqa_prediction(prediction: Optional[str], references: List[str]) -> Dict[str, float]:
    return {
        "normalized_exact_match": compute_normalized_exact_match(prediction, references),
        "vqa_score": compute_vqa_soft_score(prediction, references),
    }


def compute_vqa_metrics(
    predictions: List[str],
    references_list: List[List[str]],
    answer_types: Optional[List[str]] = None,
) -> Dict:
    if len(predictions) != len(references_list):
        raise ValueError(
            f"Length mismatch: predictions={len(predictions)} vs references_list={len(references_list)}"
        )

    total = len(predictions)
    if total == 0:
        return {
            "normalized_exact_match": 0.0,
            "vqa_score": 0.0,
            "empty_prediction_rate": 0.0,
            "num_samples": 0,
            "num_scored_samples": 0,
            "avg_reference_count": 0.0,
            "per_answer_type": {},
        }

    em_sum = 0.0
    vqa_sum = 0.0
    empty_count = 0
    scored_count = 0
    ref_count_sum = 0

    per_type_stats = defaultdict(lambda: {"count": 0, "em_sum": 0.0, "vqa_sum": 0.0})
    answer_types = answer_types or ["unknown"] * total

    for pred, refs, ans_type in zip(predictions, references_list, answer_types):
        refs = list(refs or [])
        ref_count_sum += len(refs)

        pred_text = "" if pred is None else str(pred).strip()
        if not pred_text:
            empty_count += 1

        scores = score_vqa_prediction(pred_text, refs)
        em_sum += scores["normalized_exact_match"]
        vqa_sum += scores["vqa_score"]
        if refs:
            scored_count += 1

        bucket = per_type_stats[str(ans_type) if ans_type is not None else "unknown"]
        bucket["count"] += 1
        bucket["em_sum"] += scores["normalized_exact_match"]
        bucket["vqa_sum"] += scores["vqa_score"]

    per_type = {}
    for ans_type, stats in sorted(per_type_stats.items(), key=lambda x: x[0]):
        count = max(1, int(stats["count"]))
        per_type[ans_type] = {
            "count": int(stats["count"]),
            "normalized_exact_match": round(float(stats["em_sum"]) / count, 4),
            "vqa_score": round(float(stats["vqa_sum"]) / count, 4),
        }

    return {
        "normalized_exact_match": round(em_sum / total, 4),
        "vqa_score": round(vqa_sum / total, 4),
        "empty_prediction_rate": round(float(empty_count) / total, 4),
        "num_samples": int(total),
        "num_scored_samples": int(scored_count),
        "avg_reference_count": round(float(ref_count_sum) / total, 4),
        "per_answer_type": per_type,
    }
