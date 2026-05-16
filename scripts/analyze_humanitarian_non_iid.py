#!/usr/bin/env python3
"""Analyze non-IID heterogeneity for humanitarian federated split."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gammaln


DEFAULT_LABEL_ORDER = [
    "affected_individuals",
    "infrastructure_and_utility_damage",
    "injured_or_dead_people",
    "missing_or_found_people",
    "rescue_volunteering_or_donation_effort",
    "vehicle_damage",
    "other_relevant_information",
    "not_humanitarian",
]

LETTER_TO_LABEL = {chr(ord("A") + i): label for i, label in enumerate(DEFAULT_LABEL_ORDER)}


@dataclass
class SampleRecord:
    client: str
    sample_id: str
    label: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze non-IID properties for humanitarian FL split.")
    parser.add_argument("--federated_dir", default="partition-alpha0.5-clt10", type=str)
    parser.add_argument(
        "--central_tsv",
        default="crisismmd_datasplit_all/task_humanitarian_text_img_train.tsv",
        type=str,
    )
    parser.add_argument("--out_dir", default="analysis/non_iid_humanitarian", type=str)
    parser.add_argument("--alpha_grid", default="0.05,0.1,0.2,0.3,0.5,0.8,1.0,2.0", type=str)
    parser.add_argument("--num_sim", default=2000, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--bootstrap", default=400, type=int)
    return parser.parse_args()


def normalize_label(text: str) -> str:
    text = (text or "").strip().lower()
    text = text.replace("-", "_").replace(" ", "_")
    text = re.sub(r"^\(\s*[a-h]\s*\)\s*", "", text)
    text = re.sub(r"[^a-z_]", "", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def extract_assistant_content(record: dict) -> str:
    convs = record.get("conversations") or record.get("messages") or []
    for msg in convs:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or msg.get("from") or "").strip().lower()
        if role in {"assistant", "gpt"}:
            return str(msg.get("content") or msg.get("value") or "")
    return ""


def extract_label_from_assistant(text: str, label_set: set[str]) -> str | None:
    if not text:
        return None

    letter_text = re.search(r"\(\s*([A-Ha-h])\s*\)\s*([A-Za-z_ ]+)", text)
    if letter_text:
        letter = letter_text.group(1).upper()
        tail = normalize_label(letter_text.group(2))
        if tail in label_set:
            return tail
        if letter in LETTER_TO_LABEL and LETTER_TO_LABEL[letter] in label_set:
            return LETTER_TO_LABEL[letter]

    compact = normalize_label(text)
    if compact in label_set:
        return compact

    letter_only = re.search(r"^\s*\(?\s*([A-Ha-h])\s*\)?\s*$", text.strip())
    if letter_only:
        label = LETTER_TO_LABEL.get(letter_only.group(1).upper())
        if label in label_set:
            return label

    any_letter = re.search(r"\(\s*([A-Ha-h])\s*\)", text)
    if any_letter:
        label = LETTER_TO_LABEL.get(any_letter.group(1).upper())
        if label in label_set:
            return label

    for label in sorted(label_set, key=len, reverse=True):
        if re.search(rf"\b{re.escape(label)}\b", compact):
            return label
    return None


def parse_alpha_grid(alpha_grid: str) -> List[float]:
    values = []
    for x in alpha_grid.split(","):
        x = x.strip()
        if not x:
            continue
        values.append(float(x))
    if not values:
        raise ValueError("alpha_grid is empty.")
    values = sorted(set(values))
    if any(v <= 0 for v in values):
        raise ValueError("alpha_grid must contain only positive values.")
    return values


def load_central_labels(path: Path) -> Dict[str, str]:
    id_to_label: Dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        required = {"image_id", "label_image"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"TSV missing required columns {required}, got {reader.fieldnames}")
        for row in reader:
            sample_id = str(row["image_id"]).strip()
            label = normalize_label(str(row["label_image"]))
            if not sample_id:
                continue
            if sample_id in id_to_label and id_to_label[sample_id] != label:
                raise ValueError(f"Duplicate sample_id with inconsistent label in TSV: {sample_id}")
            id_to_label[sample_id] = label
    if not id_to_label:
        raise ValueError("No data loaded from central TSV.")
    return id_to_label


def load_federated_samples(federated_dir: Path, label_set: set[str]) -> List[SampleRecord]:
    client_files = sorted(
        federated_dir.glob("client_*.json"),
        key=lambda p: int(re.search(r"client_(\d+)", p.stem).group(1)) if re.search(r"client_(\d+)", p.stem) else p.stem,
    )
    if not client_files:
        raise ValueError(f"No client_*.json files found in {federated_dir}")

    samples: List[SampleRecord] = []
    unknown_labels = []
    missing_id = []
    for client_path in client_files:
        client_name = client_path.stem
        records = json.loads(client_path.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise ValueError(f"{client_path} expected list JSON, got {type(records).__name__}")
        for rec in records:
            if not isinstance(rec, dict):
                continue
            sample_id = str(rec.get("id") or rec.get("image_id") or rec.get("guid") or "").strip()
            if not sample_id:
                missing_id.append({"client": client_name, "record": rec})
                continue
            assistant_text = extract_assistant_content(rec)
            label = extract_label_from_assistant(assistant_text, label_set)
            if label is None:
                unknown_labels.append(
                    {
                        "client": client_name,
                        "sample_id": sample_id,
                        "assistant_preview": assistant_text[:200].replace("\n", " "),
                    }
                )
                continue
            samples.append(SampleRecord(client=client_name, sample_id=sample_id, label=label))

    if missing_id:
        show = [x["client"] for x in missing_id[:5]]
        raise ValueError(f"Found records without sample id in federated data. Examples clients: {show}")
    if unknown_labels:
        show = unknown_labels[:10]
        msg = "\n".join(
            [
                f"client={x['client']} sample_id={x['sample_id']} assistant='{x['assistant_preview']}'"
                for x in show
            ]
        )
        raise ValueError(
            f"Unknown labels found in assistant answers: {len(unknown_labels)} entries. First examples:\n{msg}"
        )
    return samples


def ensure_consistency(samples: Sequence[SampleRecord], central_id_to_label: Dict[str, str]) -> None:
    fed_ids = [s.sample_id for s in samples]
    fed_id_set = set(fed_ids)
    if len(fed_id_set) != len(fed_ids):
        dup_n = len(fed_ids) - len(fed_id_set)
        raise ValueError(f"Duplicate sample ids found in federated data: {dup_n}")

    central_ids = set(central_id_to_label.keys())
    missing_in_fed = sorted(central_ids - fed_id_set)
    extra_in_fed = sorted(fed_id_set - central_ids)
    if missing_in_fed or extra_in_fed:
        raise ValueError(
            "Federated IDs and central train IDs mismatch. "
            f"missing_in_fed={len(missing_in_fed)}, extra_in_fed={len(extra_in_fed)}. "
            f"Examples missing={missing_in_fed[:5]} extra={extra_in_fed[:5]}"
        )

    mismatch = []
    for s in samples:
        if central_id_to_label[s.sample_id] != s.label:
            mismatch.append((s.sample_id, s.label, central_id_to_label[s.sample_id]))
            if len(mismatch) >= 10:
                break
    if mismatch:
        lines = [
            f"id={sid}, fed_label={fed}, central_label={cen}" for sid, fed, cen in mismatch
        ]
        raise ValueError("Label mismatch between federated assistant and central label_image. Examples:\n" + "\n".join(lines))


def build_count_matrix(samples: Sequence[SampleRecord], labels: Sequence[str]) -> Tuple[np.ndarray, List[str]]:
    client_names = sorted(set(s.client for s in samples), key=lambda x: int(re.search(r"(\d+)$", x).group(1)))
    client_to_idx = {c: i for i, c in enumerate(client_names)}
    label_to_idx = {l: i for i, l in enumerate(labels)}
    counts = np.zeros((len(client_names), len(labels)), dtype=np.int64)
    for s in samples:
        counts[client_to_idx[s.client], label_to_idx[s.label]] += 1
    return counts, client_names


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))


def compute_client_metrics(
    counts: np.ndarray,
    global_prob: np.ndarray,
) -> Dict[str, np.ndarray]:
    client_sizes = counts.sum(axis=1)
    client_probs = np.divide(
        counts,
        client_sizes[:, None],
        out=np.zeros_like(counts, dtype=float),
        where=client_sizes[:, None] > 0,
    )
    kl_vals = np.array([kl_divergence(client_probs[i], global_prob) for i in range(counts.shape[0])], dtype=float)
    top_sorted = np.sort(client_probs, axis=1)[:, ::-1]
    top1 = top_sorted[:, 0]
    top2 = top_sorted[:, 1]
    coverage = counts > 0
    missing_per_client = (~coverage).sum(axis=1)
    overexpr = client_probs - global_prob[None, :]
    return {
        "client_sizes": client_sizes,
        "client_probs": client_probs,
        "kl_vals": kl_vals,
        "top1": top1,
        "top2": top2,
        "coverage": coverage,
        "missing_per_client": missing_per_client,
        "dominant_flag": (top1 > 0.5),
        "overexpr": overexpr,
        "overexpr_idx": overexpr.argmax(axis=1),
        "overexpr_delta": overexpr.max(axis=1),
    }


def simulate_iid_kl(
    client_sizes: np.ndarray,
    global_prob: np.ndarray,
    num_sim: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n_clients = len(client_sizes)
    kl_null = np.zeros((num_sim, n_clients), dtype=float)
    for s in range(num_sim):
        sim_counts = np.vstack([rng.multinomial(int(n), global_prob) for n in client_sizes])
        sim_probs = sim_counts / client_sizes[:, None]
        kl_null[s] = np.array([kl_divergence(sim_probs[i], global_prob) for i in range(n_clients)], dtype=float)
    return kl_null


def gini_coefficient(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if np.any(x < 0):
        raise ValueError("gini input contains negative values.")
    if np.allclose(x.sum(), 0.0):
        return 0.0
    x_sorted = np.sort(x)
    n = x_sorted.size
    numer = 2.0 * np.sum((np.arange(1, n + 1) * x_sorted))
    denom = n * x_sorted.sum()
    return float(numer / denom - (n + 1) / n)


def cv(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    mean = float(np.mean(x))
    if mean == 0:
        return 0.0
    return float(np.std(x, ddof=1) / mean)


def simulate_quantity_null(total_n: int, n_clients: int, num_sim: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    probs = np.full(n_clients, 1.0 / n_clients)
    cv_null = np.zeros(num_sim, dtype=float)
    gini_null = np.zeros(num_sim, dtype=float)
    for i in range(num_sim):
        sizes = rng.multinomial(total_n, probs)
        cv_null[i] = cv(sizes)
        gini_null[i] = gini_coefficient(sizes)
    return cv_null, gini_null


def dirichlet_multinomial_loglik(alpha: float, class_counts: np.ndarray) -> float:
    # class_counts: [n_classes, n_clients]
    if alpha <= 0:
        return -np.inf
    n_classes, n_clients = class_counts.shape
    ll = 0.0
    for k in range(n_classes):
        nk = int(class_counts[k].sum())
        ll += gammaln(n_clients * alpha) - gammaln(nk + n_clients * alpha)
        ll += float(np.sum(gammaln(class_counts[k] + alpha) - gammaln(alpha)))
    return float(ll)


def fit_alpha_mle(class_counts: np.ndarray, alpha_candidates: np.ndarray) -> Tuple[float, float, Tuple[float, float], np.ndarray]:
    lls = np.array([dirichlet_multinomial_loglik(a, class_counts) for a in alpha_candidates], dtype=float)
    idx = int(np.argmax(lls))
    best_alpha = float(alpha_candidates[idx])
    best_ll = float(lls[idx])
    cutoff = best_ll - 1.92  # ~95% profile likelihood interval (1 parameter)
    valid = alpha_candidates[lls >= cutoff]
    ci = (float(valid.min()), float(valid.max())) if valid.size else (best_alpha, best_alpha)
    return best_alpha, best_ll, ci, lls


def metrics_from_counts(counts: np.ndarray, global_prob: np.ndarray) -> Dict[str, float]:
    client_sizes = counts.sum(axis=1)
    probs = np.divide(
        counts,
        client_sizes[:, None],
        out=np.zeros_like(counts, dtype=float),
        where=client_sizes[:, None] > 0,
    )
    mean_kl = float(np.mean([kl_divergence(probs[i], global_prob) for i in range(counts.shape[0])]))
    missing_rate = float(np.mean(counts == 0))
    dominant_rate = float(np.mean(np.max(probs, axis=1) > 0.5))
    return {
        "mean_kl": mean_kl,
        "missing_rate": missing_rate,
        "dominant_rate": dominant_rate,
    }


def simulate_dirichlet_partition(global_counts: np.ndarray, n_clients: int, alpha: float, rng: np.random.Generator) -> np.ndarray:
    n_classes = len(global_counts)
    sim = np.zeros((n_clients, n_classes), dtype=np.int64)
    for k, nk in enumerate(global_counts):
        props = rng.dirichlet(np.full(n_clients, alpha, dtype=float))
        alloc = rng.multinomial(int(nk), props)
        sim[:, k] = alloc
    return sim


def score_alpha_grid(
    counts: np.ndarray,
    global_counts: np.ndarray,
    global_prob: np.ndarray,
    alpha_grid: Sequence[float],
    num_sim: int,
    seed: int,
) -> Tuple[float, Dict[float, dict], dict]:
    rng = np.random.default_rng(seed)
    n_clients = counts.shape[0]
    observed = metrics_from_counts(counts, global_prob)
    stats_by_alpha: Dict[float, dict] = {}
    best_alpha = None
    best_score = None

    for alpha in alpha_grid:
        sims = np.zeros((num_sim, 3), dtype=float)
        for i in range(num_sim):
            sim_counts = simulate_dirichlet_partition(global_counts, n_clients, alpha, rng)
            m = metrics_from_counts(sim_counts, global_prob)
            sims[i, 0] = m["mean_kl"]
            sims[i, 1] = m["missing_rate"]
            sims[i, 2] = m["dominant_rate"]

        mu = sims.mean(axis=0)
        sd = sims.std(axis=0, ddof=1)
        sd = np.where(sd < 1e-9, 1e-9, sd)
        obs_vec = np.array([observed["mean_kl"], observed["missing_rate"], observed["dominant_rate"]], dtype=float)
        z = (obs_vec - mu) / sd
        score = float(np.sqrt(np.sum(z ** 2)))
        stats_by_alpha[alpha] = {
            "sim_mean_kl_mean": float(mu[0]),
            "sim_missing_rate_mean": float(mu[1]),
            "sim_dominant_rate_mean": float(mu[2]),
            "sim_mean_kl_std": float(sd[0]),
            "sim_missing_rate_std": float(sd[1]),
            "sim_dominant_rate_std": float(sd[2]),
            "score": score,
        }
        if best_score is None or score < best_score:
            best_score = score
            best_alpha = float(alpha)

    assert best_alpha is not None
    return best_alpha, stats_by_alpha, observed


def bootstrap_alpha_ci_grid(
    best_alpha: float,
    global_counts: np.ndarray,
    global_prob: np.ndarray,
    alpha_grid: Sequence[float],
    stats_by_alpha: Dict[float, dict],
    n_clients: int,
    n_boot: int,
    seed: int,
) -> Tuple[Tuple[float, float], np.ndarray]:
    rng = np.random.default_rng(seed)
    vals = []
    obs_mu = np.array(
        [
            [stats_by_alpha[a]["sim_mean_kl_mean"], stats_by_alpha[a]["sim_missing_rate_mean"], stats_by_alpha[a]["sim_dominant_rate_mean"]]
            for a in alpha_grid
        ],
        dtype=float,
    )
    obs_sd = np.array(
        [
            [stats_by_alpha[a]["sim_mean_kl_std"], stats_by_alpha[a]["sim_missing_rate_std"], stats_by_alpha[a]["sim_dominant_rate_std"]]
            for a in alpha_grid
        ],
        dtype=float,
    )
    obs_sd = np.where(obs_sd < 1e-9, 1e-9, obs_sd)

    for _ in range(n_boot):
        sim_counts = simulate_dirichlet_partition(global_counts, n_clients, best_alpha, rng)
        m = metrics_from_counts(sim_counts, global_prob)
        vec = np.array([m["mean_kl"], m["missing_rate"], m["dominant_rate"]], dtype=float)[None, :]
        z = (vec - obs_mu) / obs_sd
        scores = np.sqrt(np.sum(z ** 2, axis=1))
        vals.append(float(alpha_grid[int(np.argmin(scores))]))
    vals_arr = np.array(vals, dtype=float)
    ci = (float(np.percentile(vals_arr, 2.5)), float(np.percentile(vals_arr, 97.5)))
    return ci, vals_arr


def classify_strength(best_alpha: float, mean_kl_z: float) -> str:
    if best_alpha >= 1.0 and mean_kl_z < 2.0:
        return "weak"
    if best_alpha <= 0.3 or mean_kl_z >= 4.0:
        return "strong"
    return "medium"


def classify_non_iid_type(label_skew: bool, quantity_skew: bool) -> str:
    if label_skew and quantity_skew:
        return "mixed"
    if label_skew:
        return "label skew"
    if quantity_skew:
        return "quantity skew"
    return "near-IID"


def format_paper_conclusion(summary: dict) -> str:
    q1 = "Yes" if summary["dirichlet_consistent"] else "No"
    lines = [
        "Non-IID Conclusion (Humanitarian, partition-alpha0.5-clt10)",
        f"1) Dirichlet-based split?: {q1}",
        (
            "   Evidence: avg class coverage per class="
            f"{summary['dirichlet_evidence']['mean_class_coverage_clients']:.2f}/10, "
            f"non-zero ratio={summary['dirichlet_evidence']['nonzero_cell_ratio']:.3f}, "
            f"best alpha(grid)={summary['alpha_grid_fit']['best_alpha']:.3f}."
        ),
        (
            "2) Estimated alpha range: "
            f"grid-best={summary['alpha_grid_fit']['best_alpha']:.3f}, "
            f"bootstrap 95% CI=[{summary['alpha_grid_fit']['bootstrap_ci_95'][0]:.3f}, "
            f"{summary['alpha_grid_fit']['bootstrap_ci_95'][1]:.3f}], "
            f"continuous MLE={summary['alpha_mle']['best_alpha']:.3f} "
            f"(profile 95% CI [{summary['alpha_mle']['profile_ci_95'][0]:.3f}, "
            f"{summary['alpha_mle']['profile_ci_95'][1]:.3f}])."
        ),
        (
            "3) Non-IID type: "
            f"{summary['non_iid_type']} "
            f"(label_skew={summary['label_skew_significant']}, "
            f"quantity_skew={summary['quantity_skew_significant']})."
        ),
        (
            "4) Non-IID intensity: "
            f"{summary['non_iid_strength']} "
            f"(mean-KL z={summary['kl_summary']['mean_kl_zscore_vs_iid']:.2f})."
        ),
    ]
    return "\n".join(lines)


def save_csv(path: Path, header: Sequence[str], rows: Iterable[Sequence]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    alpha_grid = parse_alpha_grid(args.alpha_grid)

    central_path = Path(args.central_tsv)
    fed_dir = Path(args.federated_dir)
    central_id_to_label = load_central_labels(central_path)

    central_label_set = set(central_id_to_label.values())
    if len(central_label_set) != 8:
        raise ValueError(f"Expected 8 labels from central TSV, got {len(central_label_set)}: {sorted(central_label_set)}")

    if set(DEFAULT_LABEL_ORDER) == central_label_set:
        labels = list(DEFAULT_LABEL_ORDER)
    else:
        labels = sorted(central_label_set)
    label_to_idx = {l: i for i, l in enumerate(labels)}

    samples = load_federated_samples(fed_dir, set(labels))
    ensure_consistency(samples, central_id_to_label)

    counts, client_names = build_count_matrix(samples, labels)
    n_clients, n_labels = counts.shape
    total_fed = int(counts.sum())
    global_counts = np.zeros(n_labels, dtype=np.int64)
    for label in central_id_to_label.values():
        global_counts[label_to_idx[label]] += 1
    global_prob = global_counts / global_counts.sum()

    if total_fed != int(global_counts.sum()):
        raise ValueError(f"Total federated samples {total_fed} != central total {int(global_counts.sum())}")
    if not np.all(counts.sum(axis=0) == global_counts):
        raise ValueError("Client-wise class counts do not sum to global class counts.")

    metrics = compute_client_metrics(counts, global_prob)
    client_sizes = metrics["client_sizes"]
    client_probs = metrics["client_probs"]
    kl_vals = metrics["kl_vals"]

    # Step 3: KL and IID null
    iid_rng = np.random.default_rng(args.seed + 101)
    kl_null = simulate_iid_kl(client_sizes, global_prob, args.num_sim, iid_rng)
    kl_null_mean_per_client = kl_null.mean(axis=0)
    kl_null_std_per_client = kl_null.std(axis=0, ddof=1)
    kl_z = (kl_vals - kl_null_mean_per_client) / np.where(kl_null_std_per_client < 1e-12, 1e-12, kl_null_std_per_client)
    kl_p = (1 + np.sum(kl_null >= kl_vals[None, :], axis=0)) / (args.num_sim + 1)

    obs_mean_kl = float(np.mean(kl_vals))
    null_mean_kl_samples = kl_null.mean(axis=1)
    null_mean_kl_mu = float(np.mean(null_mean_kl_samples))
    null_mean_kl_sd = float(np.std(null_mean_kl_samples, ddof=1))
    mean_kl_z = float((obs_mean_kl - null_mean_kl_mu) / (null_mean_kl_sd if null_mean_kl_sd > 1e-12 else 1e-12))
    mean_kl_p = float((1 + np.sum(null_mean_kl_samples >= obs_mean_kl)) / (args.num_sim + 1))

    # Step 4: coverage / dominance / over-expression
    coverage = metrics["coverage"]
    missing_per_client = metrics["missing_per_client"]
    missing_clients_per_label = (~coverage).sum(axis=0)
    class_coverage_clients = coverage.sum(axis=0)
    dominant_flag = metrics["dominant_flag"]
    top1 = metrics["top1"]
    top2 = metrics["top2"]
    overexpr_idx = metrics["overexpr_idx"]
    overexpr_delta = metrics["overexpr_delta"]

    # quantity skew
    obs_cv = cv(client_sizes)
    obs_gini = gini_coefficient(client_sizes)
    q_rng = np.random.default_rng(args.seed + 202)
    cv_null, gini_null = simulate_quantity_null(total_fed, n_clients, args.num_sim, q_rng)
    p_cv = float((1 + np.sum(cv_null >= obs_cv)) / (args.num_sim + 1))
    p_gini = float((1 + np.sum(gini_null >= obs_gini)) / (args.num_sim + 1))

    # Step 5: Dirichlet evidence and alpha
    class_counts = counts.T
    alpha_candidates = np.exp(np.linspace(np.log(0.02), np.log(5.0), 500))
    best_alpha_mle, best_ll, profile_ci, lls = fit_alpha_mle(class_counts, alpha_candidates)

    # bootstrap for continuous MLE alpha
    boot_rng = np.random.default_rng(args.seed + 303)
    boot_alpha_mle = []
    n_classes = class_counts.shape[0]
    for _ in range(args.bootstrap):
        idx = boot_rng.integers(0, n_classes, size=n_classes)
        a, _, _, _ = fit_alpha_mle(class_counts[idx], alpha_candidates)
        boot_alpha_mle.append(a)
    boot_alpha_mle = np.array(boot_alpha_mle, dtype=float)
    alpha_mle_boot_ci = (
        float(np.percentile(boot_alpha_mle, 2.5)),
        float(np.percentile(boot_alpha_mle, 97.5)),
    )

    best_alpha_grid, alpha_grid_stats, observed_grid_stats = score_alpha_grid(
        counts=counts,
        global_counts=global_counts,
        global_prob=global_prob,
        alpha_grid=alpha_grid,
        num_sim=args.num_sim,
        seed=args.seed + 404,
    )
    alpha_grid_boot_ci, alpha_grid_boot_vals = bootstrap_alpha_ci_grid(
        best_alpha=best_alpha_grid,
        global_counts=global_counts,
        global_prob=global_prob,
        alpha_grid=alpha_grid,
        stats_by_alpha=alpha_grid_stats,
        n_clients=n_clients,
        n_boot=args.bootstrap,
        seed=args.seed + 505,
    )

    class_client_share = class_counts / global_counts[:, None]
    nonzero_cell_ratio = float(np.mean(class_client_share > 0))
    unique_positive_per_class = [int(np.unique(np.round(r[r > 0], 4)).size) for r in class_client_share]
    mean_class_coverage_clients = float(np.mean(class_coverage_clients))

    # Step 6: classify
    label_skew_significant = bool(mean_kl_p < 0.05 and obs_mean_kl > null_mean_kl_mu)
    quantity_skew_significant = bool((p_cv < 0.05) or (p_gini < 0.05))
    non_iid_type = classify_non_iid_type(label_skew_significant, quantity_skew_significant)
    non_iid_strength = classify_strength(best_alpha_grid, mean_kl_z)
    dirichlet_consistent = bool(
        mean_class_coverage_clients >= 2.0
        and nonzero_cell_ratio >= 0.35
        and 0.2 <= best_alpha_grid <= 0.8
    )

    # seed stability check
    stability_seeds = [args.seed, args.seed + 11, args.seed + 29]
    stability_alpha = []
    stability_strength = []
    for sd in stability_seeds:
        ba, _, _ = score_alpha_grid(
            counts=counts,
            global_counts=global_counts,
            global_prob=global_prob,
            alpha_grid=alpha_grid,
            num_sim=max(300, args.num_sim // 3),
            seed=sd + 404,
        )
        stability_alpha.append(float(ba))
        stability_strength.append(classify_strength(float(ba), mean_kl_z))

    summary = {
        "inputs": {
            "federated_dir": str(fed_dir),
            "central_tsv": str(central_path),
            "out_dir": str(out_dir),
            "alpha_grid": alpha_grid,
            "num_sim": args.num_sim,
            "seed": args.seed,
            "bootstrap": args.bootstrap,
        },
        "data_check": {
            "n_clients": int(n_clients),
            "n_labels": int(n_labels),
            "federated_total": int(total_fed),
            "central_total": int(global_counts.sum()),
            "id_match_exact": True,
            "label_match_exact": True,
        },
        "global_distribution": [
            {"label": labels[i], "count": int(global_counts[i]), "prob": float(global_prob[i])}
            for i in range(n_labels)
        ],
        "client_size": {client_names[i]: int(client_sizes[i]) for i in range(n_clients)},
        "kl_summary": {
            "per_client_kl": {client_names[i]: float(kl_vals[i]) for i in range(n_clients)},
            "mean": float(np.mean(kl_vals)),
            "var": float(np.var(kl_vals, ddof=1)),
            "min": float(np.min(kl_vals)),
            "max": float(np.max(kl_vals)),
            "mean_kl_iid_null_mean": null_mean_kl_mu,
            "mean_kl_iid_null_std": null_mean_kl_sd,
            "mean_kl_zscore_vs_iid": mean_kl_z,
            "mean_kl_pvalue_vs_iid": mean_kl_p,
            "iid_95_interval_for_mean_kl": [
                float(np.percentile(null_mean_kl_samples, 2.5)),
                float(np.percentile(null_mean_kl_samples, 97.5)),
            ],
        },
        "coverage_and_dominance": {
            "all_labels_covered_client_count": int(np.sum(np.all(coverage, axis=1))),
            "missing_labels_per_client": {client_names[i]: int(missing_per_client[i]) for i in range(n_clients)},
            "missing_clients_per_label": {labels[i]: int(missing_clients_per_label[i]) for i in range(n_labels)},
            "dominant_client_count_top1_gt_0_5": int(np.sum(dominant_flag)),
            "dominant_client_ratio_top1_gt_0_5": float(np.mean(dominant_flag)),
            "top1_ratio_per_client": {client_names[i]: float(top1[i]) for i in range(n_clients)},
            "top2_ratio_per_client": {client_names[i]: float(top2[i]) for i in range(n_clients)},
            "overexpressed_label_per_client": {
                client_names[i]: {
                    "label": labels[int(overexpr_idx[i])],
                    "delta_prob": float(overexpr_delta[i]),
                }
                for i in range(n_clients)
            },
        },
        "quantity_skew": {
            "client_size_cv": float(obs_cv),
            "client_size_gini": float(obs_gini),
            "cv_null_mean": float(np.mean(cv_null)),
            "gini_null_mean": float(np.mean(gini_null)),
            "cv_pvalue": p_cv,
            "gini_pvalue": p_gini,
        },
        "alpha_mle": {
            "best_alpha": float(best_alpha_mle),
            "profile_ci_95": [float(profile_ci[0]), float(profile_ci[1])],
            "bootstrap_ci_95": [float(alpha_mle_boot_ci[0]), float(alpha_mle_boot_ci[1])],
            "best_loglik": float(best_ll),
        },
        "alpha_grid_fit": {
            "best_alpha": float(best_alpha_grid),
            "bootstrap_ci_95": [float(alpha_grid_boot_ci[0]), float(alpha_grid_boot_ci[1])],
            "observed_stats": observed_grid_stats,
            "per_alpha_stats": alpha_grid_stats,
        },
        "dirichlet_evidence": {
            "class_coverage_clients": {labels[i]: int(class_coverage_clients[i]) for i in range(n_labels)},
            "mean_class_coverage_clients": mean_class_coverage_clients,
            "nonzero_cell_ratio": nonzero_cell_ratio,
            "unique_positive_proportion_count_per_class": {
                labels[i]: unique_positive_per_class[i] for i in range(n_labels)
            },
        },
        "label_skew_significant": label_skew_significant,
        "quantity_skew_significant": quantity_skew_significant,
        "non_iid_type": non_iid_type,
        "non_iid_strength": non_iid_strength,
        "dirichlet_consistent": dirichlet_consistent,
        "stability_check": {
            "seeds": stability_seeds,
            "best_alpha_list": stability_alpha,
            "best_alpha_range": [float(min(stability_alpha)), float(max(stability_alpha))],
            "strength_list": stability_strength,
        },
    }

    # Write CSVs
    save_csv(
        out_dir / "global_distribution.csv",
        ["label", "count", "prob"],
        [[labels[i], int(global_counts[i]), float(global_prob[i])] for i in range(n_labels)],
    )
    save_csv(
        out_dir / "client_distribution.csv",
        ["client", "label", "count", "prob"],
        [
            [client_names[c], labels[k], int(counts[c, k]), float(client_probs[c, k])]
            for c in range(n_clients)
            for k in range(n_labels)
        ],
    )
    save_csv(
        out_dir / "client_summary.csv",
        [
            "client",
            "client_size",
            "kl",
            "kl_zscore_vs_iid",
            "kl_pvalue_vs_iid",
            "missing_label_count",
            "top1_ratio",
            "top2_ratio",
            "dominant_top1_gt_0_5",
            "overexpressed_label",
            "overexpressed_delta",
        ],
        [
            [
                client_names[i],
                int(client_sizes[i]),
                float(kl_vals[i]),
                float(kl_z[i]),
                float(kl_p[i]),
                int(missing_per_client[i]),
                float(top1[i]),
                float(top2[i]),
                bool(dominant_flag[i]),
                labels[int(overexpr_idx[i])],
                float(overexpr_delta[i]),
            ]
            for i in range(n_clients)
        ],
    )
    save_csv(
        out_dir / "label_coverage.csv",
        ["label", "covered_clients", "missing_clients"],
        [[labels[i], int(class_coverage_clients[i]), int(missing_clients_per_label[i])] for i in range(n_labels)],
    )
    save_csv(
        out_dir / "alpha_grid_fit.csv",
        [
            "alpha",
            "score",
            "sim_mean_kl_mean",
            "sim_mean_kl_std",
            "sim_missing_rate_mean",
            "sim_missing_rate_std",
            "sim_dominant_rate_mean",
            "sim_dominant_rate_std",
        ],
        [
            [
                float(a),
                float(alpha_grid_stats[a]["score"]),
                float(alpha_grid_stats[a]["sim_mean_kl_mean"]),
                float(alpha_grid_stats[a]["sim_mean_kl_std"]),
                float(alpha_grid_stats[a]["sim_missing_rate_mean"]),
                float(alpha_grid_stats[a]["sim_missing_rate_std"]),
                float(alpha_grid_stats[a]["sim_dominant_rate_mean"]),
                float(alpha_grid_stats[a]["sim_dominant_rate_std"]),
            ]
            for a in alpha_grid
        ],
    )

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with (out_dir / "paper_conclusion.txt").open("w", encoding="utf-8") as f:
        f.write(format_paper_conclusion(summary) + "\n")

    # Step 7 figures
    plt.figure(figsize=(11, 5))
    im = plt.imshow(client_probs, aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=float(np.max(client_probs)))
    plt.colorbar(im, fraction=0.046, pad=0.04, label="P(label|client)")
    plt.xticks(np.arange(n_labels), labels, rotation=35, ha="right")
    plt.yticks(np.arange(n_clients), client_names)
    plt.title("Figure 1. Client Label Distribution Heatmap")
    plt.tight_layout()
    plt.savefig(out_dir / "fig1_client_label_distribution_heatmap.png", dpi=220)
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(n_clients)
    ax.bar(x, kl_vals, color="#2f6f9f", alpha=0.9, label="Observed KL(Pc||G)")
    ax.axhline(null_mean_kl_mu, color="#d1495b", linestyle="--", linewidth=1.8, label="IID null mean")
    low, high = np.percentile(null_mean_kl_samples, [2.5, 97.5])
    ax.axhspan(low, high, color="#edae49", alpha=0.25, label="IID null 95% interval (mean KL)")
    ax.set_xticks(x)
    ax.set_xticklabels(client_names, rotation=0)
    ax.set_ylabel("KL Divergence")
    ax.set_title("Figure 2. KL by Client with IID Baseline")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "fig2_kl_with_iid_baseline.png", dpi=220)
    plt.close(fig)

    plt.figure(figsize=(10, 4.8))
    im2 = plt.imshow(class_client_share, aspect="auto", cmap="OrRd", vmin=0.0, vmax=float(np.max(class_client_share)))
    plt.colorbar(im2, fraction=0.046, pad=0.04, label="Share of class assigned to client")
    plt.yticks(np.arange(n_labels), labels)
    plt.xticks(np.arange(n_clients), client_names)
    plt.title("Figure 3. Class-Client Proportion Heatmap")
    plt.tight_layout()
    plt.savefig(out_dir / "fig3_class_client_share_heatmap.png", dpi=220)
    plt.close()

    print(f"Analysis completed. Outputs saved to: {out_dir.resolve()}")
    print(f"Best alpha (grid): {best_alpha_grid:.3f}; bootstrap CI95: [{alpha_grid_boot_ci[0]:.3f}, {alpha_grid_boot_ci[1]:.3f}]")
    print(f"Best alpha (MLE): {best_alpha_mle:.3f}; profile CI95: [{profile_ci[0]:.3f}, {profile_ci[1]:.3f}]")
    print(f"Non-IID type: {non_iid_type}; strength: {non_iid_strength}; dirichlet_consistent: {dirichlet_consistent}")


if __name__ == "__main__":
    main()
