#!/usr/bin/env python3
"""
VQAv2 训练集图像级 Supercategory-Dirichlet Non-IID 划分脚本。

关键点：
1) 最小划分单元是 image_id（同图像全部 QA 必须同客户端）
2) 用 COCO instances 给每图定义“主导 supercategory”
3) 按主导 supercategory 分组后，组内按 Dirichlet(alpha) 分到多客户端
4) 后处理修复最小图片数与最小 supercategory 覆盖约束
5) 输出每个 alpha 的联邦客户端 JSON 与统计摘要
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import numpy as np

NO_INSTANCE_SUPERCAT = "__no_instance__"


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_vqa_questions(path: Path) -> dict:
    data = load_json(path)
    if "questions" not in data or not isinstance(data["questions"], list):
        raise ValueError(f"Invalid VQAv2 questions format: {path}")
    return data


def load_vqa_annotations(path: Path) -> dict:
    data = load_json(path)
    if "annotations" not in data or not isinstance(data["annotations"], list):
        raise ValueError(f"Invalid VQAv2 annotations format: {path}")
    return data


def load_coco_instances(path: Path) -> dict:
    data = load_json(path)
    if "annotations" not in data or "categories" not in data:
        raise ValueError(f"Invalid COCO instances format: {path}")
    return data


def build_vqa_indices(
    questions: List[dict],
    annotations: List[dict],
) -> Tuple[Dict[int, List[dict]], Dict[int, List[dict]], Set[int], Set[int], Set[int], Set[int]]:
    """构建 image_id->samples 索引，并返回 questions/annotations 的 qid 与 image_id 集合。"""
    questions_by_image: Dict[int, List[dict]] = defaultdict(list)
    annotations_by_image: Dict[int, List[dict]] = defaultdict(list)

    qids_q: Set[int] = set()
    qids_a: Set[int] = set()
    image_ids_q: Set[int] = set()
    image_ids_a: Set[int] = set()

    dup_q = 0
    dup_a = 0

    for q in questions:
        if "image_id" not in q or "question_id" not in q:
            continue
        iid = int(q["image_id"])
        qid = int(q["question_id"])
        questions_by_image[iid].append(q)
        image_ids_q.add(iid)
        if qid in qids_q:
            dup_q += 1
        qids_q.add(qid)

    for ann in annotations:
        if "image_id" not in ann or "question_id" not in ann:
            continue
        iid = int(ann["image_id"])
        qid = int(ann["question_id"])
        annotations_by_image[iid].append(ann)
        image_ids_a.add(iid)
        if qid in qids_a:
            dup_a += 1
        qids_a.add(qid)

    if dup_q > 0:
        logging.warning("Questions contain duplicated question_id entries: %d", dup_q)
    if dup_a > 0:
        logging.warning("Annotations contain duplicated question_id entries: %d", dup_a)

    return questions_by_image, annotations_by_image, qids_q, qids_a, image_ids_q, image_ids_a


def build_image_supercategory_stats(
    coco_instances_data: dict,
    target_image_ids: Set[int],
) -> Tuple[Dict[int, Counter], Dict[int, str], Counter]:
    """
    为目标图像构建 supercategory 实例计数。

    返回：
    - image_supercat_counts: image_id -> Counter(supercategory -> instance_count)
    - category_id_to_supercat: category_id -> supercategory
    - unknown_category_counter: COCO 中 category_id 缺失映射时的计数
    """
    category_id_to_supercat: Dict[int, str] = {}
    for cat in coco_instances_data.get("categories", []):
        if "id" not in cat:
            continue
        category_id_to_supercat[int(cat["id"])] = str(cat.get("supercategory", "unknown"))

    image_supercat_counts: Dict[int, Counter] = defaultdict(Counter)
    unknown_category_counter = Counter()

    target = target_image_ids
    for ann in coco_instances_data.get("annotations", []):
        if "image_id" not in ann or "category_id" not in ann:
            continue
        iid = int(ann["image_id"])
        if iid not in target:
            continue

        cid = int(ann["category_id"])
        supercat = category_id_to_supercat.get(cid)
        if supercat is None:
            supercat = "unknown"
            unknown_category_counter[cid] += 1

        image_supercat_counts[iid][supercat] += 1

    return image_supercat_counts, category_id_to_supercat, unknown_category_counter


def assign_dominant_supercategory(
    image_ids: Iterable[int],
    image_supercat_counts: Dict[int, Counter],
) -> Dict[int, str]:
    """
    为每张图像指定单一“主导 supercategory”。

    规则：
    1) 实例数最大者为主导
    2) 若并列，采用稳定 tie-break：supercategory 名称字典序最小
    3) 若图像无实例标注，记为 NO_INSTANCE_SUPERCAT
    """
    dominant = {}
    for iid in image_ids:
        c = image_supercat_counts.get(iid)
        if not c:
            dominant[iid] = NO_INSTANCE_SUPERCAT
            continue
        max_count = max(c.values())
        candidates = [k for k, v in c.items() if v == max_count]
        dominant[iid] = min(candidates)
    return dominant


def group_images_by_dominant_supercategory(
    dominant_supercat_by_image: Dict[int, str],
) -> Dict[str, List[int]]:
    groups: Dict[str, List[int]] = defaultdict(list)
    for iid, sc in dominant_supercat_by_image.items():
        groups[sc].append(iid)
    for sc in groups:
        groups[sc].sort()
    return groups


def dirichlet_partition_by_supercategory(
    groups: Dict[str, List[int]],
    num_clients: int,
    alpha: float,
    rng: np.random.Generator,
) -> List[Set[int]]:
    """
    对每个 supercategory 的图像组独立采样 Dirichlet 分布并分配。

    说明：
    - 每组一条 p^(s) ~ Dir(alpha * 1)
    - 再用 multinomial 将该组图像数映射为每个客户端的整数配额
    - 图像一旦分配，整图像全部 QA 一起归属该客户端
    """
    client_images = [set() for _ in range(num_clients)]

    for sc in sorted(groups.keys()):
        image_ids = list(groups[sc])
        if not image_ids:
            continue

        rng.shuffle(image_ids)
        probs = rng.dirichlet(np.full(num_clients, alpha, dtype=np.float64))
        counts = rng.multinomial(len(image_ids), probs)

        start = 0
        for cid, cnt in enumerate(counts.tolist()):
            if cnt <= 0:
                continue
            end = start + cnt
            client_images[cid].update(image_ids[start:end])
            start = end

    return client_images


def _build_client_supercat_index(
    client_images: List[Set[int]],
    dominant_supercat_by_image: Dict[int, str],
) -> List[Dict[str, Set[int]]]:
    idx: List[Dict[str, Set[int]]] = [defaultdict(set) for _ in range(len(client_images))]
    for cid, imgs in enumerate(client_images):
        for iid in imgs:
            idx[cid][dominant_supercat_by_image[iid]].add(iid)
    return idx


def _move_one_image(
    image_id: int,
    src: int,
    dst: int,
    client_images: List[Set[int]],
    client_sc_idx: List[Dict[str, Set[int]]],
    dominant_supercat_by_image: Dict[int, str],
) -> None:
    if image_id not in client_images[src]:
        return

    sc = dominant_supercat_by_image[image_id]

    client_images[src].remove(image_id)
    src_sc_set = client_sc_idx[src].get(sc)
    if src_sc_set is not None:
        src_sc_set.remove(image_id)
        if not src_sc_set:
            del client_sc_idx[src][sc]

    client_images[dst].add(image_id)
    client_sc_idx[dst][sc].add(image_id)


def _select_donor_for_supercat(
    target_sc: str,
    dst: int,
    client_images: List[Set[int]],
    client_sc_idx: List[Dict[str, Set[int]]],
    effective_min_images: int,
    min_supercats_per_client: int,
) -> int | None:
    candidates = []
    for src in range(len(client_images)):
        if src == dst:
            continue

        sc_set = client_sc_idx[src].get(target_sc)
        if not sc_set:
            continue

        src_size = len(client_images[src])
        if src_size <= 1:
            continue

        src_cov = len(client_sc_idx[src])
        lose_coverage = (len(sc_set) == 1 and src_cov <= min_supercats_per_client)
        hit_min_images = src_size <= effective_min_images

        # 排序优先级：不破坏最小图片约束 -> 不破坏 donor 覆盖 -> donor 越大越好
        candidates.append((1 if hit_min_images else 0, 1 if lose_coverage else 0, -src_size, src))

    if not candidates:
        return None
    candidates.sort()
    return candidates[0][3]


def _select_image_from_donor_supercat(
    donor: int,
    sc: str,
    client_sc_idx: List[Dict[str, Set[int]]],
    min_supercats_per_client: int,
) -> int | None:
    sc_set = client_sc_idx[donor].get(sc)
    if not sc_set:
        return None

    # 优先不让 donor 丢失该 supercategory
    if len(sc_set) >= 2:
        return min(sc_set)

    # 如果 donor 覆盖仍 > 最小要求，可转移该唯一图像
    if len(client_sc_idx[donor]) > min_supercats_per_client:
        return min(sc_set)

    return None


def _select_donor_for_min_images(
    dst: int,
    client_images: List[Set[int]],
    effective_min_images: int,
) -> int | None:
    preferred = []
    fallback = []

    for src in range(len(client_images)):
        if src == dst:
            continue
        s = len(client_images[src])
        if s > effective_min_images:
            preferred.append((-s, src))
        elif s > 1:
            fallback.append((-s, src))

    if preferred:
        preferred.sort()
        return preferred[0][1]
    if fallback:
        fallback.sort()
        return fallback[0][1]
    return None


def _select_image_for_min_images(
    donor: int,
    client_sc_idx: List[Dict[str, Set[int]]],
    min_supercats_per_client: int,
) -> int | None:
    # 优先选不会让 donor 丢失 supercategory 覆盖的图像
    for sc in sorted(client_sc_idx[donor].keys()):
        imgs = client_sc_idx[donor][sc]
        if len(imgs) >= 2:
            return min(imgs)

    # donor 覆盖富余时，允许移动单样本 supercat
    if len(client_sc_idx[donor]) > min_supercats_per_client:
        for sc in sorted(client_sc_idx[donor].keys()):
            imgs = client_sc_idx[donor][sc]
            if imgs:
                return min(imgs)

    return None


def enforce_constraints(
    client_images: List[Set[int]],
    dominant_supercat_by_image: Dict[int, str],
    min_images_per_client: int,
    min_supercats_per_client: int,
    effective_min_images: int,
    max_rounds: int = 10,
) -> Tuple[List[Set[int]], dict]:
    """
    约束修复：
    1) 每客户端至少 effective_min_images 张图（effective 由可行性决定）
    2) 每客户端至少 min_supercats_per_client 个主导 supercategory

    说明：
    - 先修 supercategory 覆盖，再修最小图片数
    - 每轮都是“整图像搬运”，绝不拆分图像内 QA
    - 若无法完全满足，返回告警并给出最终实际最小值
    """
    num_clients = len(client_images)
    client_sc_idx = _build_client_supercat_index(client_images, dominant_supercat_by_image)

    total_images = sum(len(s) for s in client_images)
    feasible_floor = total_images // max(1, num_clients)

    report = {
        "requested_min_images_per_client": int(min_images_per_client),
        "effective_min_images_per_client": int(effective_min_images),
        "global_feasible_image_floor": int(feasible_floor),
        "requested_min_supercats_per_client": int(min_supercats_per_client),
        "moves_for_supercat_coverage": 0,
        "moves_for_min_images": 0,
        "rounds_executed": 0,
        "warnings": [],
    }

    all_supercats = sorted(set(dominant_supercat_by_image.values()))
    if len(all_supercats) < min_supercats_per_client:
        report["warnings"].append(
            f"Global dominant supercategory count={len(all_supercats)} < min_supercats_per_client={min_supercats_per_client}, impossible to fully satisfy."
        )

    if min_images_per_client > feasible_floor:
        report["warnings"].append(
            f"Requested min_images_per_client={min_images_per_client} is infeasible. Feasible floor is {feasible_floor}."
        )

    global_sc_freq = Counter(dominant_supercat_by_image.values())

    for round_idx in range(1, max_rounds + 1):
        moved_this_round = 0

        # A) 先修 supercategory 覆盖不足
        order_by_cov = sorted(range(num_clients), key=lambda c: (len(client_sc_idx[c]), len(client_images[c]), c))
        for dst in order_by_cov:
            guard = 0
            while len(client_sc_idx[dst]) < min_supercats_per_client:
                guard += 1
                if guard > 100000:
                    break

                have = set(client_sc_idx[dst].keys())
                missing = [sc for sc in all_supercats if sc not in have]
                if not missing:
                    break

                # 优先常见 supercat，成功率更高
                missing.sort(key=lambda sc: (-global_sc_freq[sc], sc))

                moved = False
                for target_sc in missing:
                    donor = _select_donor_for_supercat(
                        target_sc=target_sc,
                        dst=dst,
                        client_images=client_images,
                        client_sc_idx=client_sc_idx,
                        effective_min_images=effective_min_images,
                        min_supercats_per_client=min_supercats_per_client,
                    )
                    if donor is None:
                        continue

                    iid = _select_image_from_donor_supercat(
                        donor=donor,
                        sc=target_sc,
                        client_sc_idx=client_sc_idx,
                        min_supercats_per_client=min_supercats_per_client,
                    )
                    if iid is None:
                        continue

                    _move_one_image(
                        image_id=iid,
                        src=donor,
                        dst=dst,
                        client_images=client_images,
                        client_sc_idx=client_sc_idx,
                        dominant_supercat_by_image=dominant_supercat_by_image,
                    )
                    report["moves_for_supercat_coverage"] += 1
                    moved_this_round += 1
                    moved = True
                    break

                if not moved:
                    break

        # B) 再修最小图片数不足
        order_by_size = sorted(range(num_clients), key=lambda c: (len(client_images[c]), c))
        for dst in order_by_size:
            guard = 0
            while len(client_images[dst]) < effective_min_images:
                guard += 1
                if guard > 200000:
                    break

                donor = _select_donor_for_min_images(
                    dst=dst,
                    client_images=client_images,
                    effective_min_images=effective_min_images,
                )
                if donor is None:
                    break

                iid = _select_image_for_min_images(
                    donor=donor,
                    client_sc_idx=client_sc_idx,
                    min_supercats_per_client=min_supercats_per_client,
                )
                if iid is None:
                    # 兜底：donor 任取一张
                    if not client_images[donor]:
                        break
                    iid = min(client_images[donor])

                _move_one_image(
                    image_id=iid,
                    src=donor,
                    dst=dst,
                    client_images=client_images,
                    client_sc_idx=client_sc_idx,
                    dominant_supercat_by_image=dominant_supercat_by_image,
                )
                report["moves_for_min_images"] += 1
                moved_this_round += 1

        report["rounds_executed"] = round_idx
        if moved_this_round == 0:
            break

    final_sizes = [len(s) for s in client_images]
    final_covs = [len(client_sc_idx[c]) for c in range(num_clients)]

    report["final_min_images_per_client"] = int(min(final_sizes)) if final_sizes else 0
    report["final_min_supercats_per_client"] = int(min(final_covs)) if final_covs else 0

    if final_sizes and min(final_sizes) < effective_min_images:
        report["warnings"].append(
            f"After repair, min images per client={min(final_sizes)} < effective target={effective_min_images}."
        )
    if final_covs and min(final_covs) < min_supercats_per_client:
        report["warnings"].append(
            f"After repair, min dominant supercategory coverage={min(final_covs)} < target={min_supercats_per_client}."
        )

    return client_images, report


def _sorted_counter(counter: Counter) -> Dict[str, int]:
    return {k: int(counter[k]) for k in sorted(counter.keys())}


def _safe_prob(counter_dict: Dict[str, int], labels: List[str]) -> List[float]:
    total = float(sum(counter_dict.get(lb, 0) for lb in labels))
    if total <= 0.0:
        return [0.0 for _ in labels]
    return [float(counter_dict.get(lb, 0)) / total for lb in labels]


def _kl_div(p: List[float], q: List[float], eps: float = 1e-12) -> float:
    val = 0.0
    for pi, qi in zip(p, q):
        if pi <= 0.0:
            continue
        val += pi * math.log(pi / max(qi, eps))
    return val


def _js_div(p: List[float], q: List[float], eps: float = 1e-12) -> float:
    m = [(pi + qi) / 2.0 for pi, qi in zip(p, q)]
    return 0.5 * _kl_div(p, m, eps) + 0.5 * _kl_div(q, m, eps)


def _l1_dist(p: List[float], q: List[float]) -> float:
    return sum(abs(pi - qi) for pi, qi in zip(p, q))


def gather_client_questions_annotations(
    image_ids: Set[int],
    questions_by_image: Dict[int, List[dict]],
    annotations_by_image: Dict[int, List[dict]],
) -> Tuple[List[dict], List[dict], Set[int]]:
    """按 image_id 收集该客户端 questions/annotations，并用 question_id 做对齐过滤。"""
    selected_questions: List[dict] = []
    selected_qids: Set[int] = set()

    for iid in sorted(image_ids):
        qs = questions_by_image.get(iid, [])
        selected_questions.extend(qs)
        for q in qs:
            qid = q.get("question_id")
            if qid is not None:
                selected_qids.add(int(qid))

    selected_annotations: List[dict] = []
    for iid in sorted(image_ids):
        anns = annotations_by_image.get(iid, [])
        for ann in anns:
            qid = ann.get("question_id")
            if qid is not None and int(qid) in selected_qids:
                selected_annotations.append(ann)

    return selected_questions, selected_annotations, selected_qids


def gather_client_stats(
    client_id: int,
    alpha: float,
    seed: int,
    image_ids: Set[int],
    questions_by_image: Dict[int, List[dict]],
    annotations_by_image: Dict[int, List[dict]],
    dominant_supercat_by_image: Dict[int, str],
    image_supercat_counts: Dict[int, Counter],
) -> dict:
    dominant_counter = Counter()
    all_supercat_counter = Counter()

    num_questions = 0
    num_annotations_raw = 0

    for iid in image_ids:
        dominant_counter[dominant_supercat_by_image[iid]] += 1

        sc_counter = image_supercat_counts.get(iid)
        if sc_counter:
            all_supercat_counter.update(sc_counter)
        else:
            all_supercat_counter[NO_INSTANCE_SUPERCAT] += 1

        num_questions += len(questions_by_image.get(iid, []))
        num_annotations_raw += len(annotations_by_image.get(iid, []))

    return {
        "client_id": int(client_id),
        "alpha": float(alpha),
        "random_seed": int(seed),
        "num_images": int(len(image_ids)),
        "num_questions": int(num_questions),
        "num_annotations_raw": int(num_annotations_raw),
        "num_dominant_supercategories": int(len(dominant_counter)),
        "dominant_supercategory_distribution": _sorted_counter(dominant_counter),
        "all_supercategory_distribution": _sorted_counter(all_supercat_counter),
    }


def export_client_dataset(
    client_dir: Path,
    client_id: int,
    alpha: float,
    seed: int,
    image_ids: Set[int],
    questions_header: dict,
    annotations_header: dict,
    questions_by_image: Dict[int, List[dict]],
    annotations_by_image: Dict[int, List[dict]],
    dominant_supercat_by_image: Dict[int, str],
    image_supercat_counts: Dict[int, Counter],
) -> dict:
    client_dir.mkdir(parents=True, exist_ok=True)

    questions_list, annotations_list, _ = gather_client_questions_annotations(
        image_ids=image_ids,
        questions_by_image=questions_by_image,
        annotations_by_image=annotations_by_image,
    )

    q_out = dict(questions_header)
    q_out["questions"] = questions_list

    a_out = dict(annotations_header)
    a_out["annotations"] = annotations_list

    with (client_dir / "train_questions.json").open("w", encoding="utf-8") as f:
        json.dump(q_out, f, ensure_ascii=False)

    with (client_dir / "train_annotations.json").open("w", encoding="utf-8") as f:
        json.dump(a_out, f, ensure_ascii=False)

    meta = gather_client_stats(
        client_id=client_id,
        alpha=alpha,
        seed=seed,
        image_ids=image_ids,
        questions_by_image=questions_by_image,
        annotations_by_image=annotations_by_image,
        dominant_supercat_by_image=dominant_supercat_by_image,
        image_supercat_counts=image_supercat_counts,
    )
    meta["num_annotations_aligned"] = int(len(annotations_list))

    if len(questions_list) != len(annotations_list):
        logging.warning(
            "Client %d alpha=%.4g question/annotation mismatch: %d vs %d",
            client_id,
            alpha,
            len(questions_list),
            len(annotations_list),
        )

    with (client_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return meta


def validate_partition(
    all_image_ids: Set[int],
    client_images: List[Set[int]],
    questions_by_image: Dict[int, List[dict]],
    annotations_by_image: Dict[int, List[dict]],
    dominant_supercat_by_image: Dict[int, str],
    min_images_per_client: int,
    min_supercats_per_client: int,
) -> dict:
    """基础一致性校验。"""
    issues: List[str] = []
    warnings: List[str] = []

    owner: Dict[int, int] = {}
    dup_count = 0

    for cid, imgs in enumerate(client_images):
        for iid in imgs:
            if iid in owner:
                dup_count += 1
            else:
                owner[iid] = cid

    if dup_count > 0:
        issues.append(f"Duplicate image assignments detected: {dup_count}")

    assigned = set(owner.keys())
    missing = all_image_ids - assigned
    extra = assigned - all_image_ids

    if missing:
        issues.append(f"Missing images not assigned: {len(missing)}")
    if extra:
        issues.append(f"Extra images assigned (not in train images): {len(extra)}")

    per_client = []
    for cid, imgs in enumerate(client_images):
        dom_cov = len({dominant_supercat_by_image[i] for i in imgs}) if imgs else 0

        num_q = 0
        num_a = 0
        qids = set()
        aqids = set()

        for iid in imgs:
            qs = questions_by_image.get(iid, [])
            anns = annotations_by_image.get(iid, [])
            num_q += len(qs)
            num_a += len(anns)

            for q in qs:
                qid = q.get("question_id")
                if qid is not None:
                    qids.add(int(qid))
            for ann in anns:
                qid = ann.get("question_id")
                if qid is not None:
                    aqids.add(int(qid))

        missing_ann_qids = qids - aqids
        extra_ann_qids = aqids - qids

        if missing_ann_qids:
            issues.append(
                f"Client {cid}: {len(missing_ann_qids)} question_ids in questions but missing in annotations"
            )
        if extra_ann_qids:
            warnings.append(
                f"Client {cid}: {len(extra_ann_qids)} annotation question_ids not found in questions"
            )

        if len(imgs) < min_images_per_client:
            issues.append(
                f"Client {cid}: num_images={len(imgs)} < min_images_per_client={min_images_per_client}"
            )
        if dom_cov < min_supercats_per_client:
            issues.append(
                f"Client {cid}: dominant supercategory coverage={dom_cov} < min_supercats_per_client={min_supercats_per_client}"
            )

        per_client.append(
            {
                "client_id": cid,
                "num_images": len(imgs),
                "num_questions": num_q,
                "num_annotations_raw": num_a,
                "dominant_supercategory_coverage": dom_cov,
            }
        )

    return {
        "is_valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "per_client": per_client,
    }


def compute_partition_summary(
    alpha: float,
    num_clients: int,
    seed: int,
    client_images: List[Set[int]],
    questions_by_image: Dict[int, List[dict]],
    annotations_by_image: Dict[int, List[dict]],
    dominant_supercat_by_image: Dict[int, str],
    image_supercat_counts: Dict[int, Counter],
    constraint_report: dict,
    validation_result: dict,
) -> dict:
    per_client_meta = []
    global_dom_counter = Counter(dominant_supercat_by_image.values())

    for cid in range(num_clients):
        meta = gather_client_stats(
            client_id=cid,
            alpha=alpha,
            seed=seed,
            image_ids=client_images[cid],
            questions_by_image=questions_by_image,
            annotations_by_image=annotations_by_image,
            dominant_supercat_by_image=dominant_supercat_by_image,
            image_supercat_counts=image_supercat_counts,
        )
        per_client_meta.append(meta)

    labels = sorted(global_dom_counter.keys())
    global_prob = _safe_prob(_sorted_counter(global_dom_counter), labels)

    divergence = []
    for meta in per_client_meta:
        p = _safe_prob(meta["dominant_supercategory_distribution"], labels)
        divergence.append(
            {
                "client_id": meta["client_id"],
                "kl_to_global": float(_kl_div(p, global_prob)),
                "jsd_to_global": float(_js_div(p, global_prob)),
                "l1_to_global": float(_l1_dist(p, global_prob)),
            }
        )

    return {
        "alpha": float(alpha),
        "num_clients": int(num_clients),
        "random_seed": int(seed),
        "total_images": int(sum(m["num_images"] for m in per_client_meta)),
        "total_questions": int(sum(m["num_questions"] for m in per_client_meta)),
        "per_client_num_images": [m["num_images"] for m in per_client_meta],
        "per_client_num_questions": [m["num_questions"] for m in per_client_meta],
        "per_client_supercategory_coverage": [m["num_dominant_supercategories"] for m in per_client_meta],
        "dominant_supercategory_histogram_per_client": [m["dominant_supercategory_distribution"] for m in per_client_meta],
        "global_dominant_supercategory_distribution": _sorted_counter(global_dom_counter),
        "distribution_divergence_to_global": divergence,
        "constraint_report": constraint_report,
        "validation": validation_result,
    }


def alpha_to_dirname(alpha: float) -> str:
    s = f"{alpha:.10g}"
    if "." not in s:
        s += ".0"
    return f"alpha_{s}"


def seed_for_alpha(base_seed: int, alpha: float, idx: int) -> int:
    return int(base_seed + round(alpha * 10000) + idx * 1000003)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Partition VQAv2 train set by image-level supercategory Dirichlet Non-IID."
    )
    parser.add_argument("--questions_train_path", type=Path, required=True)
    parser.add_argument("--annotations_train_path", type=Path, required=True)
    parser.add_argument("--coco_instances_train_path", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)

    parser.add_argument("--num_clients", type=int, default=10)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.1, 0.5, 1.0])
    parser.add_argument("--min_images_per_client", type=int, default=5000)
    parser.add_argument("--min_supercats_per_client", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true", help="Only partition+stats, do not write files")

    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()

    if args.num_clients <= 0:
        raise ValueError("--num_clients must be > 0")
    if any(a <= 0 for a in args.alphas):
        raise ValueError("All --alphas must be > 0")

    logging.info("Loading VQAv2 questions: %s", args.questions_train_path)
    questions_data = load_vqa_questions(args.questions_train_path)
    logging.info("Loading VQAv2 annotations: %s", args.annotations_train_path)
    annotations_data = load_vqa_annotations(args.annotations_train_path)
    logging.info("Loading COCO instances: %s", args.coco_instances_train_path)
    coco_instances_data = load_coco_instances(args.coco_instances_train_path)

    questions = questions_data["questions"]
    annotations = annotations_data["annotations"]

    (
        questions_by_image,
        annotations_by_image,
        qids_q,
        qids_a,
        image_ids_q,
        image_ids_a,
    ) = build_vqa_indices(questions, annotations)

    if qids_q - qids_a:
        logging.warning("question_ids in questions but missing in annotations: %d", len(qids_q - qids_a))
    if qids_a - qids_q:
        logging.warning("question_ids in annotations but missing in questions: %d", len(qids_a - qids_q))
    if image_ids_q - image_ids_a:
        logging.warning("image_ids in questions but missing in annotations: %d", len(image_ids_q - image_ids_a))
    if image_ids_a - image_ids_q:
        logging.warning("image_ids in annotations but missing in questions: %d", len(image_ids_a - image_ids_q))

    # 训练图像集合按 questions 定义，保证问题样本不丢失
    all_train_image_ids = set(image_ids_q)

    logging.info("Total train images: %d", len(all_train_image_ids))
    logging.info("Total questions: %d", len(questions))
    logging.info("Total annotations: %d", len(annotations))

    image_supercat_counts, _cat_map, unknown_cat_counter = build_image_supercategory_stats(
        coco_instances_data=coco_instances_data,
        target_image_ids=all_train_image_ids,
    )
    if unknown_cat_counter:
        logging.warning(
            "Unknown category_ids in instances: %d ids, %d occurrences",
            len(unknown_cat_counter),
            sum(unknown_cat_counter.values()),
        )

    dominant_supercat_by_image = assign_dominant_supercategory(
        image_ids=all_train_image_ids,
        image_supercat_counts=image_supercat_counts,
    )
    groups = group_images_by_dominant_supercategory(dominant_supercat_by_image)

    global_dom = Counter(dominant_supercat_by_image.values())
    logging.info("Global dominant supercategory count: %d", len(global_dom))
    logging.info("Global dominant supercategory top20: %s", global_dom.most_common(20))

    max_feasible_floor = len(all_train_image_ids) // args.num_clients
    effective_min_images = min(args.min_images_per_client, max_feasible_floor)
    if effective_min_images < args.min_images_per_client:
        logging.warning(
            "Requested min_images_per_client=%d infeasible. Use effective_min_images=%d.",
            args.min_images_per_client,
            effective_min_images,
        )

    q_header = {k: v for k, v in questions_data.items() if k != "questions"}
    a_header = {k: v for k, v in annotations_data.items() if k != "annotations"}

    if not args.dry_run:
        args.output_root.mkdir(parents=True, exist_ok=True)

    for idx, alpha in enumerate(args.alphas):
        cur_seed = seed_for_alpha(args.seed, alpha, idx)
        rng = np.random.default_rng(cur_seed)

        logging.info("=" * 80)
        logging.info("Start alpha=%.4g seed=%d", alpha, cur_seed)

        initial_client_images = dirichlet_partition_by_supercategory(
            groups=groups,
            num_clients=args.num_clients,
            alpha=alpha,
            rng=rng,
        )

        init_img_counts = [len(s) for s in initial_client_images]
        init_q_counts = [sum(len(questions_by_image.get(i, [])) for i in s) for s in initial_client_images]
        init_cov = [len({dominant_supercat_by_image[i] for i in s}) for s in initial_client_images]

        logging.info("Initial per-client images: %s", init_img_counts)
        logging.info("Initial per-client questions: %s", init_q_counts)
        logging.info("Initial per-client dominant supercat coverage: %s", init_cov)

        repaired_client_images, repair_report = enforce_constraints(
            client_images=initial_client_images,
            dominant_supercat_by_image=dominant_supercat_by_image,
            min_images_per_client=args.min_images_per_client,
            min_supercats_per_client=args.min_supercats_per_client,
            effective_min_images=effective_min_images,
        )

        final_img_counts = [len(s) for s in repaired_client_images]
        final_q_counts = [sum(len(questions_by_image.get(i, [])) for i in s) for s in repaired_client_images]
        final_cov = [len({dominant_supercat_by_image[i] for i in s}) for s in repaired_client_images]

        logging.info("Final per-client images: %s", final_img_counts)
        logging.info("Final per-client questions: %s", final_q_counts)
        logging.info("Final per-client dominant supercat coverage: %s", final_cov)
        logging.info(
            "Repair moves: supercat=%d, min_images=%d, rounds=%d",
            repair_report["moves_for_supercat_coverage"],
            repair_report["moves_for_min_images"],
            repair_report["rounds_executed"],
        )
        for w in repair_report.get("warnings", []):
            logging.warning("[alpha=%.4g] %s", alpha, w)

        validation = validate_partition(
            all_image_ids=all_train_image_ids,
            client_images=repaired_client_images,
            questions_by_image=questions_by_image,
            annotations_by_image=annotations_by_image,
            dominant_supercat_by_image=dominant_supercat_by_image,
            min_images_per_client=effective_min_images,
            min_supercats_per_client=args.min_supercats_per_client,
        )

        if validation["is_valid"]:
            logging.info("Validation passed for alpha=%.4g", alpha)
        else:
            logging.warning("Validation failed for alpha=%.4g", alpha)
            for issue in validation["issues"]:
                logging.warning("  ISSUE: %s", issue)
        for w in validation["warnings"]:
            logging.warning("  WARN: %s", w)

        summary = compute_partition_summary(
            alpha=alpha,
            num_clients=args.num_clients,
            seed=cur_seed,
            client_images=repaired_client_images,
            questions_by_image=questions_by_image,
            annotations_by_image=annotations_by_image,
            dominant_supercat_by_image=dominant_supercat_by_image,
            image_supercat_counts=image_supercat_counts,
            constraint_report=repair_report,
            validation_result=validation,
        )

        if args.dry_run:
            logging.info("Dry-run enabled, skip writing files for alpha=%.4g", alpha)
            continue

        alpha_dir = args.output_root / alpha_to_dirname(alpha)
        alpha_dir.mkdir(parents=True, exist_ok=True)

        clients_meta = []
        for cid in range(args.num_clients):
            client_dir = alpha_dir / f"client_{cid}"
            meta = export_client_dataset(
                client_dir=client_dir,
                client_id=cid,
                alpha=alpha,
                seed=cur_seed,
                image_ids=repaired_client_images[cid],
                questions_header=q_header,
                annotations_header=a_header,
                questions_by_image=questions_by_image,
                annotations_by_image=annotations_by_image,
                dominant_supercat_by_image=dominant_supercat_by_image,
                image_supercat_counts=image_supercat_counts,
            )
            clients_meta.append(meta)

        summary["clients_meta"] = clients_meta

        with (alpha_dir / "partition_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        logging.info("Wrote alpha=%.4g results to %s", alpha, alpha_dir)

    logging.info("All done.")


if __name__ == "__main__":
    main()
