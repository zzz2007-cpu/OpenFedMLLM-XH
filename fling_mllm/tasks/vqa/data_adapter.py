import json
import os
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..registry import register_data_loader
from .formatter import format_vqa_sample
from .metrics import normalize_vqa_answer
from .prompt_builder import normalize_question_text


def _read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _dedup_keep_order(items: List[str]) -> List[str]:
    out = []
    seen = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _resolve_relative_path(path_str: str, base_dir: str) -> str:
    if os.path.isabs(path_str):
        return path_str
    return os.path.abspath(os.path.join(base_dir, path_str))


def _extract_answers_from_annotation(annotation: Optional[Dict]) -> Tuple[str, List[str], Optional[str]]:
    if not annotation:
        return "", [], None

    mc_answer = str(annotation.get("multiple_choice_answer", "")).strip()

    answers = []
    for ans in annotation.get("answers", []) or []:
        raw = ans.get("answer") if isinstance(ans, dict) else ans
        text = "" if raw is None else str(raw).strip()
        if text:
            answers.append(text)

    if mc_answer:
        primary = mc_answer
    elif answers:
        norm_counter = Counter(normalize_vqa_answer(a) for a in answers if normalize_vqa_answer(a))
        if norm_counter:
            top_norm = norm_counter.most_common(1)[0][0]
            primary = next((a for a in answers if normalize_vqa_answer(a) == top_norm), answers[0])
        else:
            primary = answers[0]
    else:
        primary = ""

    answer_type = annotation.get("answer_type")
    return primary, answers, str(answer_type) if answer_type is not None else None


def _resolve_vqa_image_path(
    image_id: int,
    data_subtype: Optional[str],
    image_root: Optional[str],
    fallback_roots: List[str],
) -> str:
    subtype = str(data_subtype or "train2014").strip()
    file_name = f"COCO_{subtype}_{int(image_id):012d}.jpg"

    candidates: List[str] = []
    root_candidates: List[str] = []

    if image_root:
        root_candidates.append(os.path.abspath(image_root))
    env_root = os.environ.get("OPENFED_VQA_IMAGE_ROOT")
    if env_root:
        root_candidates.append(os.path.abspath(env_root))

    root_candidates.extend([os.path.abspath(p) for p in fallback_roots if p])
    root_candidates.append(os.path.abspath(os.path.join(os.getcwd(), "VQAv2")))

    for root in _dedup_keep_order(root_candidates):
        if os.path.basename(root).lower() == subtype.lower():
            candidates.append(os.path.join(root, file_name))
        else:
            candidates.append(os.path.join(root, subtype, file_name))
            candidates.append(os.path.join(root, file_name))

    for candidate in _dedup_keep_order(candidates):
        if os.path.exists(candidate):
            return candidate

    # Return the first candidate for clear downstream error messages.
    return candidates[0] if candidates else file_name


def _resolve_questions_annotations_from_dir(data_dir: str, split: str) -> Tuple[str, Optional[str]]:
    split_l = str(split or "train").strip().lower()
    d = Path(data_dir)

    if split_l in {"train", "training"}:
        q_candidates = [
            d / "train_questions.json",
            d / "v2_OpenEnded_mscoco_train2014_questions.json",
            d / "questions.json",
        ]
        a_candidates = [
            d / "train_annotations.json",
            d / "v2_mscoco_train2014_annotations.json",
            d / "annotations.json",
        ]
    elif split_l in {"eval", "val", "validation", "dev"}:
        q_candidates = [
            d / "val_questions.json",
            d / "dev_questions.json",
            d / "test_questions.json",
            d / "v2_OpenEnded_mscoco_val2014_questions.json",
            d / "questions.json",
        ]
        a_candidates = [
            d / "val_annotations.json",
            d / "dev_annotations.json",
            d / "test_annotations.json",
            d / "v2_mscoco_val2014_annotations.json",
            d / "annotations.json",
        ]
    else:
        q_candidates = [d / "test_questions.json", d / "questions.json"]
        a_candidates = [d / "test_annotations.json", d / "annotations.json"]

    question_path = next((str(p) for p in q_candidates if p.exists()), None)
    if question_path is None:
        globs = sorted(d.glob("*questions*.json"))
        if len(globs) == 1:
            question_path = str(globs[0])
        elif globs:
            preferred = [g for g in globs if split_l in g.name.lower()]
            question_path = str(preferred[0] if preferred else globs[0])

    annotation_path = next((str(p) for p in a_candidates if p.exists()), None)
    if annotation_path is None:
        globs = sorted(d.glob("*annotations*.json"))
        if len(globs) == 1:
            annotation_path = str(globs[0])
        elif globs:
            preferred = [g for g in globs if split_l in g.name.lower()]
            annotation_path = str(preferred[0] if preferred else globs[0])

    if question_path is None:
        raise FileNotFoundError(
            f"VQA questions file not found in directory={data_dir!r} for split={split!r}."
        )
    return question_path, annotation_path


def _extract_question_and_answer_from_conversations(conversations: List[Dict]) -> Tuple[str, str]:
    question = ""
    answer = ""
    for turn in conversations or []:
        role = str(turn.get("role", "")).strip().lower()
        content = str(turn.get("content", "")).strip()
        if role == "user" and not question:
            question = content
        if role == "assistant":
            answer = content
    return question, answer


def _extract_reference_answers(item: Dict) -> List[str]:
    candidates = item.get("answers")
    if candidates is None:
        candidates = item.get("answer_candidates")
    if candidates is None:
        candidates = item.get("references")
    refs = []
    if isinstance(candidates, list):
        for v in candidates:
            s = "" if v is None else str(v).strip()
            if s:
                refs.append(s)
    elif isinstance(candidates, str) and candidates.strip():
        refs.append(candidates.strip())
    return refs


def _from_record_list(
    records: List[Dict],
    base_dir: str,
    split: str,
    data_subtype: Optional[str],
    image_root: Optional[str],
    prompt_template: Optional[str],
    require_answer: bool,
    strict_image_path: bool,
) -> List[Dict]:
    output = []
    skipped_missing_answer = 0

    for idx, item in enumerate(records):
        if not isinstance(item, dict):
            continue

        sample_id = str(item.get("id", item.get("question_id", idx)))
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()

        if not question and isinstance(item.get("conversations"), list):
            q2, a2 = _extract_question_and_answer_from_conversations(item.get("conversations", []))
            question = question or q2
            answer = answer or a2

        question = normalize_question_text(question)
        refs = _extract_reference_answers(item)
        if not answer and refs:
            answer = refs[0]

        if require_answer and not answer:
            skipped_missing_answer += 1
            continue

        image_path = item.get("image")
        if isinstance(image_path, str) and image_path.strip():
            image_path = _resolve_relative_path(image_path.strip(), base_dir)
        else:
            image_id = item.get("image_id")
            if image_id is None:
                raise ValueError(
                    f"VQA record idx={idx} has no 'image' or 'image_id'. sample_id={sample_id}"
                )
            image_path = _resolve_vqa_image_path(
                image_id=int(image_id),
                data_subtype=item.get("data_subtype") or data_subtype,
                image_root=image_root,
                fallback_roots=[base_dir, os.path.dirname(base_dir)],
            )

        if strict_image_path and not os.path.exists(image_path):
            raise FileNotFoundError(
                "VQA image path not found: "
                f"sample_id={sample_id}, image={image_path}. "
                "Please verify data_args.vqa_image_root or input JSON image paths."
            )

        metadata = {
            "question_id": item.get("question_id"),
            "image_id": item.get("image_id"),
            "answer_type": item.get("answer_type"),
            "split": split,
        }

        if require_answer:
            sample = format_vqa_sample(
                sample_id=sample_id,
                image_path=image_path,
                question=question,
                answer=answer,
                answers=refs,
                prompt_template=prompt_template,
                metadata=metadata,
            )
        else:
            # Eval/test records may not carry labels.
            sample = {
                "id": sample_id,
                "image": image_path,
                "question": question,
                "answer": answer,
                "answers": refs,
                "split": split,
                **metadata,
            }
            if answer:
                sample["conversations"] = format_vqa_sample(
                    sample_id=sample_id,
                    image_path=image_path,
                    question=question,
                    answer=answer,
                    answers=refs,
                    prompt_template=prompt_template,
                )["conversations"]

        output.append(sample)

    if require_answer and not output:
        raise ValueError(
            f"No valid VQA samples loaded from record list. skipped_missing_answer={skipped_missing_answer}"
        )
    return output


def _build_samples_from_question_answer_files(
    question_data: Dict,
    annotation_data: Optional[Dict],
    base_dir: str,
    split: str,
    image_root: Optional[str],
    prompt_template: Optional[str],
    require_answer: bool,
    strict_image_path: bool,
) -> List[Dict]:
    questions = question_data.get("questions", []) if isinstance(question_data, dict) else []
    if not isinstance(questions, list):
        raise ValueError("VQA questions payload is invalid: 'questions' should be a list.")

    annotations = []
    if isinstance(annotation_data, dict):
        annotations = annotation_data.get("annotations", []) or []

    ann_by_qid: Dict[int, Dict] = {}
    for ann in annotations:
        if not isinstance(ann, dict):
            continue
        qid = ann.get("question_id")
        if qid is None:
            continue
        ann_by_qid[int(qid)] = ann

    data_subtype = (
        question_data.get("data_subtype")
        if isinstance(question_data, dict)
        else None
    )
    if not data_subtype and isinstance(annotation_data, dict):
        data_subtype = annotation_data.get("data_subtype")

    output = []
    skipped_missing_answer = 0

    for idx, q in enumerate(questions):
        if not isinstance(q, dict):
            continue

        qid = q.get("question_id")
        image_id = q.get("image_id")
        if qid is None or image_id is None:
            continue

        question = normalize_question_text(q.get("question", ""))
        sample_id = str(qid)
        ann = ann_by_qid.get(int(qid))
        answer, answers, answer_type = _extract_answers_from_annotation(ann)

        if require_answer and not answer:
            skipped_missing_answer += 1
            continue

        image_path = _resolve_vqa_image_path(
            image_id=int(image_id),
            data_subtype=data_subtype,
            image_root=image_root,
            fallback_roots=[base_dir, os.path.dirname(base_dir)],
        )
        if strict_image_path and not os.path.exists(image_path):
            raise FileNotFoundError(
                "VQA image path not found: "
                f"question_id={qid}, image_id={image_id}, resolved={image_path}. "
                "Please set data_args.vqa_image_root correctly."
            )

        metadata = {
            "question_id": int(qid),
            "image_id": int(image_id),
            "answer_type": answer_type,
            "split": split,
        }

        if require_answer:
            sample = format_vqa_sample(
                sample_id=sample_id,
                image_path=image_path,
                question=question,
                answer=answer,
                answers=answers,
                prompt_template=prompt_template,
                metadata=metadata,
            )
        else:
            sample = {
                "id": sample_id,
                "image": image_path,
                "question": question,
                "answer": answer,
                "answers": answers,
                **metadata,
            }
            if answer:
                sample["conversations"] = format_vqa_sample(
                    sample_id=sample_id,
                    image_path=image_path,
                    question=question,
                    answer=answer,
                    answers=answers,
                    prompt_template=prompt_template,
                )["conversations"]

        output.append(sample)

    if require_answer and not output:
        raise ValueError(
            "No valid VQA samples were built from question/annotation files. "
            f"skipped_missing_answer={skipped_missing_answer}"
        )
    return output


@register_data_loader("vqa")
def load_vqa_samples(
    data_path: str,
    split: str = "train",
    data_format: str = "auto",
    vqa_image_root: Optional[str] = None,
    vqa_prompt_template: Optional[str] = None,
    require_answer: Optional[bool] = None,
    strict_image_path: bool = True,
    **kwargs,
) -> List[Dict]:
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"VQA data path not found: {data_path}")

    if require_answer is None:
        require_answer = str(split or "train").lower() in {"train", "training"}

    if os.path.isdir(data_path):
        q_path, a_path = _resolve_questions_annotations_from_dir(data_path, split=split)
        question_data = _read_json(q_path)
        annotation_data = _read_json(a_path) if a_path and os.path.exists(a_path) else None
        if require_answer and annotation_data is None:
            raise FileNotFoundError(
                f"VQA training split requires annotations, but none were found in directory={data_path!r}."
            )
        return _build_samples_from_question_answer_files(
            question_data=question_data,
            annotation_data=annotation_data,
            base_dir=os.path.dirname(q_path),
            split=split,
            image_root=vqa_image_root,
            prompt_template=vqa_prompt_template,
            require_answer=bool(require_answer),
            strict_image_path=bool(strict_image_path),
        )

    payload = _read_json(data_path)
    base_dir = os.path.dirname(os.path.abspath(data_path))

    if isinstance(payload, list):
        return _from_record_list(
            records=payload,
            base_dir=base_dir,
            split=split,
            data_subtype=None,
            image_root=vqa_image_root,
            prompt_template=vqa_prompt_template,
            require_answer=bool(require_answer),
            strict_image_path=bool(strict_image_path),
        )

    if isinstance(payload, dict):
        if "questions" in payload:
            question_data = payload
            annotation_data = payload if "annotations" in payload else None

            # Try sibling annotations when a standalone questions file is passed.
            if annotation_data is None:
                sibling = None
                if "train" in os.path.basename(data_path):
                    sibling = data_path.replace("questions", "annotations")
                elif "val" in os.path.basename(data_path):
                    sibling = data_path.replace("questions", "annotations")
                if sibling and os.path.exists(sibling):
                    annotation_data = _read_json(sibling)

            if require_answer and annotation_data is None:
                raise FileNotFoundError(
                    f"No annotation payload found for VQA training file: {data_path}"
                )

            return _build_samples_from_question_answer_files(
                question_data=question_data,
                annotation_data=annotation_data,
                base_dir=base_dir,
                split=split,
                image_root=vqa_image_root,
                prompt_template=vqa_prompt_template,
                require_answer=bool(require_answer),
                strict_image_path=bool(strict_image_path),
            )

    raise ValueError(
        "Unsupported VQA data format. Expected one of: \n"
        "1) directory with train_questions/train_annotations\n"
        "2) JSON list of records\n"
        "3) VQAv2 questions/annotations JSON dict"
    )
