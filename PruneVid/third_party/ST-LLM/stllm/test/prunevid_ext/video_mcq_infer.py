import argparse
import json
import os
import re
from pathlib import Path

import torch
from tqdm import tqdm

from stllm.common.config import Config
from stllm.common.registry import registry
from stllm.conversation.conversation import Chat, CONV_VIDEO_Vicuna0, CONV_instructblip_Vicuna0

# Imports modules for registration.
from stllm.datasets.builders import *  # noqa: F401,F403
from stllm.models import *  # noqa: F401,F403
from stllm.processors import *  # noqa: F401,F403
from stllm.runners import *  # noqa: F401,F403
from stllm.tasks import *  # noqa: F401,F403


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg-path", required=True)
    parser.add_argument("--ckpt-path", required=True)
    parser.add_argument("--anno-path", required=True, nargs="+")
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--options", nargs="+")
    return parser.parse_args()


def option_letter(idx: int) -> str:
    return chr(ord("A") + idx)


def build_question(sample: dict) -> tuple[str, str]:
    candidates = sample["candidates"]
    question = f"Question: {sample['question']}\nOptions:\n"
    for idx, candidate in enumerate(candidates):
        question += f"({option_letter(idx)}) {candidate}\n"
    question += "\nOnly give the best option."

    ai = sample.get("answer_idx")
    if ai is not None and str(ai).strip().lstrip("-").isdigit():
        answer_idx = int(ai)
    elif sample.get("answer") in candidates:
        answer_idx = candidates.index(sample["answer"])
    else:
        return question, ""  # fullset: label is private -> no local ground truth
    return question, f"({option_letter(answer_idx)}) {candidates[answer_idx]}"


def normalize_pred(pred: str, candidates: list[str]) -> str:
    valid = {option_letter(idx) for idx in range(len(candidates))}
    pred_clean = pred.strip()

    # 1) First parenthesized option letter, e.g. "Answer: (C) ..." -> (C).
    #    Models commonly prefix with "Answer:", so a bare startswith(letter)
    #    check is unsafe (it matches the 'A' in "Answer").
    match = re.search(r"\(([A-Za-z])\)", pred_clean)
    if match and match.group(1).upper() in valid:
        return f"({match.group(1).upper()})"

    # 2) Strip leading "answer"/"option" lead-ins, then check a leading letter.
    stripped = re.sub(
        r"^\s*(the\s+)?(answer|option|ans)\s*(is)?\s*[:\.\-]?\s*",
        "",
        pred_clean,
        flags=re.IGNORECASE,
    ).strip()
    stripped_upper = stripped.upper()
    for letter in valid:
        if (
            stripped_upper == letter
            or stripped_upper.startswith(f"({letter})")
            or stripped_upper.startswith(f"{letter})")
            or stripped_upper.startswith(f"{letter}.")
            or stripped_upper.startswith(f"{letter}:")
            or stripped_upper.startswith(f"{letter} ")
        ):
            return f"({letter})"

    # 3) Fall back to matching the candidate text.
    pred_lower = pred_clean.lower()
    for idx, candidate in enumerate(candidates):
        needle = candidate.lower().strip(". ")
        if needle and needle in pred_lower:
            return f"({option_letter(idx)})"
    return pred_clean


def find_video(video_root: Path, video_name: str) -> Path:
    path = video_root / video_name
    if path.exists():
        return path
    stem = Path(video_name).stem
    for suffix in [".mp4", ".avi", ".mov", ".mkv"]:
        candidate = video_root / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing video: {video_name} under {video_root}")


def load_samples(paths: list[str]):
    samples = []
    for path in paths:
        with open(path) as f:
            data = json.load(f)
        task_name = Path(path).stem
        for sample in data:
            sample = dict(sample)
            sample.setdefault("task_type", task_name)
            samples.append(sample)
    return samples


def main():
    args = parse_args()
    cfg = Config(args)
    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_config.ckpt = args.ckpt_path

    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to(f"cuda:{args.gpu_id}")
    for _, param in model.named_parameters():
        param.requires_grad = False
    model.eval()
    model = model.to(torch.float16)

    conv_dict = {
        "minigpt4_vicuna0": CONV_VIDEO_Vicuna0,
        "minigpt4_vicuna0_btadapter": CONV_VIDEO_Vicuna0,
        "instructblip_vicuna0": CONV_instructblip_Vicuna0,
        "instructblip_vicuna0_btadapter": CONV_instructblip_Vicuna0,
    }
    conv_template = conv_dict[model_config.model_type]
    chat = Chat(model, device=f"cuda:{args.gpu_id}")

    samples = load_samples(args.anno_path)
    if args.limit:
        samples = samples[: args.limit]

    video_root = Path(args.video_root)
    output = []
    acc_dict = {}

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"{args.output_name}.json")

    # resume: reload prior predictions, skip completed items, rebuild acc from labeled ones
    done_keys = set()
    if os.path.exists(out_path):
        try:
            prev = json.load(open(out_path)).get("res_list", [])
            output = prev
            done_keys = {(o.get("video"), o.get("question")) for o in prev}
            for o in prev:
                if o.get("correct") is not None:
                    tt = o.get("task_type", "default")
                    acc_dict.setdefault(tt, [0, 0])
                    acc_dict[tt][1] += 1
                    if o["correct"]:
                        acc_dict[tt][0] += 1
            print(f"[resume] loaded {len(prev)} prior items", flush=True)
        except Exception as e:
            print(f"[resume] failed ({e}); starting fresh", flush=True)
            output, done_keys, acc_dict = [], set(), {}

    def save():
        acc = dict(acc_dict)
        tc = sum(v[0] for v in acc.values() if isinstance(v, list))
        tt = sum(v[1] for v in acc.values() if isinstance(v, list))
        acc["Total Acc"] = f"{(tc / tt * 100) if tt else 0:.2f}%"
        tmp = out_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"acc_dict": acc, "res_list": output}, f, indent=2)
        os.replace(tmp, out_path)

    for i, sample in enumerate(tqdm(samples)):
        if (sample["video"], sample["question"]) in done_keys:
            continue
        task_type = sample.get("task_type", "default")
        acc_dict.setdefault(task_type, [0, 0])

        video_path = find_video(video_root, sample["video"])
        question, gt = build_question(sample)

        chat_state = conv_template.copy()
        img_list = []
        chat.upload_video(str(video_path), chat_state, img_list, args.num_frames, question)
        chat.ask(question, chat_state)
        pred = chat.answer(
            conv=chat_state,
            img_list=img_list,
            num_beams=5,
            do_sample=False,
            temperature=1,
            max_new_tokens=64,
            max_length=2000,
        )[0]

        pred_norm = normalize_pred(pred, sample["candidates"])
        labeled = bool(gt)
        correct = (pred_norm[:3] == gt[:3]) if labeled else None
        if labeled:
            acc_dict[task_type][1] += 1
            if correct:
                acc_dict[task_type][0] += 1
        output.append(
            {
                "video": sample["video"],
                "question_idx": sample.get("question_idx"),
                "question": sample["question"],
                "pred": pred,
                "pred_norm": pred_norm,
                "gt": gt,
                "task_type": task_type,
                "correct": correct,
            }
        )
        if (i + 1) % 100 == 0:
            save()

    save()


if __name__ == "__main__":
    main()
