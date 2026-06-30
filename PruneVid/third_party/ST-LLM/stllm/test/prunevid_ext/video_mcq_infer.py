import argparse
import json
import os
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

    if "answer_idx" in sample:
        answer_idx = int(sample["answer_idx"])
    else:
        answer_idx = candidates.index(sample["answer"])
    return question, f"({option_letter(answer_idx)}) {candidates[answer_idx]}"


def normalize_pred(pred: str, candidates: list[str]) -> str:
    pred_upper = pred.strip().upper()
    for idx, _ in enumerate(candidates):
        letter = option_letter(idx)
        if pred_upper.startswith(f"({letter})") or pred_upper.startswith(letter):
            return f"({letter})"
    pred_lower = pred.lower()
    for idx, candidate in enumerate(candidates):
        if candidate.lower().strip(". ") in pred_lower:
            return f"({option_letter(idx)})"
    return pred.strip()


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

    for sample in tqdm(samples):
        task_type = sample["task_type"]
        acc_dict.setdefault(task_type, [0, 0])
        acc_dict[task_type][1] += 1

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
        correct = pred_norm[:3] == gt[:3]
        if correct:
            acc_dict[task_type][0] += 1
        output.append(
            {
                "video": sample["video"],
                "question": sample["question"],
                "pred": pred,
                "pred_norm": pred_norm,
                "gt": gt,
                "task_type": task_type,
                "correct": correct,
            }
        )

    total_correct = sum(v[0] for v in acc_dict.values())
    total = sum(v[1] for v in acc_dict.values())
    acc_dict["Total Acc"] = f"{(total_correct / total * 100) if total else 0:.2f}%"

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, f"{args.output_name}.json"), "w") as f:
        json.dump({"acc_dict": acc_dict, "res_list": output}, f, indent=2)


if __name__ == "__main__":
    main()
