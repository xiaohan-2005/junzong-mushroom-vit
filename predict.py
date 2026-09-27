"""Run image matching with the released 169-label ViT checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="169 类菌类图像匹配")
    parser.add_argument("image", type=Path, help="输入图片路径")
    parser.add_argument("--weights", type=Path, default=ROOT / "best_model.pth")
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    size = int(config["img_size"])
    classes = int(config["num_classes"])
    labels = config["idx_to_class"]
    if len(labels) != classes:
        raise ValueError("类别映射数量与 num_classes 不一致")

    model = timm.create_model(
        "vit_base_patch16_224",
        pretrained=False,
        img_size=size,
        num_classes=classes,
        drop_path_rate=0.08,
    )
    state = torch.load(args.weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()

    with Image.open(args.image) as image:
        image = image.convert("RGB").resize((size, size), resample=Image.Resampling.BICUBIC)
        pixels = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0)
    mean = torch.tensor(config["mean"], dtype=tensor.dtype).view(1, 3, 1, 1)
    std = torch.tensor(config["std"], dtype=tensor.dtype).view(1, 3, 1, 1)
    tensor = (tensor - mean) / std
    with torch.inference_mode():
        logits = model(tensor)[0]
        relative_scores = torch.softmax(logits, dim=0)
        values, indices = torch.topk(relative_scores, k=min(max(args.top_k, 1), classes))

    for rank, (index, score) in enumerate(zip(indices.tolist(), values.tolist()), 1):
        print(f"{rank}. {labels[str(index)]}: {score:.4f} (相对分数)")
    print("仅用于图像匹配与学习；分数不代表可食用概率，不可据此判断食用安全。")


if __name__ == "__main__":
    main()
