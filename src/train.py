from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from checkpointing import load_state, save_state
from data import JsonImageDataset, build_transforms
from engine import evaluate, train_one_epoch
from labels import LabelSpec, infer_label_spec
from model import ModelConfig, create_model, freeze_backbone
from sampler import make_weighted_sampler


def _parse_comma_floats(value: str) -> list[float]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    return [float(p) for p in parts]


def _load_label_map_json(path: str) -> dict[Any, int]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError("label map JSON must be a JSON object")

    out: dict[Any, int] = {}
    for k, v in raw.items():
        try:
            out[k] = int(v)
        except Exception as e:
            raise ValueError(f"Invalid mapping value for key {k!r}: {v!r}") from e
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train anti-spoofing classifier")

    p.add_argument("--data-root", required=True, help="Dataset root directory")
    p.add_argument("--train-json", default=None, help="Path to train label JSON")
    p.add_argument("--val-json", default=None, help="Path to validation label JSON")

    p.add_argument("--label-col", default=None, help="Label column name or index")
    p.add_argument("--label-map-json", default=None, help="JSON file mapping raw labels to class ids")
    p.add_argument("--strip-prefix", default=None, help="Optional path prefix to strip from JSON index")

    p.add_argument("--model-name", default="mobilenetv4_conv_small.e2400_r224_in1k")
    p.add_argument("--num-classes", type=int, default=3)
    p.add_argument("--pretrained", action="store_true", help="Use ImageNet pretrained backbone")

    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)

    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)

    p.add_argument("--class-weights", default=None, help="Comma-separated weights, e.g. 1,2,4")

    p.add_argument("--weighted-sampler", action="store_true", help="Use class-balanced sampling")
    p.add_argument("--no-weighted-sampler", action="store_true", help="Disable class-balanced sampling")

    p.add_argument("--save-dir", default="./models")
    p.add_argument("--resume", default=None, help="Path to a saved training state (.pth)")
    p.add_argument("--freeze-backbone", action="store_true", help="Train only the classifier head")

    return p


def _resolve_label_col(value: Optional[str]) -> Optional[object]:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return value


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    default_train = os.path.join(args.data_root, "metas", "labels", "train_label.json")
    default_val = os.path.join(args.data_root, "metas", "labels", "test_label.json")
    train_json = args.train_json or default_train
    val_json = args.val_json or default_val

    # Determine label spec (column + optional mapping)
    meta_df = pd.read_json(train_json, orient="index")
    preferred_col = _resolve_label_col(args.label_col)
    label_spec = infer_label_spec(meta_df.columns, preferred=preferred_col)
    if args.label_map_json:
        label_spec = LabelSpec(label_column=label_spec.label_column, raw_to_class=_load_label_map_json(args.label_map_json))

    train_tf, val_tf = build_transforms(args.image_size)

    train_ds = JsonImageDataset(
        root_dir=args.data_root,
        json_path=train_json,
        label_spec=label_spec,
        transform=train_tf,
        strip_prefix=args.strip_prefix,
        mode="train",
    )
    val_ds = JsonImageDataset(
        root_dir=args.data_root,
        json_path=val_json,
        label_spec=label_spec,
        transform=val_tf,
        strip_prefix=args.strip_prefix,
        mode="val",
    )

    use_sampler = args.weighted_sampler and not args.no_weighted_sampler
    if use_sampler:
        labels = [label_spec.to_class_id(v) for v in train_ds.raw_labels]  # type: ignore[attr-defined]
        sampler = make_weighted_sampler(labels)
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        shuffle = False
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        shuffle = True

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = create_model(
        ModelConfig(model_name=args.model_name, num_classes=args.num_classes, pretrained=args.pretrained),
        device=device,
    )

    if args.freeze_backbone:
        freeze_backbone(model)

    class_weights = None
    if args.class_weights:
        w = torch.tensor(_parse_comma_floats(args.class_weights), dtype=torch.float32, device=device)
        if w.numel() != args.num_classes:
            raise ValueError("--class-weights length must match --num-classes")
        class_weights = w

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 1
    best_acc = 0.0

    if args.resume:
        state = load_state(path=args.resume, model=model, optimizer=optimizer, scheduler=scheduler, map_location=str(device))
        start_epoch = int(state.epoch) + 1
        best_acc = float(state.best_metric)

    print("Training configuration")
    print(f"  device: {device}")
    print(f"  data_root: {args.data_root}")
    print(f"  train_json: {train_json}")
    print(f"  val_json: {val_json}")
    print(f"  model: {args.model_name}")
    print(f"  num_classes: {args.num_classes}")
    print(f"  image_size: {args.image_size}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  weighted_sampler: {use_sampler}")
    print(f"  freeze_backbone: {bool(args.freeze_backbone)}")
    print(f"  save_dir: {args.save_dir}")
    print(f"  epochs: {args.epochs}")
    print(f"  start_epoch: {start_epoch}")

    state_path = os.path.join(args.save_dir, "state.pth")

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            num_classes=args.num_classes,
        )

        scheduler.step()

        print(
            f"epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.2f} | "
            f"val_loss={val_metrics.loss:.4f} val_acc={val_metrics.accuracy:.2f} | "
            f"recall={','.join([f'{r*100:.2f}' for r in val_metrics.recall_per_class])}"
        )

        is_best = val_metrics.accuracy > best_acc
        if is_best:
            best_acc = float(val_metrics.accuracy)
            best_path = os.path.join(args.save_dir, "best.pth")
            torch.save(model.state_dict(), best_path)

            epoch_path = os.path.join(args.save_dir, f"epoch{epoch}_metric{best_acc:.2f}.pth")
            torch.save(model.state_dict(), epoch_path)

        torch.save(model.state_dict(), os.path.join(args.save_dir, "last.pth"))
        save_state(
            path=state_path,
            epoch=epoch,
            best_metric=best_acc,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )

    print(f"done | best_val_acc={best_acc:.2f} | best.pth saved")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
