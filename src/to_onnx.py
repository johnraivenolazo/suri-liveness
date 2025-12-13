from __future__ import annotations

import argparse
from pathlib import Path

import timm
import torch
import torch.nn as nn


def _replace_classifier_head(model: torch.nn.Module, num_classes: int) -> None:
    if hasattr(model, "classifier") and getattr(model, "classifier") is not None:
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
        return

    if hasattr(model, "head") and getattr(model, "head") is not None:
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes)
        return

    if hasattr(model, "fc") and getattr(model, "fc") is not None:
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return

    raise ValueError("Could not locate classifier head on model")


def export_onnx(*, input_path: Path, output_path: Path, model_name: str, num_classes: int, image_size: int) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"PyTorch checkpoint not found: {input_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = timm.create_model(model_name, pretrained=False)
    _replace_classifier_head(model, num_classes)

    state_dict = torch.load(input_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.eval().to(device)

    dummy_input = torch.randn(1, 3, image_size, image_size, device=device)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Export PyTorch checkpoint to ONNX")
    p.add_argument("--input", "-i", required=True, help="Path to input checkpoint (.pth)")
    p.add_argument("--output", "-o", required=True, help="Path to output ONNX file (.onnx)")
    p.add_argument("--model-name", default="mobilenetv4_conv_small.e2400_r224_in1k")
    p.add_argument("--num-classes", type=int, default=3)
    p.add_argument("--image-size", type=int, default=224)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    export_onnx(
        input_path=Path(args.input),
        output_path=Path(args.output),
        model_name=args.model_name,
        num_classes=args.num_classes,
        image_size=args.image_size,
    )

    print(f"done | wrote: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
