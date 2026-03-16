#!/usr/bin/env python3
"""Export MayaScan v2 checkpoints to ONNX and browser-friendly variants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import onnx
import torch

from mayascan.config import V2_ARCH, V2_CLASSES, V2_ENCODER
from mayascan.detect import _load_v2_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export MayaScan binary class checkpoints to ONNX, FP16, and INT8.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models"),
        help="Directory containing mayascan_v2_* checkpoints.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("web/public/models"),
        help="Directory to write exported ONNX artifacts and manifest.",
    )
    parser.add_argument("--arch", default=V2_ARCH, help="Segmentation architecture name.")
    parser.add_argument("--encoder", default=V2_ENCODER, help="Encoder backbone name.")
    parser.add_argument(
        "--tile-size",
        type=int,
        default=480,
        help="Square input size used during browser inference.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version to export.",
    )
    return parser.parse_args()


def ensure_dependencies() -> tuple[Any, Any, Any]:
    try:
        import onnx  # noqa: F401
    except ImportError as exc:
        raise SystemExit("Install `onnx` before running this export script.") from exc

    try:
        from onnxconverter_common import float16
    except ImportError as exc:
        raise SystemExit(
            "Install `onnxconverter-common` for FP16 conversion."
        ) from exc

    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError as exc:
        raise SystemExit(
            "Install `onnxruntime` for INT8 quantization."
        ) from exc

    return float16, QuantType, quantize_dynamic


def checkpoint_path(model_dir: Path, cls_name: str, arch: str, encoder: str) -> Path:
    return model_dir / f"mayascan_v2_{cls_name}_{arch}_{encoder}.pth"


def export_class_model(
    class_id: int,
    cls_name: str,
    model_path: Path,
    output_dir: Path,
    arch: str,
    encoder: str,
    tile_size: int,
    opset: int,
    float16_module: Any,
    quant_type: Any,
    quantize_dynamic: Any,
) -> dict[str, Any]:
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {model_path}")

    model = _load_v2_model(str(model_path), arch=arch, encoder=encoder, device=torch.device("cpu"))
    model.cpu()
    model.eval()

    dummy_input = torch.randn(1, 3, tile_size, tile_size, dtype=torch.float32)
    stem = f"mayascan_v2_{cls_name}_{arch}_{encoder}"
    fp32_path = output_dir / f"{stem}.onnx"
    fp16_path = output_dir / f"{stem}_fp16.onnx"
    int8_path = output_dir / f"{stem}_int8.onnx"

    print(f"[{cls_name}] exporting FP32 ONNX -> {fp32_path}")
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            fp32_path,
            input_names=["input"],
            output_names=["output"],
            opset_version=opset,
            do_constant_folding=True,
        )

    print(f"[{cls_name}] converting -> FP16")
    fp16_model = float16_module.convert_float_to_float16_model_path(str(fp32_path))
    onnx.save(fp16_model, str(fp16_path))

    print(f"[{cls_name}] quantizing -> INT8")
    quantize_dynamic(
        str(fp32_path),
        str(int8_path),
        weight_type=quant_type.QInt8,
    )

    return {
        "class_id": class_id,
        "checkpoint": model_path.name,
        "fp32": fp32_path.name,
        "fp16": fp16_path.name,
        "int8": int8_path.name,
    }


def main() -> None:
    args = parse_args()
    float16_module, quant_type, quantize_dynamic = ensure_dependencies()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "arch": args.arch,
        "encoder": args.encoder,
        "tile_size": args.tile_size,
        "classes": {},
    }

    for class_id, cls_name in V2_CLASSES.items():
        exported = export_class_model(
            class_id=class_id,
            cls_name=cls_name,
            model_path=checkpoint_path(args.model_dir, cls_name, args.arch, args.encoder),
            output_dir=args.output_dir,
            arch=args.arch,
            encoder=args.encoder,
            tile_size=args.tile_size,
            opset=args.opset,
            float16_module=float16_module,
            quant_type=quant_type,
            quantize_dynamic=quantize_dynamic,
        )
        manifest["classes"][cls_name] = exported

    manifest_path = args.output_dir / "models-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
