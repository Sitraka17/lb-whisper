#!/usr/bin/env python3
"""ONNX export for the browser demo (web/index.html, transformers.js on WebGPU).

    python export_onnx.py --model SitrakaForler/whisper-large-v3-turbo-lb --out runs/onnx

Writes, in the layout transformers.js expects:
    <out>/onnx/encoder_model_q4.onnx           about 425 MB, 4-bit MatMul weights
    <out>/onnx/encoder_model_fp16.onnx         about 1.3 GB
    <out>/onnx/decoder_model_merged_q4.onnx    about 334 MB
plus the config, generation, preprocessor and tokenizer files.

Pinned environment (Python 3.12), the one that worked on 25/10/2026: torch==2.5.1,
transformers==4.49.0, onnx==1.17.0, onnxruntime==1.20.1, and optimum at commit
b04feaea78cda58d79b8da67dca3fd0c4ab33435. Newer torch exports through dynamo and fails here.

Review note: the merged decoder keeps the weights inside its two If branches in fp32 (the 4-bit
quantizer skips subgraphs), hence 334 MB instead of about 200. Works, just heavier than it could be.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import onnx
from huggingface_hub import hf_hub_download, list_repo_files
from onnxruntime.quantization.matmul_4bits_quantizer import MatMul4BitsQuantizer
from onnxruntime.transformers.float16 import convert_float_to_float16

TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "processor_config.json")


def quantize(fp32_dir: Path, onnx_dir: Path) -> None:
    onnx_dir.mkdir(parents=True, exist_ok=True)
    for name in ("encoder_model", "decoder_model_merged"):
        quantizer = MatMul4BitsQuantizer(onnx.load(str(fp32_dir / f"{name}.onnx")), block_size=32, is_symmetric=True)
        quantizer.process()
        onnx.save(quantizer.model.model, str(onnx_dir / f"{name}_q4.onnx"))
        print(f"[onnx] {name}_q4.onnx")
    fp16 = convert_float_to_float16(onnx.load(str(fp32_dir / "encoder_model.onnx")), keep_io_types=True, disable_shape_infer=True)
    onnx.save(fp16, str(onnx_dir / "encoder_model_fp16.onnx"))
    print("[onnx] encoder_model_fp16.onnx")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="SitrakaForler/whisper-large-v3-turbo-lb")
    ap.add_argument("--out", type=Path, default=Path("runs/onnx"))
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        fp32 = Path(tmp) / "fp32"
        subprocess.run(["optimum-cli", "export", "onnx", "--model", args.model,
                        "--task", "automatic-speech-recognition-with-past", "--opset", "17", str(fp32)], check=True)
        quantize(fp32, args.out / "onnx")
        for cfg in fp32.glob("*.json"):
            shutil.copy(cfg, args.out / cfg.name)
    # optimum does not copy the tokenizer files; transformers.js needs them next to the config.
    available = set(list_repo_files(args.model)) if not Path(args.model).is_dir() else set()
    for name in TOKENIZER_FILES:
        local = Path(args.model) / name
        if local.exists():
            shutil.copy(local, args.out / name)
        elif name in available:
            shutil.copy(hf_hub_download(args.model, name), args.out / name)
    print(f"[onnx] wrote {args.out}")


if __name__ == "__main__":
    main()
