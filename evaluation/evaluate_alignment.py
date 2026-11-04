"""Evaluate stitching alignment without modifying the Warp/Fusion projects.

PSNR, SSIM and LPIPS are measured only inside the valid overlap of warp1 and
warp2. JPEG ringing in saved masks is removed with a fixed threshold.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import binary_erosion
from skimage.metrics import structural_similarity


FIELDNAMES = [
    "image_id",
    "psnr",
    "ssim",
    "lpips",
    "overlap_pixels",
    "overlap_ratio",
    "valid",
    "error",
]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
ALEXNET_FILENAME = "alexnet-owt-7be5be79.pth"
ALEXNET_SHA256 = "7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02"
LOCAL_TORCH_HOME = Path(__file__).resolve().parent / "third_party" / "torch"
LOCAL_PYTHON_DEPS = Path(__file__).resolve().parent / "third_party" / "python"

if LOCAL_PYTHON_DEPS.is_dir():
    sys.path.insert(0, str(LOCAL_PYTHON_DEPS))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure PSNR/SSIM/LPIPS in the valid Warp overlap."
    )
    parser.add_argument(
        "--warp-root",
        type=Path,
        required=True,
        help="Directory containing warp1, warp2, mask1 and mask2.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name", required=True, help="Short model/result name.")
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="LPIPS device, for example cuda:0 or cpu.",
    )
    parser.add_argument(
        "--no-lpips",
        action="store_true",
        help="Only calculate PSNR and SSIM.",
    )
    parser.add_argument(
        "--alexnet-weights",
        type=Path,
        default=None,
        help=(
            "Optional local alexnet-owt-7be5be79.pth. It is SHA256-checked "
            "and copied into Evaluation/third_party/torch."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Evaluate only the first N images; 0 means all images.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep valid rows already present in the output CSV.",
    )
    parser.add_argument(
        "--mask-threshold",
        type=int,
        default=127,
        help="Threshold for JPEG-compressed mask images.",
    )
    parser.add_argument(
        "--min-overlap-pixels",
        type=int,
        default=1024,
        help="Samples below this valid overlap size are reported as invalid.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_alexnet_weights(source: Path | None) -> Path:
    cache_dir = LOCAL_TORCH_HOME / "hub" / "checkpoints"
    target = cache_dir / ALEXNET_FILENAME
    if source is not None:
        source = source.resolve()
        if not source.is_file():
            raise FileNotFoundError(f"AlexNet weights not found: {source}")
        actual_hash = sha256(source)
        if actual_hash != ALEXNET_SHA256:
            raise RuntimeError(
                f"Invalid AlexNet SHA256: {actual_hash}; expected {ALEXNET_SHA256}"
            )
        cache_dir.mkdir(parents=True, exist_ok=True)
        if source != target.resolve():
            shutil.copy2(source, target)

    if not target.is_file():
        raise FileNotFoundError(
            "LPIPS requires the official AlexNet weights. Download "
            f"{ALEXNET_FILENAME}, then pass --alexnet-weights PATH. "
            "Random weights are intentionally forbidden."
        )
    actual_hash = sha256(target)
    if actual_hash != ALEXNET_SHA256:
        raise RuntimeError(
            f"Cached AlexNet SHA256 is invalid: {actual_hash}; "
            f"expected {ALEXNET_SHA256}"
        )
    os.environ["TORCH_HOME"] = str(LOCAL_TORCH_HOME)
    return target


def list_common_images(root: Path) -> list[str]:
    folders = [root / name for name in ("warp1", "warp2", "mask1", "mask2")]
    missing = [str(path) for path in folders if not path.is_dir()]
    if missing:
        raise FileNotFoundError("Missing required directories: " + ", ".join(missing))

    name_sets = []
    for folder in folders:
        name_sets.append(
            {
                path.name
                for path in folder.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            }
        )
    common = set.intersection(*name_sets)
    if not common:
        raise RuntimeError(f"No matching image names found under {root}")
    return sorted(common)


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def read_mask(path: Path, threshold: int) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image.convert("L"), dtype=np.uint8)
    return array > threshold


def validate_shapes(
    image1: np.ndarray,
    image2: np.ndarray,
    mask1: np.ndarray,
    mask2: np.ndarray,
) -> None:
    shape = image1.shape[:2]
    if image2.shape[:2] != shape or mask1.shape != shape or mask2.shape != shape:
        raise ValueError(
            f"Shape mismatch: warp1={image1.shape}, warp2={image2.shape}, "
            f"mask1={mask1.shape}, mask2={mask2.shape}"
        )


def masked_psnr(image1: np.ndarray, image2: np.ndarray, mask: np.ndarray) -> float:
    diff = image1[mask] - image2[mask]
    mse = float(np.mean(np.square(diff), dtype=np.float64))
    if mse == 0:
        return math.inf
    return 10.0 * math.log10(1.0 / mse)


def masked_ssim(image1: np.ndarray, image2: np.ndarray, mask: np.ndarray) -> float:
    _, score_map = structural_similarity(
        image1,
        image2,
        data_range=1.0,
        channel_axis=-1,
        full=True,
    )
    if score_map.ndim == 3:
        score_map = score_map.mean(axis=2)

    # A 7x7 SSIM window must remain completely inside the overlap.
    inner_mask = binary_erosion(mask, iterations=3, border_value=0)
    if int(inner_mask.sum()) < 64:
        inner_mask = mask
    return float(np.mean(score_map[inner_mask], dtype=np.float64))


def overlap_crop(
    image1: np.ndarray,
    image2: np.ndarray,
    mask: np.ndarray,
    padding: int = 16,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ys, xs = np.nonzero(mask)
    y0 = max(0, int(ys.min()) - padding)
    y1 = min(mask.shape[0], int(ys.max()) + 1 + padding)
    x0 = max(0, int(xs.min()) - padding)
    x1 = min(mask.shape[1], int(xs.max()) + 1 + padding)

    crop1 = image1[y0:y1, x0:x1].copy()
    crop2 = image2[y0:y1, x0:x1].copy()
    crop_mask = mask[y0:y1, x0:x1]

    # Make invalid pixels identical so they cannot create a false LPIPS penalty.
    neutral = 0.5 * (crop1 + crop2)
    crop1[~crop_mask] = neutral[~crop_mask]
    crop2[~crop_mask] = neutral[~crop_mask]
    return crop1, crop2, crop_mask


def to_lpips_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))
    return tensor.unsqueeze(0).to(device=device, dtype=torch.float32) * 2.0 - 1.0


@torch.inference_mode()
def masked_lpips(
    model: torch.nn.Module,
    image1: np.ndarray,
    image2: np.ndarray,
    mask: np.ndarray,
    device: torch.device,
) -> float:
    crop1, crop2, crop_mask = overlap_crop(image1, image2, mask)
    if min(crop_mask.shape) < 32:
        raise ValueError(f"LPIPS overlap crop is too narrow: {crop_mask.shape}")

    tensor1 = to_lpips_tensor(crop1, device)
    tensor2 = to_lpips_tensor(crop2, device)
    distance_map = model(tensor1, tensor2, normalize=False)
    if distance_map.shape[-2:] != crop_mask.shape:
        distance_map = F.interpolate(
            distance_map,
            size=crop_mask.shape,
            mode="bilinear",
            align_corners=False,
        )
    mask_tensor = torch.from_numpy(crop_mask.astype(np.float32)).to(device)
    mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
    denominator = mask_tensor.sum().clamp_min(1.0)
    return float((distance_map * mask_tensor).sum().div(denominator).item())


def load_existing(
    csv_path: Path, require_lpips: bool
) -> dict[str, dict[str, str]]:
    if not csv_path.exists():
        return {}
    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = {}
        for row in csv.DictReader(handle):
            if not row.get("image_id") or row.get("valid") != "1":
                continue
            if require_lpips:
                try:
                    if not math.isfinite(float(row.get("lpips", ""))):
                        continue
                except (TypeError, ValueError):
                    continue
            rows[row["image_id"]] = row
        return rows


def finite_values(rows: list[dict[str, object]], key: str) -> np.ndarray:
    values = []
    for row in rows:
        try:
            value = float(row[key])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return np.asarray(values, dtype=np.float64)


def describe(values: np.ndarray) -> dict[str, float | int | None]:
    if values.size == 0:
        return {"count": 0, "mean": None, "median": None, "std": None}
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def write_outputs(
    rows: list[dict[str, object]],
    csv_path: Path,
    summary_path: Path,
    args: argparse.Namespace,
) -> None:
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    valid_count = sum(str(row["valid"]) == "1" for row in rows)
    summary = {
        "name": args.name,
        "warp_root": str(args.warp_root.resolve()),
        "protocol": {
            "mask_threshold": args.mask_threshold,
            "min_overlap_pixels": args.min_overlap_pixels,
            "psnr": "RGB pixels inside mask1 AND mask2",
            "ssim": "mean SSIM map inside a 3-pixel-eroded overlap mask",
            "lpips": (
                None
                if args.no_lpips
                else "LPIPS 0.1 AlexNet spatial map averaged inside overlap"
            ),
        },
        "images_total": len(rows),
        "images_valid": valid_count,
        "images_invalid": len(rows) - valid_count,
        "psnr": describe(finite_values(rows, "psnr")),
        "ssim": describe(finite_values(rows, "ssim")),
        "lpips": describe(finite_values(rows, "lpips")),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    args.warp_root = args.warp_root.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / f"{args.name}_alignment_per_image.csv"
    summary_path = args.output_dir / f"{args.name}_alignment_summary.json"

    names = list_common_images(args.warp_root)
    if args.limit > 0:
        names = names[: args.limit]
    existing = (
        load_existing(csv_path, require_lpips=not args.no_lpips)
        if args.resume
        else {}
    )

    device = torch.device(args.device)
    lpips_model = None
    if not args.no_lpips:
        prepare_alexnet_weights(args.alexnet_weights)
        try:
            import lpips
        except ImportError as error:
            raise RuntimeError(
                "LPIPS is not installed. Run: python -m pip install lpips"
            ) from error
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA requested but unavailable: {device}")
        lpips_model = lpips.LPIPS(net="alex", version="0.1", spatial=True)
        lpips_model = lpips_model.eval().to(device)

    rows: list[dict[str, object]] = []
    start = time.time()
    for index, name in enumerate(names, start=1):
        if name in existing:
            rows.append(existing[name])
            continue

        row: dict[str, object] = {
            "image_id": Path(name).stem,
            "psnr": "",
            "ssim": "",
            "lpips": "",
            "overlap_pixels": 0,
            "overlap_ratio": 0.0,
            "valid": 0,
            "error": "",
        }
        try:
            image1 = read_rgb(args.warp_root / "warp1" / name)
            image2 = read_rgb(args.warp_root / "warp2" / name)
            mask1 = read_mask(
                args.warp_root / "mask1" / name, args.mask_threshold
            )
            mask2 = read_mask(
                args.warp_root / "mask2" / name, args.mask_threshold
            )
            validate_shapes(image1, image2, mask1, mask2)
            overlap = mask1 & mask2
            overlap_pixels = int(overlap.sum())
            row["overlap_pixels"] = overlap_pixels
            row["overlap_ratio"] = overlap_pixels / float(overlap.size)
            if overlap_pixels < args.min_overlap_pixels:
                raise ValueError(
                    f"Too few overlap pixels: {overlap_pixels} "
                    f"< {args.min_overlap_pixels}"
                )

            row["psnr"] = masked_psnr(image1, image2, overlap)
            row["ssim"] = masked_ssim(image1, image2, overlap)
            if lpips_model is not None:
                row["lpips"] = masked_lpips(
                    lpips_model, image1, image2, overlap, device
                )
            row["valid"] = 1
        except Exception as error:  # Preserve every failed sample in the CSV.
            row["error"] = f"{type(error).__name__}: {error}"
        rows.append(row)

        if index % 25 == 0 or index == len(names):
            elapsed = time.time() - start
            print(
                f"[{index}/{len(names)}] elapsed={elapsed:.1f}s "
                f"latest={name} valid={row['valid']}",
                flush=True,
            )
            write_outputs(rows, csv_path, summary_path, args)

    write_outputs(rows, csv_path, summary_path, args)
    print(f"Per-image results: {csv_path}")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
