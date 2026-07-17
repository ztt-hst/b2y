#!/usr/bin/env python3
"""Convert blue tones in eye-tracking background images to cat-eye golden yellow.

Preserves gradients by remapping hue in HSL while keeping lightness and saturation.
Neutral / low-saturation pixels (whites and grays) are left unchanged.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def _rgb_to_hsl(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized RGB [0,1] -> HSL. Hue in [0,1)."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    max_c = np.max(rgb, axis=-1)
    min_c = np.min(rgb, axis=-1)
    delta = max_c - min_c

    lightness = (max_c + min_c) / 2.0

    saturation = np.zeros_like(lightness)
    mask = delta > 1e-8
    sat_denom = 1.0 - np.abs(2.0 * lightness - 1.0)
    saturation[mask] = delta[mask] / np.maximum(sat_denom[mask], 1e-8)

    hue = np.zeros_like(lightness)
    r_eq_max = (max_c == r) & mask
    g_eq_max = (max_c == g) & mask
    b_eq_max = (max_c == b) & mask

    hue[r_eq_max] = ((g[r_eq_max] - b[r_eq_max]) / delta[r_eq_max]) % 6.0
    hue[g_eq_max] = ((b[g_eq_max] - r[g_eq_max]) / delta[g_eq_max]) + 2.0
    hue[b_eq_max] = ((r[b_eq_max] - g[b_eq_max]) / delta[b_eq_max]) + 4.0
    hue = hue / 6.0

    return hue, saturation, lightness


def _hsl_to_rgb(hue: np.ndarray, saturation: np.ndarray, lightness: np.ndarray) -> np.ndarray:
    """Vectorized HSL -> RGB [0,1]."""
    chroma = (1.0 - np.abs(2.0 * lightness - 1.0)) * saturation
    hue_scaled = hue * 6.0
    x = chroma * (1.0 - np.abs(hue_scaled % 2.0 - 1.0))
    m = lightness - chroma / 2.0
    sector = np.floor(hue_scaled).astype(np.int32) % 6

    zero = np.zeros_like(chroma)
    r = np.select(
        [sector == 0, sector == 1, sector == 2, sector == 3, sector == 4, sector == 5],
        [chroma, x, zero, zero, x, chroma],
        default=zero,
    )
    g = np.select(
        [sector == 0, sector == 1, sector == 2, sector == 3, sector == 4, sector == 5],
        [x, chroma, chroma, x, zero, zero],
        default=zero,
    )
    b = np.select(
        [sector == 0, sector == 1, sector == 2, sector == 3, sector == 4, sector == 5],
        [zero, zero, x, chroma, chroma, x],
        default=zero,
    )

    return np.clip(np.stack([r, g, b], axis=-1) + m[..., None], 0.0, 1.0)


def _remap_hue(
    hue: np.ndarray,
    src_start: float,
    src_end: float,
    dst_start: float,
    dst_end: float,
) -> np.ndarray:
    """Linearly map hue from source range to destination range."""
    src_span = src_end - src_start
    t = np.clip((hue - src_start) / src_span, 0.0, 1.0)
    return (dst_start + t * (dst_end - dst_start)) % 1.0


def _smoothstep(edge0: float, edge1: float, value: np.ndarray) -> np.ndarray:
    """Hermite interpolation from 0 to 1 between edge0 and edge1."""
    if edge0 == edge1:
        return np.zeros_like(value)
    t = np.clip((value - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _circular_hue_distance(hue: np.ndarray, center: float) -> np.ndarray:
    distance = np.abs(hue - center)
    return np.minimum(distance, 1.0 - distance)


def _blue_hue_weight(hue: np.ndarray) -> np.ndarray:
    """Soft weight peaking at blue/cyan/violet/magenta hues instead of a hard mask."""
    blue_weight = _smoothstep(0.20, 0.0, _circular_hue_distance(hue, 0.58))
    cyan_weight = _smoothstep(0.16, 0.0, _circular_hue_distance(hue, 0.50))
    violet_weight = _smoothstep(0.16, 0.0, _circular_hue_distance(hue, 0.72))
    magenta_weight = _smoothstep(0.18, 0.0, _circular_hue_distance(hue, 0.83))
    return np.clip(
        np.maximum.reduce([blue_weight, cyan_weight, violet_weight, magenta_weight]),
        0.0,
        1.0,
    )


def _blue_dominance_weight(rgb: np.ndarray) -> np.ndarray:
    """Soft weight for pixels where blue is slightly stronger than red/green."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    max_c = np.max(rgb, axis=-1)
    dominance = (b - np.maximum(r, g)) / (18.0 / 255.0)
    visible = _smoothstep(12.0 / 255.0, 28.0 / 255.0, max_c)
    return np.clip(dominance, 0.0, 1.0) * visible


def _conversion_weight(
    hue: np.ndarray,
    saturation: np.ndarray,
    rgb: np.ndarray,
) -> np.ndarray:
    """Return a continuous 0-1 blend factor for blue -> gold conversion."""
    hue_weight = _blue_hue_weight(hue)
    sat_weight = _smoothstep(0.01, 0.10, saturation)
    dom_weight = _blue_dominance_weight(rgb)
    raw_weight = np.maximum(
        hue_weight * np.maximum(sat_weight, 0.35),
        np.maximum(dom_weight * hue_weight, dom_weight * 0.75),
    )
    return np.clip(_smoothstep(0.0, 0.35, raw_weight), 0.0, 1.0)


def _soften_blue_cast(rgb: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Gradually remove residual blue without creating hard edges."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    excess_blue = np.clip(b - np.maximum(r, g), 0.0, None)
    warm_r = r + excess_blue * 0.35
    warm_g = g + excess_blue * 0.18
    warm_b = b - excess_blue * 0.85
    blend = weight[..., None]
    warm = np.stack([warm_r, warm_g, warm_b], axis=-1)
    return np.clip(rgb * (1.0 - blend) + warm * blend, 0.0, 1.0)


def blue_to_gold(
    image: Image.Image,
    *,
    sat_threshold: float = 0.03,
    src_hue_start: float = 0.40,
    src_hue_end: float = 0.88,
    dst_hue_start: float = 0.08,
    dst_hue_end: float = 0.14,
    low_sat_dst_hue_start: float = 0.06,
    low_sat_dst_hue_end: float = 0.10,
    low_sat_threshold: float = 0.15,
    sat_boost: float = 1.05,
) -> Image.Image:
    """Convert blue pixels to golden yellow while preserving L/S gradients."""
    del sat_threshold  # kept for CLI compatibility

    rgba = np.array(image.convert("RGBA"), dtype=np.float64)
    rgb = rgba[..., :3] / 255.0
    alpha = rgba[..., 3]

    hue, saturation, lightness = _rgb_to_hsl(rgb)
    weight = _conversion_weight(hue, saturation, rgb)

    sat_mix = _smoothstep(0.0, low_sat_threshold, saturation)
    target_start = low_sat_dst_hue_start + sat_mix * (dst_hue_start - low_sat_dst_hue_start)
    target_end = low_sat_dst_hue_end + sat_mix * (dst_hue_end - low_sat_dst_hue_end)

    remapped_hue = _remap_hue(
        hue,
        src_hue_start,
        src_hue_end,
        target_start,
        target_end,
    )
    new_hue = hue + (remapped_hue - hue) * weight

    boosted_saturation = np.clip(saturation * (1.0 + (sat_boost - 1.0) * weight), 0.0, 1.0)
    low_sat_floor = np.maximum(saturation * 0.92, 0.02)
    new_saturation = saturation + (boosted_saturation - saturation) * weight
    new_saturation = np.where(
        saturation < low_sat_threshold,
        saturation + (low_sat_floor - saturation) * weight,
        new_saturation,
    )

    converted_rgb = _hsl_to_rgb(new_hue, new_saturation, lightness)
    converted_rgb = _soften_blue_cast(converted_rgb, weight)
    out_rgb = rgb * (1.0 - weight[..., None]) + converted_rgb * weight[..., None]
    out = np.zeros_like(rgba)
    out[..., :3] = np.round(out_rgb * 255.0)
    out[..., 3] = alpha

    return Image.fromarray(out.astype(np.uint8), mode="RGBA")


IMAGE_EXTENSIONS = {".png", ".gif"}


def _convert_kwargs(
    *,
    sat_threshold: float,
    sat_boost: float,
) -> dict[str, float]:
    return {
        "sat_threshold": sat_threshold,
        "sat_boost": sat_boost,
    }


def convert_png(
    input_path: Path,
    output_path: Path,
    **kwargs,
) -> Path:
    image = Image.open(input_path)
    result = blue_to_gold(image, **kwargs)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path)
    return output_path


def convert_gif(
    input_path: Path,
    output_path: Path,
    **kwargs,
) -> Path:
    image = Image.open(input_path)
    frames: list[Image.Image] = []
    durations: list[int] = []
    disposals: list[int | None] = []
    loop = image.info.get("loop", 0)

    for frame_index in range(image.n_frames):
        image.seek(frame_index)
        frames.append(blue_to_gold(image.convert("RGBA"), **kwargs))
        durations.append(image.info.get("duration", 100))
        disposals.append(image.info.get("disposal"))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs: dict[str, object] = {
        "save_all": True,
        "append_images": frames[1:],
        "duration": durations,
        "loop": loop,
        "optimize": False,
    }
    if any(disposal is not None for disposal in disposals):
        save_kwargs["disposal"] = disposals

    frames[0].save(output_path, **save_kwargs)
    return output_path


def convert_file(
    input_path: Path,
    output_path: Path | None = None,
    *,
    overwrite: bool = False,
    **kwargs,
) -> Path:
    if output_path is None:
        if overwrite:
            output_path = input_path
        else:
            output_path = input_path.with_name(f"{input_path.stem}_gold{input_path.suffix}")

    suffix = input_path.suffix.lower()
    if suffix == ".gif":
        return convert_gif(input_path, output_path, **kwargs)
    if suffix == ".png":
        return convert_png(input_path, output_path, **kwargs)
    raise ValueError(f"Unsupported file type: {input_path}")


def iter_media_files(root: Path) -> list[Path]:
    files = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(files)


def process_directories(
    source_dirs: list[Path],
    output_dir: Path,
    *,
    project_root: Path | None = None,
    **kwargs,
) -> list[Path]:
    if project_root is None:
        project_root = Path.cwd()

    output_paths: list[Path] = []
    for source_dir in source_dirs:
        source_dir = source_dir.resolve()
        if not source_dir.is_dir():
            raise FileNotFoundError(f"Source directory not found: {source_dir}")

        for input_path in iter_media_files(source_dir):
            relative_path = input_path.relative_to(project_root.resolve())
            output_path = output_dir / relative_path
            convert_file(input_path, output_path, **kwargs)
            output_paths.append(output_path)
            print(f"Saved: {output_path}")

    return output_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert blue eye backgrounds to cat-eye golden yellow."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help="Input image path(s), e.g. bg_1.png",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output path (only valid for a single input file)",
    )
    parser.add_argument(
        "--batch",
        nargs="+",
        type=Path,
        metavar="DIR",
        help="Batch-convert all PNG/GIF files under these directories",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Output root for batch mode (default: output)",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="Project root used to mirror folder structure in batch mode",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite input file in place",
    )
    parser.add_argument(
        "--sat-threshold",
        type=float,
        default=0.03,
        help="Minimum saturation to treat a pixel as colored (default: 0.03)",
    )
    parser.add_argument(
        "--sat-boost",
        type=float,
        default=1.05,
        help="Saturation multiplier for converted pixels (default: 1.05)",
    )
    args = parser.parse_args()
    convert_options = _convert_kwargs(
        sat_threshold=args.sat_threshold,
        sat_boost=args.sat_boost,
    )

    if args.batch:
        if args.output is not None:
            parser.error("--output cannot be used with --batch")
        process_directories(
            args.batch,
            args.output_dir,
            project_root=args.project_root,
            **convert_options,
        )
        return

    if not args.inputs:
        process_directories(
            [Path("EyeTrackingFrames"), Path("EyeExpressionsVideo")],
            args.output_dir,
            project_root=args.project_root,
            **convert_options,
        )
        return

    if args.output is not None and len(args.inputs) != 1:
        parser.error("--output can only be used with a single input file")

    for input_path in args.inputs:
        out = convert_file(
            input_path,
            args.output,
            overwrite=args.overwrite,
            **convert_options,
        )
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
