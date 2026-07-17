"""
Replace eye-fill blue in GIFs with orange-gold colors sampled from colours/orange.png.

Unlike make_eye_transparent.py (chroma-key holes), this keeps the iris complete by
recoloring blue pixels in place. GIF duration / disposal handling follows enhance_media.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def is_eye_blue(arr: np.ndarray) -> np.ndarray:
    """
    Detect all iris blues: bright cyan, mid blue, and dark navy rim.
    Pure black / warm-brown eyelids are left alone.
    """
    r = arr[:, :, 0].astype(np.int16)
    g = arr[:, :, 1].astype(np.int16)
    b = arr[:, :, 2].astype(np.int16)
    rf, gf, bf = r.astype(np.float32), g.astype(np.float32), b.astype(np.float32)

    mx = np.maximum(np.maximum(rf, gf), bf)
    mn = np.minimum(np.minimum(rf, gf), bf)
    diff = np.maximum(mx - mn, 1e-6)
    sat = np.where(mx == 0, 0.0, (mx - mn) / np.maximum(mx, 1.0))

    hue = np.zeros_like(mx)
    mask_r = mx == rf
    mask_g = (mx == gf) & ~mask_r
    mask_b = ~mask_r & ~mask_g
    hue[mask_r] = (60.0 * (((gf - bf) / diff) % 6.0))[mask_r]
    hue[mask_g] = (60.0 * (((bf - rf) / diff) + 2.0))[mask_g]
    hue[mask_b] = (60.0 * (((rf - gf) / diff) + 4.0))[mask_b]

    # Mid / bright iris fill
    is_blue_hue = (hue >= 165.0) & (hue <= 250.0)
    is_fill = is_blue_hue & (
        ((sat > 0.12) & (mx > 70.0))
        | ((bf > 160.0) & (gf > 140.0) & (rf < gf) & ((bf - rf) > 12.0) & (sat > 0.03))
    )

    # Dark navy / indigo / cool-purple pupil rim
    is_dark_navy = (
        (bf >= rf - 2)
        & (bf >= gf - 12)
        & (bf >= 22)
        & (mx <= 145)
        & (mx >= 15)
        & (rf < 110)
        & ((bf > rf + 3) | ((sat > 0.05) & (bf >= rf)))
    )

    is_eye = is_fill | is_dark_navy
    is_eye = is_eye & ~((r > 230) & (g > 240) & (b > 240))
    # Skip warm-dark eyelid browns (clearly R-led, not cool/blue)
    is_eye = is_eye & ~((rf > bf + 4) & (rf >= gf) & (mx < 80))
    return is_eye


def load_colormap(path: Path, size: tuple[int, int]) -> np.ndarray:
    """Load colour map and resize to (width, height) → RGB uint8 array [H,W,3]."""
    img = Image.open(path).convert("RGB")
    if img.size != size:
        img = img.resize(size, Image.Resampling.LANCZOS)
    return np.asarray(img, dtype=np.uint8)


def replace_blue_with_colormap(frame: Image.Image, colormap: np.ndarray) -> Image.Image:
    """
    Replace eye-blue with orange-gold.
    Dark navy → deep brown, mid → orange, bright → gold (黑→棕→橘→金).
    """
    arr = np.asarray(frame.convert("RGBA"), dtype=np.uint8).copy()
    h, w = arr.shape[:2]
    if colormap.shape[0] != h or colormap.shape[1] != w:
        raise ValueError(f"colormap size {colormap.shape[:2]} != frame {(h, w)}")

    mask = is_eye_blue(arr)
    if not mask.any():
        return Image.fromarray(arr, "RGBA")

    src = arr[mask, :3].astype(np.float32)
    lum = 0.299 * src[:, 0] + 0.587 * src[:, 1] + 0.114 * src[:, 2]
    mapped = colormap[mask].astype(np.float32)

    # 0 = shadow brown, 1+ = full gold from colour map
    t = np.clip(lum / 150.0, 0.0, 1.25)
    brown = np.array([78.0, 38.0, 10.0], dtype=np.float32)
    brown_w = np.clip(1.0 - t / 0.55, 0.0, 1.0)[:, None]

    factor = np.clip(0.12 + t * 0.95, 0.10, 1.40)[:, None]
    warm = mapped * factor
    out_rgb = warm * (1.0 - brown_w * 0.92) + brown * (brown_w * 0.92)
    out_rgb *= np.clip(lum / 100.0, 0.25, 1.15)[:, None]

    # Kill residual blue cast — keep warm R≥B
    out_rgb[:, 2] = np.minimum(out_rgb[:, 2], out_rgb[:, 0] * 0.55)
    out_rgb[:, 1] = np.minimum(out_rgb[:, 1], out_rgb[:, 0] * 0.85 + 20.0)

    arr[mask, :3] = np.clip(out_rgb, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGBA")


def _quantize_for_gif(
    img: Image.Image,
    *,
    colors: int,
    dither: Image.Dither = Image.Dither.NONE,
) -> Image.Image:
    if img.mode == "RGBA":
        methods = (Image.Quantize.LIBIMAGEQUANT, Image.Quantize.FASTOCTREE)
    else:
        methods = (
            Image.Quantize.LIBIMAGEQUANT,
            Image.Quantize.MEDIANCUT,
            Image.Quantize.FASTOCTREE,
        )
    last_exc: Exception | None = None
    for method in methods:
        try:
            return img.quantize(colors=colors, method=method, dither=dither)
        except Exception as exc:
            last_exc = exc
    assert last_exc is not None
    raise last_exc


def _iter_gif_composited(img: Image.Image):
    n_frames = getattr(img, "n_frames", 1)
    default_duration = img.info.get("duration", 80)
    default_disposal = img.info.get("disposal", 0)
    for i in range(n_frames):
        img.seek(i)
        frame = img.convert("RGBA")
        duration = int(img.info.get("duration", default_duration))
        disposal = int(img.info.get("disposal", default_disposal))
        yield frame, duration, disposal


def _p_image_from(arr: np.ndarray, palette: list[int]) -> Image.Image:
    while len(palette) < 768:
        palette.append(0)
    out = Image.fromarray(np.asarray(arr, dtype=np.uint8), mode="P")
    out.putpalette(palette[:768], rawmode="RGB")
    return out


def _quantize_rgb_frame(frame: Image.Image) -> Image.Image:
    rgb = frame.convert("RGB")
    q = _quantize_for_gif(rgb, colors=256, dither=Image.Dither.NONE)
    palette = list(q.getpalette() or [])
    return _p_image_from(np.asarray(q, dtype=np.uint8), palette)


def _uniquify_identical_frames(frames: list[Image.Image]) -> None:
    """Prevent Pillow from merging identical consecutive frames (preserves duration)."""
    prev: bytes | None = None
    for i, frame in enumerate(frames):
        cur = np.asarray(frame.convert("RGB")).tobytes()
        if prev is not None and cur == prev:
            arr = np.asarray(frame, dtype=np.uint8).copy()
            palette = list(frame.getpalette() or [])
            while len(palette) < 768:
                palette.append(0)
            y = x = 0
            src_idx = int(arr[y, x])
            used = set(np.unique(arr).tolist())
            free = next((idx for idx in range(256) if idx not in used), 255)
            r, g, b = palette[src_idx * 3 : src_idx * 3 + 3]
            palette[free * 3 : free * 3 + 3] = [(r + 1) % 256, g, b]
            arr[y, x] = free
            frames[i] = _p_image_from(arr, palette)
            cur = np.asarray(frames[i].convert("RGB")).tobytes()
        prev = cur


def _finalize_frames(frames: list[Image.Image]) -> list[Image.Image]:
    out: list[Image.Image] = []
    for frame in frames:
        palette = list(frame.getpalette() or [])
        out.append(_p_image_from(np.asarray(frame, dtype=np.uint8), palette))
    return out


def process_gif(src: Path, dst: Path, colormap_path: Path) -> dict:
    with Image.open(src) as img:
        loop = img.info.get("loop", 0)
        size = img.size
        colormap = load_colormap(colormap_path, size)

        frames: list[Image.Image] = []
        durations: list[int] = []
        replaced_total = 0

        for composited, duration, _disposal in _iter_gif_composited(img):
            recolored = replace_blue_with_colormap(composited, colormap)
            mask = is_eye_blue(np.asarray(composited.convert("RGBA")))
            replaced_total += int(mask.sum())
            frames.append(_quantize_rgb_frame(recolored))
            durations.append(duration)

    if not frames:
        raise ValueError("empty gif")

    _uniquify_identical_frames(frames)
    frames = _finalize_frames(frames)

    # Full composited frames — clear canvas each frame for stable playback.
    disposals = [2] * len(frames)
    duration_kw: int | tuple[int, ...] = (
        durations[0] if len(set(durations)) == 1 else tuple(durations)
    )
    disposal_kw: int | tuple[int, ...] = (
        disposals[0] if len(set(disposals)) == 1 else tuple(disposals)
    )

    dst.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        dst,
        save_all=True,
        append_images=frames[1:],
        loop=loop,
        duration=duration_kw,
        disposal=disposal_kw,
        optimize=False,
    )
    return {
        "frames": len(frames),
        "replaced_pixels": replaced_total,
        "durations": durations,
    }


def main() -> int:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Replace eye blue with orange-gold from a colour map PNG."
    )
    parser.add_argument("--src", type=Path, default=root / "EyeExpressionsVideo")
    parser.add_argument("--dst", type=Path, default=root / "EyeExpressionsVideoOrange")
    parser.add_argument("--colors", type=Path, default=root / "colours" / "orange.png")
    parser.add_argument("--limit", type=int, default=0, help="Process at most N files (0=all).")
    args = parser.parse_args()

    if not args.colors.is_file():
        print(f"Colour map not found: {args.colors}", file=sys.stderr)
        return 1

    gifs = sorted(args.src.rglob("*.gif"))
    if args.limit > 0:
        gifs = gifs[: args.limit]
    if not gifs:
        print(f"No GIFs under {args.src}", file=sys.stderr)
        return 1

    print(f"Colour map: {args.colors}")
    print(f"Processing {len(gifs)} GIFs: {args.src} -> {args.dst}")
    ok = 0
    for i, src in enumerate(gifs, 1):
        rel = src.relative_to(args.src)
        dst = args.dst / rel
        try:
            info = process_gif(src, dst, args.colors)
            ok += 1
            durs = info["durations"]
            dur_desc = (
                f"{durs[0]}ms"
                if len(set(durs)) == 1
                else f"mixed({min(durs)}-{max(durs)}ms)"
            )
            print(
                f"[{i}/{len(gifs)}] {rel}  frames={info['frames']} "
                f"duration={dur_desc} replaced={info['replaced_pixels']}"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[{i}/{len(gifs)}] FAIL {rel}: {exc}", file=sys.stderr)

    print(f"Done: {ok}/{len(gifs)} succeeded")
    return 0 if ok == len(gifs) else 2


if __name__ == "__main__":
    raise SystemExit(main())
