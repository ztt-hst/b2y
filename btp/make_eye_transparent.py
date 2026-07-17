"""
Make eye-fill blue pixels transparent in EyeExpressionsVideo GIFs,
so runtime background color can tint the eyes.

GIF duration / disposal handling follows enhance_media.py:
- seek composited frames (not raw partial updates)
- preserve per-frame duration exactly
- pass duration/disposal as int or tuple
- disposal=2 when using transparency (avoid ghosting)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image


TRANSPARENT_INDEX = 255


def is_eye_blue(arr: np.ndarray) -> np.ndarray:
    """Return boolean mask for iris / eye-fill blue (and light cyan) pixels."""
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

    is_blue_hue = (hue >= 170.0) & (hue <= 240.0)
    # Include darker navy blues inside the pupil fill; outline is restored later.
    is_eye = is_blue_hue & (
        ((sat > 0.15) & (mx > 55.0))
        | ((bf > 180.0) & (gf > 160.0) & (rf < gf) & ((bf - rf) > 20.0) & (sat > 0.05))
    )
    is_eye = is_eye & ~((sat < 0.12) & (mx < 160.0))
    is_eye = is_eye & ~((r > 230) & (g > 240) & (b > 240))
    # Keep near-black pupil body (not blue fill)
    is_eye = is_eye & ~((r + g + b) < 70)
    return is_eye


def _binary_dilate(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    out = mask.astype(bool)
    for _ in range(iterations):
        grown = out.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                grown |= np.roll(np.roll(out, dy, axis=0), dx, axis=1)
        out = grown
    return out


def _binary_erode(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    out = mask.astype(bool)
    for _ in range(iterations):
        shrunk = out.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                shrunk &= np.roll(np.roll(out, dy, axis=0), dx, axis=1)
        out = shrunk
    return out


def _binary_close(mask: np.ndarray, iterations: int = 2) -> np.ndarray:
    return _binary_erode(_binary_dilate(mask, iterations), iterations)


def _pupil_silhouette(arr: np.ndarray) -> np.ndarray:
    """
    Approximate pupil discs before keying so we can restore their outline.
    Takes dark / navy pixels that sit inside the blue eye area (not brown lids).
    """
    r = arr[:, :, 0].astype(np.int16)
    g = arr[:, :, 1].astype(np.int16)
    b = arr[:, :, 2].astype(np.int16)
    a = arr[:, :, 3]
    lum = r.astype(np.int32) + g.astype(np.int32) + b.astype(np.int32)
    opaque = a >= 128
    highlight = (r > 200) & (g > 200) & (b > 200)

    blue = is_eye_blue(arr)
    # Grow blue inward over the pupil; stay away from brown eyelids (r >> b)
    inside_eye = _binary_dilate(blue, 8)
    seed = (
        opaque
        & ~highlight
        & inside_eye
        & (lum < 300)
        & (b >= r - 5)  # navy/black pupil, not reddish-brown lid
        & (r < 140)
    )
    # Fill holes left by specular highlights
    return _binary_close(seed, 3)


def _restore_pupil_outline(arr: np.ndarray, pupil_mask: np.ndarray) -> np.ndarray:
    """
    After keying blues, put back a dark rim so pupils stay complete circles.
    Interior blues may stay transparent; only the contour is restored.
    """
    if not pupil_mask.any():
        return arr

    # Rim = pupil area minus its erosion (1-2px ring)
    rim = pupil_mask & ~_binary_erode(pupil_mask, 2)
    # Also heal single-pixel notches just inside the rim
    near_rim = _binary_dilate(rim, 1) & pupil_mask

    need = near_rim & (arr[:, :, 3] < 128)
    if not need.any():
        return arr

    # Sample a dark color from remaining pupil body (fallback to near-black)
    body = pupil_mask & (arr[:, :, 3] >= 128)
    if body.any():
        # Prefer darker remaining pixels for outline color
        lum = (
            arr[:, :, 0].astype(np.int32)
            + arr[:, :, 1].astype(np.int32)
            + arr[:, :, 2].astype(np.int32)
        )
        body_dark = body & (lum < 120)
        sample = body_dark if body_dark.any() else body
        ys, xs = np.where(sample)
        # median of a subset for stability
        idx = np.linspace(0, len(ys) - 1, num=min(64, len(ys)), dtype=int)
        color = np.median(arr[ys[idx], xs[idx], :3], axis=0).astype(np.uint8)
    else:
        color = np.array([20, 14, 18], dtype=np.uint8)

    arr = arr.copy()
    arr[need, 0] = color[0]
    arr[need, 1] = color[1]
    arr[need, 2] = color[2]
    arr[need, 3] = 255
    return arr


def key_blue_to_transparent(frame: Image.Image) -> Image.Image:
    """Key eye-blue to transparent, but keep pupil circular outlines intact."""
    arr = np.asarray(frame.convert("RGBA")).copy()
    pupil = _pupil_silhouette(arr)
    arr[is_eye_blue(arr), 3] = 0
    arr = _restore_pupil_outline(arr, pupil)
    return _binarize_alpha(Image.fromarray(arr, "RGBA"))


def _quantize_for_gif(
    img: Image.Image,
    *,
    colors: int,
    dither: Image.Dither = Image.Dither.NONE,
) -> Image.Image:
    """Pillow: RGBA only allows LIBIMAGEQUANT / FASTOCTREE; RGB may fall back to MEDIANCUT."""
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
    """
    Seek composited playback frames (not ImageSequence raw tiles).
    Preserves each frame's original duration / disposal.
    """
    n_frames = getattr(img, "n_frames", 1)
    default_duration = img.info.get("duration", 80)
    default_disposal = img.info.get("disposal", 0)

    for i in range(n_frames):
        img.seek(i)
        frame = img.convert("RGBA")
        duration = int(img.info.get("duration", default_duration))
        disposal = int(img.info.get("disposal", default_disposal))
        yield frame, duration, disposal


def _binarize_alpha(rgba: Image.Image, threshold: int = 128) -> Image.Image:
    """GIF only supports 1-bit transparency."""
    r, g, b, a = rgba.split()
    a = a.point(lambda x: 255 if x >= threshold else 0)
    return Image.merge("RGBA", (r, g, b, a))


def _apply_gif_transparency(
    p: Image.Image, alpha: Image.Image, trans_idx: int = TRANSPARENT_INDEX
) -> Image.Image:
    """Write alpha mask into P-mode transparency index."""
    a = np.asarray(alpha, dtype=np.uint8)
    px = np.asarray(p, dtype=np.uint8).copy()
    px[a < 128] = trans_idx
    out = Image.fromarray(px, mode="P")
    palette = list(p.getpalette() or [])
    while len(palette) < 768:
        palette.append(0)
    palette[trans_idx * 3 : trans_idx * 3 + 3] = [0, 0, 0]
    out.putpalette(palette[:768])
    out.info["transparency"] = trans_idx
    return out


def _quantize_rgba_frame(frame: Image.Image) -> Image.Image:
    rgba = frame if frame.mode == "RGBA" else frame.convert("RGBA")
    r, g, b, a = rgba.split()
    rgb = Image.merge("RGB", (r, g, b))
    q = _quantize_for_gif(rgb, colors=255, dither=Image.Dither.NONE)
    return _apply_gif_transparency(q, a)


def _uniquify_identical_frames(frames: list[Image.Image]) -> None:
    """
    Pillow merges consecutive identical frames and sums durations.
    Nudge one pixel by 1 R so frame count / per-frame intervals stay exact.
    """
    prev: bytes | None = None
    for i, frame in enumerate(frames):
        cur = np.asarray(frame.convert("RGBA")).tobytes()
        if prev is not None and cur == prev:
            arr = np.asarray(frame, dtype=np.uint8).copy()
            palette = list(frame.getpalette() or [])
            while len(palette) < 768:
                palette.append(0)

            opaque = np.argwhere(arr != TRANSPARENT_INDEX)
            if len(opaque) == 0:
                y = x = 0
                src_idx = int(arr[y, x])
            else:
                y, x = map(int, opaque[0])
                src_idx = int(arr[y, x])

            used = set(np.unique(arr).tolist())
            free = next(
                (idx for idx in range(256) if idx not in used and idx != TRANSPARENT_INDEX),
                254 if src_idx != 254 else 253,
            )
            r, g, b = palette[src_idx * 3 : src_idx * 3 + 3]
            palette[free * 3 : free * 3 + 3] = [(r + 1) % 256, g, b]
            arr[y, x] = free
            frames[i] = _p_image_from(arr, palette)
            cur = np.asarray(frames[i].convert("RGBA")).tobytes()
        prev = cur


def _p_image_from(arr: np.ndarray, palette: list[int]) -> Image.Image:
    """Build a clean P-mode image with a full RGB palette (avoids Pillow GIF writer bugs)."""
    while len(palette) < 768:
        palette.append(0)
    out = Image.fromarray(np.asarray(arr, dtype=np.uint8), mode="P")
    out.putpalette(palette[:768], rawmode="RGB")
    out.info["transparency"] = TRANSPARENT_INDEX
    return out


def _finalize_frames(frames: list[Image.Image]) -> list[Image.Image]:
    """Rebuild every frame so palette bytes/mode are consistent before save."""
    out: list[Image.Image] = []
    for frame in frames:
        palette = list(frame.getpalette() or [])
        out.append(_p_image_from(np.asarray(frame, dtype=np.uint8), palette))
    return out


def process_gif(src: Path, dst: Path) -> dict:
    with Image.open(src) as img:
        loop = img.info.get("loop", 0)

        frames: list[Image.Image] = []
        durations: list[int] = []
        keyed_total = 0

        for composited, duration, _disposal in _iter_gif_composited(img):
            keyed = key_blue_to_transparent(composited)
            keyed_total += int((np.asarray(keyed)[:, :, 3] < 128).sum())
            frames.append(_quantize_rgba_frame(keyed))
            durations.append(duration)

    if not frames:
        raise ValueError("empty gif")

    _uniquify_identical_frames(frames)
    frames = _finalize_frames(frames)

    # Full composited frames + transparency require disposal=2 (clear canvas).
    disposals = [2] * len(frames)

    # Match enhance_media.py: int when uniform, tuple when per-frame differs.
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
        transparency=TRANSPARENT_INDEX,
        optimize=False,
    )
    return {
        "frames": len(frames),
        "keyed_pixels": keyed_total,
        "durations": durations,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Key eye-blue pixels to transparent in GIFs.")
    parser.add_argument(
        "--src",
        type=Path,
        default=Path(__file__).resolve().parent / "EyeExpressionsVideo",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=Path(__file__).resolve().parent / "EyeExpressionsVideoTransparent",
    )
    parser.add_argument("--limit", type=int, default=0, help="Process at most N files (0=all).")
    args = parser.parse_args()

    gifs = sorted(args.src.rglob("*.gif"))
    if args.limit > 0:
        gifs = gifs[: args.limit]

    if not gifs:
        print(f"No GIFs under {args.src}", file=sys.stderr)
        return 1

    print(f"Processing {len(gifs)} GIFs: {args.src} -> {args.dst}")
    ok = 0
    for i, src in enumerate(gifs, 1):
        rel = src.relative_to(args.src)
        dst = args.dst / rel
        try:
            info = process_gif(src, dst)
            ok += 1
            durs = info["durations"]
            dur_desc = (
                f"{durs[0]}ms"
                if len(set(durs)) == 1
                else f"mixed({min(durs)}-{max(durs)}ms)"
            )
            print(
                f"[{i}/{len(gifs)}] {rel}  frames={info['frames']} "
                f"duration={dur_desc} keyed={info['keyed_pixels']}"
            )
        except Exception as exc:  # noqa: BLE001 - batch resilience
            print(f"[{i}/{len(gifs)}] FAIL {rel}: {exc}", file=sys.stderr)

    print(f"Done: {ok}/{len(gifs)} succeeded")
    return 0 if ok == len(gifs) else 2


if __name__ == "__main__":
    raise SystemExit(main())
