from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def key_to_path(fiw_root: str | Path, key: str) -> Path:
    return Path(fiw_root) / str(key)


def read_img(path: str | Path, fallback_size: int = 256) -> Image.Image:
    try:
        return Image.open(path).convert("RGB")
    except Exception as exc:
        print("Could not open:", path, "|", exc)
        return Image.new("RGB", (fallback_size, fallback_size), color=(230, 230, 230))


def square_center_crop(img: Image.Image) -> Image.Image:
    width, height = img.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return img.crop((left, top, left + side, top + side))


def prep_img(path: str | Path, size: int = 256) -> Image.Image:
    img = read_img(path)
    img = square_center_crop(img)
    return ImageOps.fit(img, (size, size), method=Image.Resampling.LANCZOS)


def image_quality_stats(path: str | Path) -> dict:
    img = read_img(path)
    arr = np.asarray(img).astype(np.float32) / 255.0
    height, width = arr.shape[:2]
    gray = arr.mean(axis=2)

    lap = (
        -4 * gray
        + np.roll(gray, 1, axis=0)
        + np.roll(gray, -1, axis=0)
        + np.roll(gray, 1, axis=1)
        + np.roll(gray, -1, axis=1)
    )

    contrast = float(np.std(gray))
    sharpness = float(np.var(lap))

    rg = arr[:, :, 0] - arr[:, :, 1]
    yb = 0.5 * (arr[:, :, 0] + arr[:, :, 1]) - arr[:, :, 2]
    colorfulness = float(
        np.sqrt(np.var(rg) + np.var(yb))
        + 0.3 * np.sqrt(np.mean(rg**2) + np.mean(yb**2))
    )
    channel_std = float(np.mean(np.std(arr, axis=2)))

    return {
        "min_side": min(height, width),
        "sharpness": sharpness,
        "contrast": contrast,
        "colorfulness": colorfulness,
        "gray_like": channel_std < 0.03 or colorfulness < 0.03,
    }
