import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def to_text_gray(img_bgr: np.ndarray) -> np.ndarray:
    b, g, r = cv2.split(img_bgr) # Splits the 3 layers
    gray = (0.65 * r + 0.30 * g + 0.05 * b).astype(np.uint8)
    return gray

def correct_illumination(gray: np.ndarray, scale: float = 0.08) -> np.ndarray:
    h, w = gray.shape
    k = int(max(h, w) * scale) | 1  # odd kernel
    # Closing removes dark text so the blur estimates only the paper/background
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k // 2 | 1, k // 2 | 1))
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    background = cv2.GaussianBlur(background, (k, k), 0)
    norm = cv2.divide(gray, background, scale=255)
    return norm

def sauvola(gray: np.ndarray, window: int = 31, k: float = 0.2, r: float = 128.0) -> np.ndarray:
    window = window | 1
    g = gray.astype(np.float64)
    mean = cv2.boxFilter(g, -1, (window, window), normalize=True, borderType=cv2.BORDER_REPLICATE)
    sqmean = cv2.boxFilter(g * g, -1, (window, window), normalize=True, borderType=cv2.BORDER_REPLICATE)
    std = np.sqrt(np.maximum(sqmean - mean * mean, 0))
    thresh = mean * (1 + k * (std / r - 1))
    return np.where(g > thresh, 255, 0).astype(np.uint8)

def niblack(gray: np.ndarray, window: int = 31, k: float = -0.2) -> np.ndarray:
    window = window | 1
    g = gray.astype(np.float64)
    mean = cv2.boxFilter(g, -1, (window, window), normalize=True, borderType=cv2.BORDER_REPLICATE)
    sqmean = cv2.boxFilter(g * g, -1, (window, window), normalize=True, borderType=cv2.BORDER_REPLICATE)
    std = np.sqrt(np.maximum(sqmean - mean * mean, 0))
    return np.where(g > mean + k * std, 255, 0).astype(np.uint8)


def local_contrast_enhancement(gray, strength=1.5, sigma=15):
    gray = gray.astype(np.float32)
    background = cv2.GaussianBlur(gray, (0, 0), sigma)
    detail = gray - background
    enhanced = gray + strength * detail
    enhanced = np.clip(enhanced, 0, 255)

    return enhanced.astype(np.uint8)

def remove_speckles(binary: np.ndarray, min_area: int) -> np.ndarray:
    inv = cv2.bitwise_not(binary) #inverted
    n, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8) #Connection limit (componets)
    keep = np.zeros_like(inv) # new image
    for i in range(1, n): # loop through componets
        if stats[i, cv2.CC_STAT_AREA] >= min_area: # check if componet is great enough to not be a outlier
            keep[labels == i] = 255 # set white
    return cv2.bitwise_not(keep)

def binarize_cheque(
    img_bgr: np.ndarray,
    method: str = "sauvola",
    window: int = 31,
    k: float = 0.2,
    target_width: int = 1800,
    debug_dir: Path | None = None,
) -> np.ndarray:
    # 1. Upscale small images so strokes are several pixels wide
    h, w = img_bgr.shape[:2]
    if w < target_width:
        s = target_width / w
        img_bgr = cv2.resize(img_bgr, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    else:
        s = 1.0

    # 2. Grayscale that suppresses the coloured background
    gray = to_text_gray(img_bgr)

    # 3. Flatten lighting
    flat = correct_illumination(gray)

    # # 4. Local contrast
    enhanced = local_contrast_enhancement(flat)

    # 5. Binarise
    win = max(15, int(window * max(s, 1.0) / 3)) | 1  # scale window with the upscale
    if method == "niblack":
        binary = niblack(enhanced, win, -abs(k))
    else:  # sauvola
        binary = sauvola(enhanced, win, k)

    debug_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(debug_dir / "1_gray.png"), gray)
    cv2.imwrite(str(debug_dir / "2_flat.png"), flat)
    cv2.imwrite(str(debug_dir / "3_enhanced.png"), enhanced)
    cv2.imwrite(str(debug_dir / "4_binary.png"), binary)

    return binary


def main() -> int:
    p = argparse.ArgumentParser(description="Binarize a cheque image.")
    p.add_argument("input", help="Path to cheque image (jpg/png/...)")
    p.add_argument("--method", choices=["sauvola", "niblack"], default="sauvola")
    p.add_argument("--window", type=int, default=25, help="Local window size (odd number)")
    p.add_argument("--k", type=float, default=0.34, help="Sauvola/Niblack sensitivity")
    p.add_argument("--width", type=int, default=1800, help="Upscale images narrower than this")
    args = p.parse_args()

    src = Path(args.input)
    out = Path(args.output) if args.output else src.with_name(src.stem + "_binary.png")

    img = load_image(src)
    result = binarize_cheque(
        img, args.method, args.window, args.k, args.width, Path("out_debug")
    )
    cv2.imwrite(str(out), result)
    print(f"Saved: {out}  ({result.shape[1]}x{result.shape[0]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
