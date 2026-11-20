"""
Try the detector on one photo or a whole folder.

    python detection/run_detector.py photo.jpg
    python detection/run_detector.py test_images/

Saves <name>_overlay.jpg (green outline on the photo) and <name>_crop.jpg in output/.
"""
import sys
import time
from pathlib import Path

import cv2

from detector import detect_cheque, find_cheque_corners

SUPPORTED = {".png", ".jpg", ".jpeg"}


def run_one(path, out_dir):
    if path.suffix.lower() not in SUPPORTED:
        print(f"{path.name}: FAILED (unsupported format)")
        return
    image = cv2.imread(str(path))
    if image is None:
        print(f"{path.name}: FAILED (could not read image)")
        return

    start = time.perf_counter()
    crop = detect_cheque(image, debug=True)
    ms = (time.perf_counter() - start) * 1000

    if crop is None:
        print(f"{path.name}: FAILED (no cheque found) [{ms:.0f} ms]")
        return

    corners = find_cheque_corners(image)
    overlay = image.copy()
    cv2.polylines(overlay, [corners.astype(int)], True, (0, 255, 0), 4)
    cv2.imwrite(str(out_dir / f"{path.stem}_overlay.jpg"), overlay)
    cv2.imwrite(str(out_dir / f"{path.stem}_crop.jpg"), crop)
    print(f"{path.name}: SUCCESS crop={crop.shape[1]}x{crop.shape[0]} [{ms:.0f} ms]")


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    target = Path(sys.argv[1])
    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)
    paths = sorted(target.iterdir()) if target.is_dir() else [target]
    for p in paths:
        run_one(p, out_dir)


if __name__ == "__main__":
    main()
