"""
Measure how well the detector works, for the report (README section 6.2).

Put your test photos into two folders:

    test_images/cheques/      photos that DO contain a full cheque (should succeed)
    test_images/no_cheque/    photos with no cheque, a cut-off cheque, or another
                              rectangle like a book or phone (should fail)

Then run:

    python detection/evaluate.py test_images

It prints:
    - Detection success rate: % of cheque photos where a cheque was found
    - False positive rate:    % of "no cheque" photos wrongly accepted
    - Average and median processing time per photo
and saves every photo's result to results.csv (open it in Excel or Sheets).
"""
import csv
import statistics
import sys
import time
from pathlib import Path

import cv2

from detector import detect_cheque

SUPPORTED = {".png", ".jpg", ".jpeg"}


def run_folder(folder, label):
    """Run the detector on every photo in one folder. Returns a list of result rows."""
    rows = []
    if not folder.is_dir():
        print(f"(skipping {folder}: folder not found)")
        return rows
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() not in SUPPORTED:
            continue
        image = cv2.imread(str(path))
        start = time.perf_counter()
        crop = detect_cheque(image)
        ms = (time.perf_counter() - start) * 1000
        rows.append({"file": path.name, "folder": label,
                     "detected": crop is not None, "time_ms": round(ms, 1)})
    return rows


def percent(part, whole):
    return f"{100 * part / whole:.1f}% ({part}/{whole})" if whole else "n/a (no photos)"


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    root = Path(sys.argv[1])

    positives = run_folder(root / "cheques", "cheques")
    negatives = run_folder(root / "no_cheque", "no_cheque")
    rows = positives + negatives
    if not rows:
        print("No photos found. See the instructions at the top of this file.")
        sys.exit(1)

    found = sum(r["detected"] for r in positives)
    wrongly_accepted = sum(r["detected"] for r in negatives)
    times = [r["time_ms"] for r in rows]

    print(f"Detection success rate: {percent(found, len(positives))}")
    print(f"False positive rate:    {percent(wrongly_accepted, len(negatives))}")
    print(f"Average time per photo: {statistics.mean(times):.1f} ms")
    print(f"Median time per photo:  {statistics.median(times):.1f} ms")

    # List the photos that went wrong, so you know which ones to look at.
    for r in positives:
        if not r["detected"]:
            print(f"  missed cheque:   {r['file']}")
    for r in negatives:
        if r["detected"]:
            print(f"  false positive:  {r['file']}")

    with open("results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "folder", "detected", "time_ms"])
        writer.writeheader()
        writer.writerows(rows)
    print("Saved results.csv")


if __name__ == "__main__":
    main()
