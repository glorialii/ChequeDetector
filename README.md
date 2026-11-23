<div align="center">

# ChequeDetector

**Document Detection and Adaptive Image Enhancement**

COMP 4102 – Computer Vision · Carleton University

Gloria Li & Gillian Gonzales

Using OpenCV and Python3

</div>

---

**Contents:** [Overview](#overview) · [How to run](#how-to-run) · [Pipeline](#pipeline) · [Using the detector in code](#using-the-detector-in-code) · [Project structure](#project-structure) · [Evaluation](#evaluation) · [Edge cases](#edge-cases) · [Roadmap](#roadmap) · [Team](#team)

---

## Overview

ChequeDetector takes a photo of a cheque (PNG or JPG), finds it, straightens it, and then enhances it so it's easier to read.

Our project has two parts:

1. **Detection (Gloria):** find the cheque's four corners and correct the perspective.
2. **Enhancement (Gillian):** improve the readability of the straightened cheque.

> **Note:** we test only with fictional or voided cheques. This project is not meant for real cheque validation or banking.

> **Status:** detection is working; enhancement is in progress.

---

## How to run

**1. Install the requirements** (first time only)

```bash
pip3 install opencv-python numpy
```

**2. Go to the project folder**

```bash
cd ~/repositoryof/ChequeDetector
```

**3. Run the detector on a photo**

```bash
python3 main.py ~/repositoryofimg/imgname.jpg
```

Replace `~/repositoryof/` with wherever you cloned the repo, and `~/repositoryofimg/imgname.jpg` with the path to your photo. On macOS, you can drag the photo into the terminal to paste its path.

**More options**

| Command | What it does |
| --- | --- |
| `python3 main.py ~/repositoryofimg/` | Runs every photo in a folder |
| `python3 evaluate.py test_images` | Prints detection metrics for the report (see [Evaluation](#evaluation)) |

**Output:** results are saved to `output/` inside the project folder:

| File | Contents |
| --- | --- |
| `<imgname>_overlay.jpg` | The original photo with the detected cheque outlined in green |
| `<imgname>_crop.jpg` | The straightened, full-resolution colour cheque |

The terminal also prints `SUCCESS` or `FAILED` for each photo, plus how long it took.

---

## Pipeline

```text
Input photo -> Shrink to 1000 px -> Blur -> Edges (per colour channel) -> Contours
  -> Convex hull + 4-corner check -> Order corners -> Scale back to full size
  -> Perspective warp -> Enhancement -> Final image
```

**Detection** (`detector.py`):

1. **Shrink** (`locate_cheque`): we find the corners on a copy of the photo whose longest side is 1000 px (`DETECT_SIZE`). This makes the settings work the same for any camera, including 4K photos.
2. **Edges** (`preprocess`): we apply a 5×5 Gaussian blur, then run Canny edge detection on the blue, green and red channels separately and combine them. We don't convert to grayscale, so edges that differ only in colour are kept.
3. **Shapes** (`find_cheque_corners`): `findContours` with `RETR_LIST` keeps inner outlines too. Each outline gets a convex hull and is simplified with `approxPolyDP`. This lets us find the cheque even when glare or a table seam touches its edge.
4. **Validation** (`is_valid_quad`): we keep a shape only if it has 4 corners, is convex, covers at least 10% of the photo, doesn't touch the photo border, and has a cheque-like aspect ratio (1.8 to 3.2).
5. **Straighten** (`detect_cheque`): we order the corners, scale them back to the full-size photo, and use `getPerspectiveTransform` and `warpPerspective` to get a sharp, flat colour crop.

**Enhancement** (Gillian): this is still being chosen. Options include adaptive thresholding, local contrast enhancement, illumination correction and Retinex-based enhancement, each compared against a simple baseline.

---

## Using the detector in code

```python
import cv2
from detector import detect_cheque

image = cv2.imread("photo.jpg")
cropped = detect_cheque(image)   # BGR colour crop, or None if no cheque found
if cropped is not None:
    enhanced = enhance_cheque(cropped)
```

---

## Project structure

```text
detector.py    detection module: detect_cheque(image)
main.py        runs the pipeline on a photo or folder (entry point)
evaluate.py    detection metrics for the report -> results.csv
```

---

## Evaluation

Put the test photos in two folders, then run `python3 evaluate.py test_images`:

```text
test_images/
  cheques/      photos with a full cheque (should be detected)
  no_cheque/    no cheque, a cut-off cheque, or another rectangle (should be rejected)
```

It prints the **detection success rate**, the **false positive rate** and the **average and median time** per photo, and saves every result to `results.csv`. We assess readability before vs. after enhancement using OCR accuracy or a readability rubric.

---

## Edge cases

| Scenario | What should happen |
|---|---|
| Full cheque visible | Crop, straighten and enhance |
| A corner is outside the photo | Reject |
| No cheque in the photo | Reject |
| Another rectangle (book, phone) | Reject if it isn't cheque-shaped (a cheque-shaped object may still pass) |
| Glare or a seam touching the cheque edge | Still detected (convex hull + inner outlines) |
| Too dark or blurry | Try anyway; reject if no valid shape is found |
| Unsupported or corrupted file | Report an input error |
| Cheque photographed sideways | Detected, but the crop comes out tall and may need rotating |

A geometric detector can't fully guarantee that a rectangle is a cheque. This is one of the limitations we'll discuss.

---

## Roadmap

- [x] Detection pipeline (edges, contours, corners, perspective correction)
- [x] Geometric validation and invalid input handling
- [x] Large-photo and glare handling
- [x] Main script and evaluation script
- [ ] Collect real test photos and evaluate detection
- [ ] Enhancement baseline and advanced method
- [ ] Integrate enhancement into `main.py`
- [ ] Final report and demo

---

## Team

- **Gloria:** detection, validation, detection testing and integration
- **Gillian:** enhancement research, implementation, readability evaluation and documentation
- **Both:** dataset, evaluation, final report and demo

**Built with:** Python · OpenCV · NumPy
