# ChequeDetector
### Document Detection and Adaptive Image Enhancement

**COMP 4102 – Computer Vision | Carleton University**
**Team:** Gloria Li & Gillian Gonzales

## Overview

ChequeDetector takes a photo of a cheque (PNG or JPG), finds it, straightens it, and then enhances it so it's easier to read.

Our project has two parts:

1. **Detection (Gloria):** find the cheque's four corners and correct the perspective.
2. **Enhancement (Gillian):** improve the readability of the straightened cheque.

We test only with fictional or voided cheques. This is not meant for real cheque validation or banking.

**Status:** in development.

## Pipeline

```text
Input photo -> Validate -> Blur -> Edge detection (per colour channel) -> Contours
  -> Find a 4-sided shape -> Check it looks like a cheque -> Order corners
  -> Perspective warp -> Enhancement -> Final image
```

**Detection.** We blur the colour image and run Canny edge detection on each of the blue, green and red channels, then combine the results. We don't convert to grayscale first. We then find contours, simplify them with `approxPolyDP`, and keep the first shape that passes our checks:

- exactly 4 corners
- convex
- covers at least 10% of the photo
- doesn't touch the photo's border, since that usually means a corner is cut off
- has a cheque-like width-to-height ratio (1.8 to 3.2)

Finally, we order the corners and use `getPerspectiveTransform` and `warpPerspective` to get a flat, colour crop.

**Enhancement.** We're still choosing a method. Options include adaptive thresholding, local contrast enhancement, illumination correction and Retinex-based enhancement. We'll compare each one against a simple baseline.

## Using it

```bash
pip install -r requirements.txt
python detection/make_test_image.py                    # makes a fake tilted cheque
python detection/run_detector.py synthetic_cheque.jpg  # saves overlay + crop to output/
python detection/evaluate.py test_images               # success rate, false positives, timing
```

`evaluate.py` expects two folders: `test_images/cheques/` for photos that should be detected, and `test_images/no_cheque/` for photos that should be rejected.

In code, the detector and enhancer pass NumPy arrays to each other:

```python
image = cv2.imread("sample_cheque.jpg")
cropped = detect_cheque(image)        # colour BGR crop, or None if no cheque found
if cropped is not None:
    enhanced = enhance_cheque(cropped)
```

## Project structure

```text
detection/
  detector.py          detect_cheque(image)
  run_detector.py      try it on a photo or folder
  evaluate.py          metrics for the report
  make_test_image.py   synthetic test photo
requirements.txt
```

## Edge cases

| Scenario | What should happen |
|---|---|
| Full cheque visible | Crop, straighten and enhance |
| A corner is outside the photo | Reject |
| No cheque in the photo | Reject |
| Another rectangle (book, phone) | Reject if it isn't cheque-shaped (may still pass) |
| Too dark or blurry | Try anyway, reject if no valid shape |
| Unsupported or corrupted file | Show an input error |

A geometric detector can't fully guarantee that a rectangle is a cheque. This is one of the limitations we'll discuss.

## Evaluation

- **Detection success rate:** % of cheque photos detected correctly
- **False positive rate:** % of non-cheque photos wrongly accepted
- **Readability:** before vs. after enhancement, using OCR accuracy or a readability rubric
- **Processing time:** average and median per image

## Roadmap

- [x] Detection pipeline (edges, contours, corners, perspective correction)
- [x] Geometric validation and invalid input handling
- [x] Evaluation script
- [ ] Collect real test photos and tune detection
- [ ] Enhancement baseline and advanced method
- [ ] Integrate detection and enhancement
- [ ] Evaluate, write the report and prepare the demo

## Team

- **Gloria:** detection, validation, detection testing and integration
- **Gillian:** enhancement research, implementation, readability evaluation and documentation
- **Both:** dataset, evaluation, final report and demo

**Tech:** Python, OpenCV, NumPy
