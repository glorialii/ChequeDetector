
# ChequeDetector
### Document Detection and Adaptive Image Enhancement

**COMP 4102 – Computer Vision | Carleton University**

**Team:** Gloria Li & Gillian Gonzales

## 1. Project Overview

ChequeDetector is a computer vision application designed to automatically detect, extract, and enhance cheque images captured using a smartphone or camera.

The system takes a single PNG or JPG image containing a cheque, identifies its boundaries, corrects perspective distortion, and applies image enhancement techniques to improve readability.

The project focuses on two main components:

1. **Cheque Detection and Extraction:** Locate the cheque within an image, identify its four corners, and perform perspective correction.
2. **Adaptive Image Enhancement:** Improve the readability of the extracted cheque using image processing techniques.

The primary goal is to produce a correctly extracted, readable cheque image while evaluating detection accuracy, readability, and processing speed.

**Current status:** In-development / Implementation

---

## 2. Project Objectives

- Detect cheque boundaries in photographs.
- Identify the four corners of a cheque.
- Crop and correct perspective distortion.
- Handle invalid inputs and detection failures. (OPT? EDGE CASES? ADD LATER)
- Improve cheque readability using image enhancement.
- Compare enhancement methods against a baseline.
- Evaluate detection success, readability, and runtime.

### Project Scope

- Input: One photograph containing a cheque.
- Supported formats: `.png`, `.jpg`, `.jpeg`.
- All four cheque corners should be visible.
- Output: A perspective-corrected and enhanced cheque image.
- Testing: Photographs of fictional or voided cheques with no sensitive banking information.

The application is not intended for cheque validation, bank deposits, or financial transaction processing.

---

## 3. System Pipeline

```text
Input Image (PNG/JPEG)
        |
        v
Image Validation
        |
        v
Grayscale Conversion
        |
        v
Gaussian Blur
        |
        v
Canny Edge Detection
        |
        v
Contour Detection
        |
        v
Quadrilateral Approximation
        |
        v
Geometric Validation
        |
        v
Four-Corner Detection
        |
        v
Perspective Transformation
        |
        v
Return Corrected Cheque Image
        |
        v
Image Enhancement (GILLIAN TO ADD)
        |
        v
Final Enhanced Image
```

### 3.1 Image Preprocessing

The input image is converted to grayscale to simplify processing.

Gaussian blur is applied to reduce image noise and suppress small details that could interfere with edge detection.

**Planned methods:**
- Grayscale conversion
- Gaussian filtering

### 3.2 Edge Detection

Canny edge detection is used to identify intensity changes that may correspond to cheque boundaries.

**Planned method:**
- Canny Edge Detection

### 3.3 Contour Detection

Contours are extracted from the edge map to identify possible document boundaries.

Candidate contours are evaluated based on their size and geometric properties.

**Planned methods:**
- `cv2.findContours()`
- Contour area filtering
- Contour approximation using `cv2.approxPolyDP()`

### 3.4 Quadrilateral Detection

The detector searches for a suitable four-sided polygon representing the cheque boundary.

Candidate validation includes:

- Four detected vertices
- Convex polygon geometry
- Minimum area requirement
- Plausible cheque proportions
- Sufficient distance from image boundaries

### 3.5 Perspective Correction

After identifying the four corners, the system applies a perspective transformation to produce a rectangular cheque image.

**Planned methods:**
- Corner ordering
- `cv2.getPerspectiveTransform()`
- `cv2.warpPerspective()`

The corrected image is preserved in colour before being passed to the enhancement module.

### 3.6 Image Enhancement

The extracted cheque is passed to the enhancement module to improve readability.

Potential techniques under consideration include:

- Adaptive thresholding
- Local contrast enhancement
- Illumination correction
- Retinex-based enhancement
- Single-image deblurring (optional)

The final enhancement method will be selected based on feasibility, research, and experimental results.

---

## 4. Edge Cases and Error Handling (TO EDIT)

The system should handle unexpected inputs and detection failures without producing misleading results.

| Scenario | Expected Behaviour | Status |
|---|---|---|
| All four corners visible and cheque detected | Crop, straighten, and enhance the cheque | SUCCESS |
| One or more corners outside the image frame | Reject detection and request another photograph | FAILED |
| Image does not contain a cheque | Reject if no valid cheque candidate is identified | FAILED |
| Image contains another rectangular object | Apply additional validation; reject or flag uncertain candidates | UNCERTAIN |
| Image is too dark or blurry | Attempt detection; reject or flag if confidence is insufficient | UNCERTAIN |
| Unsupported image format | Display an input validation error | FAILED |
| Corrupted or unreadable image | Display an input validation error | FAILED |
| Multiple rectangular objects detected | Evaluate candidates and reject ambiguous results | UNCERTAIN |
| Cheque boundary blends into background | Attempt alternative preprocessing or report failure | UNCERTAIN |

**Note:** These are intended behaviours. A basic geometric detector cannot guarantee that a detected rectangle is a cheque or that a partially visible cheque will always be rejected.

### 4.1 Detection Validation

Before accepting a detected contour, the system will evaluate:

1. **Corner count:** The candidate should contain four vertices.
2. **Convexity:** The candidate should form a convex quadrilateral.
3. **Area:** The candidate should occupy a reasonable portion of the image.
4. **Aspect ratio:** The corrected candidate should have plausible cheque-like dimensions.
5. **Border clearance:** Detected corners should not be too close to the image boundaries.

Additional cheque-specific validation may be investigated if time permits.

### 4.2 Detection Status

The system will distinguish between three outcomes:

**SUCCESS**
- A suitable document boundary is detected.
- Geometric validation passes.
- Perspective correction is performed.
- The corrected image is returned.

**UNCERTAIN**
- A possible document is detected.
- Validation is insufficient or ambiguous.
- The image is not automatically passed to enhancement.

**FAILED**
- The input image is invalid, or no acceptable document boundary is found.
- No corrected image is returned.
- The user is prompted to provide another image.

### 4.3 Failure Handling

The detector's initial integration interface will return:

- A BGR `uint8` NumPy array when detection succeeds.
- `None` when detection fails or is rejected.

More detailed status messages may be added for debugging and user feedback.

---

## 5. Project Structure

Planned repository structure:

```text
Project (TBA)
```

### Module Interface

The detector and enhancer will exchange NumPy arrays rather than image file paths.

```python
# Planned integration interface

image = cv2.imread("sample_cheque.jpg")

cropped = detect_cheque(image)

if cropped is not None:
    enhanced = enhance_cheque(cropped)
```

The detector must return a colour BGR image. The enhancer may return a grayscale or binary image depending on the selected method.

---

## 6. Testing and Evaluation

The application will be evaluated using photographs of cheques under different conditions.

### 6.1 Test Conditions

- Normal lighting
... TBA

### 6.2 Evaluation Metrics

**Detection Success Rate**

Percentage of cheque photographs where the document is correctly detected and perspective-corrected.

**False Positive Rate**

Percentage of negative test images incorrectly accepted as valid cheque detections.

**Readability**

Comparison of the extracted image before and after enhancement using OCR character accuracy against known fictional text or a documented human readability rubric.

**Processing Time**

Average and median execution time per image.

**Optional Geometric Accuracy**

Comparison of detected document corners against manually annotated ground-truth corners.

### 6.3 Expected Outputs

The system will save or display:

1. Original photograph
2. Detection overlay showing the identified cheque boundary
3. Perspective-corrected cheque crop
4. Enhanced cheque image
5. Evaluation results

---

## 7. Team Responsibilities

### Gloria – Detection and Extraction

- Image input handling
- Image preprocessing
- Edge detection
- Contour detection
- Four-corner identification
- Geometric validation
- Perspective transformation
- Detection testing and failure analysis
- Integration with enhancement module

### Gillian – Image Enhancement and Documentation

- Research enhancement techniques
- Implement image enhancement algorithms
- Compare enhancement methods
- Evaluate readability improvements
- Document experimental results and limitations

### Shared Responsibilities

- Dataset preparation
- Pipeline integration
- Evaluation methodology
- Testing
- Final report
- Project demonstration

---

## 8. Development Roadmap

### Phase 1: Detection Baseline
- [ ] Implement image preprocessing
- [ ] Implement Canny edge detection
- [ ] Implement contour detection
- [ ] Identify document corners
- [ ] Implement perspective correction

### Phase 2: Validation and Error Handling
- [ ] Validate quadrilateral geometry
- [ ] Handle missing or incomplete document boundaries
- [ ] Handle unrelated input images
- [ ] Handle invalid image files
- [ ] Test detection failures

### Phase 3: Enhancement
- [ ] Implement a simple enhancement baseline
- [ ] Research and select an advanced enhancement method
- [ ] Implement selected enhancement technique
- [ ] Compare enhancement outputs

### Phase 4: Integration and Evaluation
- [ ] Integrate detection and enhancement modules
- [ ] Build a reproducible test dataset
- [ ] Evaluate detection success and false positives
- [ ] Evaluate readability improvements
- [ ] Measure processing time
- [ ] Prepare final report and demonstration

---

## 9. Technologies

**Language:** Python

**Planned libraries:**
- OpenCV
- NumPy

Additional libraries may be introduced depending on the selected enhancement and evaluation methods.

---

## 10. Current Development Status

The project is currently in the development phase.

**Target:** A working, evaluated end-to-end prototype for the COMP 4102 final project.
