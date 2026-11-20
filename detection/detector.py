"""
ChequeDetector - detection module (Gloria).

What this file does, in one sentence:
    Given a photo that contains a cheque, find the cheque's 4 corners and
    return a flat, straightened, colour picture of just the cheque.

How other code uses it (the "contract" agreed with the enhancement module):
    cropped = detect_cheque(image)
        image   -> a colour photo loaded with cv2.imread (BGR, uint8 NumPy array)
        cropped -> the straightened colour cheque (BGR, uint8), or None if no
                   cheque was found

The five steps (each one is a function below):
    1. preprocess          - find edges (places where the colour changes sharply)
    2. find_cheque_corners - turn edges into shapes and simplify them to corners
    3. is_valid_quad       - check the shape looks like a cheque
    4. order_corners       - label the corners TL, TR, BR, BL
    5. detect_cheque       - stretch the cheque flat (perspective correction)

Note: we never convert the photo to grayscale. Edges are found separately in the
blue, green and red channels and then combined.
"""
import cv2          # OpenCV: the computer vision library
import numpy as np  # NumPy: images are stored as NumPy arrays of numbers

# ---- Settings you can tweak while testing ------------------------------------
# If real photos fail, these are the numbers to try changing first.

BLUR_KERNEL = (5, 5)        # How much to blur before finding edges (bigger = blurrier, must be odd)
CANNY_LOW, CANNY_HIGH = 50, 150   # Edge sensitivity. Lower numbers = more (and weaker) edges found
MIN_AREA_FRACTION = 0.10    # The cheque must cover at least 10% of the photo
APPROX_EPSILON = 0.02       # How much to simplify a wobbly outline (2% of its length)
BORDER_MARGIN = 5           # Corners closer than 5 pixels to the photo edge = probably cut off
ASPECT_MIN, ASPECT_MAX = 1.8, 3.2   # Width / height range. Real cheques are about 2.3 to 2.6


def preprocess(image):
    """Step 1: find edges. Returns a black image with white lines where edges are."""
    # Blur slightly so tiny details (paper texture, noise) don't count as edges.
    blurred = cv2.GaussianBlur(image, BLUR_KERNEL, 0)

    # Start with an all-black image the same size as the photo (height x width).
    edges = np.zeros(image.shape[:2], np.uint8)

    # A colour image has 3 layers: Blue, Green, Red. Find edges in each layer,
    # and keep a pixel as an edge if ANY layer found one there (bitwise OR).
    # This catches edges that differ in colour even when brightness is similar.
    for channel in cv2.split(blurred):
        channel_edges = cv2.Canny(channel, CANNY_LOW, CANNY_HIGH)
        edges = cv2.bitwise_or(edges, channel_edges)

    # Thicken the edge lines a little so small gaps in the outline get closed.
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    return edges


def order_corners(pts):
    """Step 4: return the 4 corners in the order top-left, top-right, bottom-right, bottom-left."""
    pts = pts.reshape(4, 2).astype("float32")  # 4 rows of (x, y)

    # Trick: x + y is smallest at the top-left and largest at the bottom-right.
    s = pts.sum(axis=1)
    # And y - x is smallest at the top-right and largest at the bottom-left.
    d = np.diff(pts, axis=1).ravel()

    return np.array([
        pts[np.argmin(s)],   # top-left
        pts[np.argmin(d)],   # top-right
        pts[np.argmax(s)],   # bottom-right
        pts[np.argmax(d)],   # bottom-left
    ], dtype="float32")


def output_size(corners):
    """How big the straightened cheque should be, based on the distances between corners."""
    tl, tr, br, bl = corners
    # Width = the longer of the top edge and the bottom edge.
    width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    # Height = the longer of the left edge and the right edge.
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    return width, height


def is_valid_quad(approx, image_shape):
    """Step 3: does this shape look like a cheque? Returns (True/False, reason)."""
    h, w = image_shape[:2]  # photo height and width in pixels

    # Check 1: exactly 4 corners.
    if len(approx) != 4:
        return False, "not 4 corners"

    # Check 2: convex, meaning no corner is dented inwards.
    if not cv2.isContourConvex(approx):
        return False, "not convex"

    # Check 3: big enough. Small rectangles are probably something else.
    if cv2.contourArea(approx) < MIN_AREA_FRACTION * w * h:
        return False, "too small"

    # Check 4: not touching the photo edge. If it does, a corner is probably cut off.
    pts = approx.reshape(4, 2)
    if (pts[:, 0].min() < BORDER_MARGIN or pts[:, 1].min() < BORDER_MARGIN or
            pts[:, 0].max() > w - BORDER_MARGIN or pts[:, 1].max() > h - BORDER_MARGIN):
        return False, "touches image border"

    # Check 5: cheque-shaped, i.e. roughly 2 to 3 times wider than tall.
    cw, ch = output_size(order_corners(approx))
    aspect = max(cw, ch) / max(1, min(cw, ch))
    if not ASPECT_MIN <= aspect <= ASPECT_MAX:
        return False, f"aspect ratio {aspect:.2f} not cheque-like"

    return True, "ok"


def find_cheque_corners(image, debug=False):
    """Step 2: find the 4 ordered corners of the cheque, or None if there isn't one."""
    edges = preprocess(image)

    # Trace the white edge lines into closed outlines ("contours").
    # RETR_EXTERNAL = only keep outer outlines, not shapes inside other shapes.
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Try the 10 biggest outlines first, since the cheque is usually the biggest thing.
    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
        # Simplify the outline to a few main corner points.
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, APPROX_EPSILON * perimeter, True)

        ok, reason = is_valid_quad(approx, image.shape)
        if debug:
            print(f"  candidate area={cv2.contourArea(c):.0f} corners={len(approx)} -> {reason}")
        if ok:
            return order_corners(approx)

    return None  # nothing passed the checks


def detect_cheque(image, debug=False):
    """Main function: colour photo in, straightened colour cheque out (or None)."""
    # Reject anything that isn't a colour image (e.g. a file that failed to load).
    if image is None or not isinstance(image, np.ndarray) or image.ndim != 3:
        return None

    corners = find_cheque_corners(image, debug=debug)
    if corners is None:
        return None

    # Step 5: where each corner should end up in the flat output image.
    width, height = output_size(corners)
    target = np.array([[0, 0],                    # top-left
                       [width - 1, 0],            # top-right
                       [width - 1, height - 1],   # bottom-right
                       [0, height - 1]],          # bottom-left
                      dtype="float32")

    # Work out the "stretch" that maps the photo corners onto the flat rectangle,
    # then apply it to the whole photo. The result is the cheque laid flat.
    M = cv2.getPerspectiveTransform(corners, target)
    return cv2.warpPerspective(image, M, (width, height))
