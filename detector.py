import cv2 # OpenCV
import numpy as np # NumPy: imgs are stored as NumPy arrays of numbers

"""

How other code uses it (the "contract" agreed with the enhancement module):
    cropped = detect_cheque(image)
        image   -> a colour photo loaded with cv2.imread (BGR, uint8 NumPy array)
        cropped -> the straightened colour cheque (BGR, uint8), or None if no
                   cheque was found

"""

# SETTINGS: can tweak while testing... if real photos fail, these are the #s to alter

# how much to blur before finding edges (bigger = blurrier; must be odd)
BLUR_KERNEL = (5, 5)

# edge sensitivity (lower numbers = more and weaker edges found)
CANNY_LOW, CANNY_HIGH = 50, 150

# the cheque must cover at least 10% of the photo
MIN_AREA_FRACTION = 0.10

# how much to simplify wobbly outline (2% of its length)
APPROX_EPSILON = 0.02

# corners closer than 5 pixels to the photo edge = probably cut off
BORDER_MARGIN = 5

# width/height range; real cheques are around: 2.3 to 2.6
ASPECT_MIN, ASPECT_MAX = 1.8, 3.2

# FUNCTIONS

"""
The five steps (each one is a function below):
    1. preprocess          - find edges (places where the colour changes sharply)
    2. find_cheque_corners - turn edges into shapes and simplify them to corners
    3. is_valid_quad       - check the shape looks like a cheque
    4. order_corners       - label the corners TL, TR, BR, BL
    5. detect_cheque       - stretch the cheque flat (perspective correction)
"""

def preprocess(image):
    # 1. find edges; returns a black image with white lines where edges are
    # slightly blur so tiny details don't count as edges
    blurred = cv2.GaussianBlur(image, BLUR_KERNEL, 0)

    # start with all-black image, SAME SIZE as photo
    edges = np.zeros(image.shape[:2], np.uint8)

    # find edges in each color layer --> Blue, Green, Red
    # keep a pixel as an edge IF ANY layer found one there (bitwise OR)
    # this catches edges that differ in color even when brightness is similar
    for channel in cv2.split(blurred):
        channel_edges = cv2.Canny(channel, CANNY_LOW, CANNY_HIGH)
        edges = cv2.bitwise_or(edges, channel_edges)

    # thicken the edge lines a little so small gaps in outline get closed
    edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
    return edges

def order_corners(pts):
    # return the 4 corners in the order top-left, top-right, bottom-right, bottom-left
    pts = pts.reshape(4, 2).astype("float32") # 4 rows of (x,y)

    # *** NOTE x + y is smallest at the top-left and largest at the bottom-right
    s = pts.sum(axis=1)
    # y - x is smallest at top-right and largest at bottom-left
    d = np.diff(pts, axis=1).ravel()

    return np.array([
        pts[np.argmin(s)], # top-left
        pts[np.argmin(d)], # top-right
        pts[np.argmax(s)], # bottom-right
        pts[np.argmax(d)], # bottom-left
    ], dtype="float32")


def output_size(corners):
    # how big the straightened cheque should be, based on distances between corners
    tl, tr, br, bl = corners
    # width = longer of the top edge and the bottom edge
    width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr-tl)))
    # height = the longer of the left edge and the right edge
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    return width, height

def is_valid_quad(approx, image_shape):

    # add more cases based on need

    # does this shape look like a cheque? returns (T/F, reason)
    h, w = image_shape[:2] # photo height and width in pixels

    # check 1: has 4 corners
    if len(approx) != 4:
        return False, "not 4 corner"

    # check 2: convex, meaning no corner is dented inwards
    if not cv2.isContourConvex(approx):
        return False, "not convex"

    # check 3: big enough? smaller rectangles are probably something else
    if cv2.contourArea(approx) < MIN_AREA_FRACTION * w * h:
        return False, "too small"

    # check 4: not touching photo edge; if does, a corner is probably cut off
    pts = approx.reshape(4,2)
    if (pts[:, 0].min() < BORDER_MARGIN or 
        pts[:, 1].min() < BORDER_MARGIN or 
        pts[:, 0].max() > w - BORDER_MARGIN or
        pts[:, 1].max() > h - BORDER_MARGIN):
        return False, "touches image border"

    # check 5: cheque-shaped, i.e. roughly 2-3x wider than tall
    cw, ch = output_size(order_corners(approx))
    aspect = max(cw, ch) / max(1, min(cw, ch))
    if not ASPECT_MIN <= aspect <= ASPECT_MAX:
        return False, f"aspect ratio {aspect: .2f} not cheque-like"
    return True, "everything good, passed is_valid_quad"

def find_cheque_corners() # IMPLEMENT...

def detect_cheque() # IMPLEMENT...