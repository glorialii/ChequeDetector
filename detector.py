# detector.py
import cv2 # OpenCV
import numpy as np # NumPy: imgs are stored as NumPy arrays of numbers

"""

How other code uses it:
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

# detect size -> shrinks photo to find corners; esp for 4k photos
DETECT_SIZE = 1000

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

def find_cheque_corners(image, debug=False):
    # find the 4 ordered corners of the cheque, or None if there isn't one
    edges = preprocess(image)

    # trace white edge lines into closed outlines (contours)
    # RETR_LIST = looks at outlines inside other shapes; so inner edges also count when outer edge has glare
    contours, _ = cv2.findContours(edges,
                    cv2.RETR_LIST,
                    cv2.CHAIN_APPROX_SIMPLE)

    # try the 10 biggest outlines first; the cheque is usually the biggest thing
    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
        # simplify wobbly outline down to its main corners
        # convex hull = "rubber band" aroudn outline - fills in small dents

        hull = cv2.convexHull(c)
        
        perimeter = cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, APPROX_EPSILON * perimeter, True)

        ok, reason = is_valid_quad(approx, image.shape)
        if debug:
            print(f"candidate corners={len(approx)} -> {reason}")

        if ok:
            return order_corners(approx)

    return None # nothing passed the checks
        
def locate_cheque(image, debug=False):
    # 4 corners of the cheque in full size photo coordinates, or None
    # shrink big photos so blur/edge settings work on the same on any camera,
    # find corners on small copy then scale them back to full size

    scale = min(1.0, DETECT_SIZE / max(image.shape[:2]))
    small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    corners = find_cheque_corners(small, debug=debug)
    if corners is None:
        return None
    return (corners / scale).astype("float32") # back to full-size photo coordinates
    
def detect_cheque(image, debug=False):
    # MAIN FUNCTION: color photo in, straightened colour cheque out (or None)
    # reject anything that isn't a colour image (e.g., file failed to load)

    if image is None or not isinstance(image, np.ndarray) or image.ndim != 3:
        return None

    corners = locate_cheque(image, debug=debug)
    if corners is None:
        return None

    # where each corner should end up in the flat output image
    width, height = output_size(corners)

    target = np.array([[0,0], # top left
                       [width - 1, 0], # top right
                       [width - 1, height - 1], # bottom right
                       [0, height - 1]], # bottom left
                       dtype="float32")

    # work out the 'stretch' from photo corners to flat rectangle, then apply it
    M = cv2.getPerspectiveTransform(corners, target)

    return cv2.warpPerspective(image, M, (width, height))

# end of detector.py
