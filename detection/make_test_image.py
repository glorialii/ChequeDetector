"""Makes a fake, tilted 'cheque' photo so you can test before taking real photos.

    python detection/make_test_image.py      -> writes synthetic_cheque.jpg
"""
import cv2
import numpy as np

# A plain white "cheque" (600 x 250, about the real 2.4:1 ratio) with fake text.
cheque = np.full((250, 600, 3), 245, np.uint8)
cv2.putText(cheque, "PAY TO THE ORDER OF  Jane Example", (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2)
cv2.putText(cheque, "$ 123.45      VOID - SAMPLE", (20, 160),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2)

# Warp it onto a dark "table" at an angle, like a phone photo.
table = np.full((800, 1000, 3), (60, 45, 35), np.uint8)
src = np.float32([[0, 0], [600, 0], [600, 250], [0, 250]])
dst = np.float32([[180, 220], [820, 160], [860, 470], [150, 520]])
M = cv2.getPerspectiveTransform(src, dst)
warped = cv2.warpPerspective(cheque, M, (1000, 800))
mask = cv2.warpPerspective(np.full((250, 600), 255, np.uint8), M, (1000, 800))
table[mask > 0] = warped[mask > 0]
cv2.imwrite("synthetic_cheque.jpg", table)
print("wrote synthetic_cheque.jpg")
