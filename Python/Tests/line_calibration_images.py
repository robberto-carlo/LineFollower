import cv2
import os
import glob
import numpy as np

CARPETA = "Python/Tests/test_images"

THRESH_MIN = 80
THRESH_MAX = 255

TOP_RATIO = 0.80

files = sorted(glob.glob(os.path.join(CARPETA, "*.png")))

if len(files) == 0:
    print("No images found in:", CARPETA)
    exit()

index = 0

# ==========================================================
# TRACKBAR
# ==========================================================
def nothing(x):
    pass

cv2.namedWindow("Threshold (INVERTIDO)")

cv2.createTrackbar("Min Threshold", "Threshold (INVERTIDO)", THRESH_MIN, 255, nothing)
cv2.createTrackbar("Max Threshold", "Threshold (INVERTIDO)", THRESH_MAX, 255, nothing)

# ==========================================================
# PROCESADO
# ==========================================================
def process(img, tmin, tmax):

    # Convertir a gris
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ======================================================
    # DETECCIÓN (Invertida)
    # ======================================================
    mask = cv2.inRange(gray, tmin, tmax)  # Umbral configurable
    mask = cv2.bitwise_not(mask) # Invertir para mostrar la línea en blanco

    output = img.copy()

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for c in contours:
        area = cv2.contourArea(c)

        if area > 100:
            cv2.drawContours(output, [c], -1, (0, 255, 0), 2)

    return output, mask

# ==========================================================
# LOOP PRINCIPAL
# ==========================================================
while True:

    img = cv2.imread(files[index])

    if img is None:
        print("Error loading:", files[index])
        break

    tmin = cv2.getTrackbarPos("Min Threshold", "Threshold (INVERTIDO)")
    tmax = cv2.getTrackbarPos("Max Threshold", "Threshold (INVERTIDO)")

    if tmin > tmax:
        tmin = tmax

    processed, mask = process(img, tmin, tmax)

    # ======================================================
    # INFO
    # ======================================================
    cv2.putText(
        processed,
        f"IMG {index+1}/{len(files)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    cv2.putText(
        processed,
        f"Min:{tmin} Max:{tmax}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    # ======================================================
    # VENTANAS
    # ======================================================
    cv2.imshow("Robot View - LINE", processed)
    cv2.imshow("Threshold (INVERTIDO)", mask)

    key = cv2.waitKey(30) & 0xFF

    if key == ord('2'):
        index = min(index + 1, len(files) - 1)

    elif key == ord('1'):
        index = max(index - 1, 0)

    elif key == ord('3'):
        print(f"[LINE-INVERTED-IMG] Min: {tmin} | Max: {tmax}")

    elif key == ord('q'):
        break

cv2.destroyAllWindows()