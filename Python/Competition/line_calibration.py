import cv2
import numpy as np

def nothing(x):
    pass

cv2.namedWindow("Threshold (INVERTED)")

# Trackbars
cv2.createTrackbar("Min Threshold", "Threshold (INVERTED)", 0, 255, nothing)
cv2.createTrackbar("Max Threshold", "Threshold (INVERTED)", 255, 255, nothing)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("No se pudo abrir la cámara")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    min_val = cv2.getTrackbarPos("Min Threshold", "Threshold (INVERTED)")
    max_val = cv2.getTrackbarPos("Max Threshold", "Threshold (INVERTED)")

    if min_val > max_val:
        min_val = max_val

    # ======================================================
    # 1. umbral normal
    # ======================================================
    thresh = cv2.inRange(gray, min_val, max_val)

    # ======================================================
    # 2. INVERTIDO (línea negra = blanco)
    # ======================================================
    thresh = cv2.bitwise_not(thresh)

    # ======================================================
    # mostrar resultado
    # ======================================================
    cv2.imshow("Threshold (INVERTED)", thresh)
    cv2.imshow("Camera - LINE", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('3'):
        print(f"[LINE-INVERTED] Min: {min_val} | Max: {max_val}")

    elif key == 27 or key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()