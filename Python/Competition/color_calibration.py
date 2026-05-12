import cv2
import numpy as np

def nothing(x):
    pass

cv2.namedWindow("Threshold (NORMAL)")

# Trackbars
cv2.createTrackbar("Min Threshold", "Threshold (NORMAL)", 0, 255, nothing)
cv2.createTrackbar("Max Threshold", "Threshold (NORMAL)", 255, 255, nothing)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("No se pudo abrir la cámara")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    min_val = cv2.getTrackbarPos("Min Threshold", "Threshold (NORMAL)")
    max_val = cv2.getTrackbarPos("Max Threshold", "Threshold (NORMAL)")

    # evitar valores inválidos
    if min_val > max_val:
        min_val = max_val

    thresh = cv2.inRange(gray, min_val, max_val)

    cv2.imshow("Threshold (NORMAL)", thresh)
    cv2.imshow("Camera - COLOR", frame)


    # Teclas:
    # 3 = Imprimir valores de los trackbars
    # q = salir
    key = cv2.waitKey(1) & 0xFF

    if key == ord('3'):
        print(f"[COLOR-NORMAL] Min: {min_val} | Max: {max_val}")

    elif key == 27 or key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()