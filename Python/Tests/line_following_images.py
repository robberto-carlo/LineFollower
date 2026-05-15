import cv2
import os
import glob
import numpy as np

# ==========================================================
# CONFIGURACIÓN
# ==========================================================
CARPETA = "Python/Tests/test_images"

THRESHOLD_BLACK = 80      # Umbral para detectar la línea negra
TOP_RATIO = 0.80          # Parte inferior de la imagen

# Rango HSV para detectar verde
LOWER_GREEN = np.array([35, 60, 60])
UPPER_GREEN = np.array([85, 255, 255])

# Mínimo de píxeles verdes para considerar detección
GREEN_THRESHOLD = 500

# Cargar imágenes .png
files = sorted(glob.glob(os.path.join(CARPETA, "*.png")))

if len(files) == 0:
    print("No se encontraron imágenes en:", CARPETA)
    exit()

index = 0


# ==========================================================
# FUNCIÓN PRINCIPAL
# ==========================================================
def process_image(img):
    # ------------------------------------------------------
    # Convertir a escala de grises y binarizar
    # ------------------------------------------------------
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(
        gray,
        THRESHOLD_BLACK,
        255,
        cv2.THRESH_BINARY_INV
    )

    # ------------------------------------------------------
    # Máscara para detectar color verde
    # ------------------------------------------------------
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)

    h, w = thresh.shape
    y_top = int(h * TOP_RATIO)

    # Estados
    interseccion = False
    fin_linea = False
    curva_fuerte = False
    cuadro_naranja = False

    # Valores de salida para detección verde
    left_value = 0
    right_value = 0

    # ------------------------------------------------------
    # Buscar contornos
    # ------------------------------------------------------
    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        fin_linea = True

    else:
        # Contorno más grande
        c = max(contours, key=cv2.contourArea)

        # Centro del contorno
        M = cv2.moments(c)

        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(img, (cx, cy), 6, (0, 0, 255), -1)

        # Dibujar contorno en máscara
        contour_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(contour_mask, [c], -1, 255, 3)

        ys, xs = np.where(contour_mask > 0)

        # Pintar contorno
        for y, x in zip(ys, xs):
            if y < y_top:
                img[y, x] = (222, 0, 0)
            else:
                img[y, x] = (255, 255, 0)

        # Parte inferior
        mask_bottom = ys >= y_top
        xs_bot = xs[mask_bottom]
        ys_bot = ys[mask_bottom]

        if len(xs_bot) <= 5:
            fin_linea = True

        else:
            cuadro_naranja = True

            x_min = np.min(xs_bot)
            x_max = np.max(xs_bot)
            y_min = np.min(ys_bot)
            y_max = np.max(ys_bot)

            expand = 10

            x1 = max(0, x_min - expand)
            y1 = max(0, y_min)
            x2 = min(w - 1, x_max + expand)
            y2 = min(h - 1, y_max)

            # Líneas verticales
            cv2.line(img, (x1, 0), (x1, h), (0, 100, 255), 2)
            cv2.line(img, (x2, 0), (x2, h), (0, 100, 255), 2)

            # Rectángulo naranja
            cv2.rectangle(
                img,
                (x1, y1),
                (x2, y2),
                (0, 185, 255),
                2
            )

            # --------------------------------------------------
            # Detectar intersección
            # --------------------------------------------------
            margen = 40

            izquierda = np.sum(xs < x1 - margen)
            derecha = np.sum(xs > x2 + margen)

            if izquierda > 300 and derecha > 300:
                interseccion = True

                puntos_laterales = (
                    (xs < x1 - margen) |
                    (xs > x2 + margen)
                )

                ys_laterales = ys[puntos_laterales]

                if len(ys_laterales) > 0:
                    y_barra = np.max(ys_laterales)

                    box_width = 120
                    box_height = 120

                    separacion_y = -1
                    separacion_x = 50

                    gy1 = min(h - 1, y_barra + separacion_y)
                    gy2 = min(h - 1, gy1 + box_height)

                    # Cuadro izquierdo
                    gx1_left = max(
                        0,
                        x1 - box_width // 2 - separacion_x
                    )
                    gx2_left = min(
                        w - 1,
                        x1 + box_width // 2 - separacion_x
                    )

                    # Cuadro derecho
                    gx1_right = max(
                        0,
                        x2 - box_width // 2 + separacion_x
                    )
                    gx2_right = min(
                        w - 1,
                        x2 + box_width // 2 + separacion_x
                    )

                    # Dibujar cuadros verdes
                    cv2.rectangle(
                        img,
                        (gx1_left, gy1),
                        (gx2_left, gy2),
                        (0, 255, 0),
                        2
                    )

                    cv2.rectangle(
                        img,
                        (gx1_right, gy1),
                        (gx2_right, gy2),
                        (0, 255, 0),
                        2
                    )

                    # ==================================================
                    # DETECTAR VERDE EN LOS CUADROS
                    # ==================================================
                    roi_green_left = green_mask[
                        gy1:gy2,
                        gx1_left:gx2_left
                    ]

                    roi_green_right = green_mask[
                        gy1:gy2,
                        gx1_right:gx2_right
                    ]

                    green_left = cv2.countNonZero(roi_green_left)
                    green_right = cv2.countNonZero(roi_green_right)

                    # Convertir a 1 o 0
                    left_value = (
                        1 if green_left > GREEN_THRESHOLD else 0
                    )

                    right_value = (
                        1 if green_right > GREEN_THRESHOLD else 0
                    )

                    # Mostrar 1 o 0 en pantalla
                    cv2.putText(
                        img,
                        f"L:{left_value} R:{right_value}",
                        (20, 115),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )

            # --------------------------------------------------
            # Cuadro morado superior
            # --------------------------------------------------
            MARGEN = 60

            alto_box = y_max - y_min
            top_morado = max(
                MARGEN,
                y_min - int(alto_box * 0.15)
            )

            cv2.rectangle(
                img,
                (MARGEN, MARGEN),
                (w - MARGEN, top_morado),
                (255, 0, 157),
                2
            )

            # Revisar si hay línea en cuadro morado
            if top_morado > MARGEN:
                region = thresh[
                    MARGEN:top_morado,
                    MARGEN:w - MARGEN
                ]

                pixeles_negros = cv2.countNonZero(region)

                area = (
                    (top_morado - MARGEN)
                    * (w - 2 * MARGEN)
                )

                ratio = pixeles_negros / area if area > 0 else 0
            else:
                ratio = 0

            # Fin de línea o curva fuerte
            if ratio < 0.10:
                cx_bottom = int(np.mean(xs_bot))
                tercio = w // 3

                if tercio <= cx_bottom < 2 * tercio:
                    fin_linea = True
                else:
                    curva_fuerte = True

    # ======================================================
    # ESTADO FINAL
    # ======================================================
    if not cuadro_naranja:
        texto = "TODO BLANCO"
    elif interseccion:
        texto = "INTERSECCION"
    elif fin_linea:
        texto = "FIN DE LINEA"
    elif curva_fuerte:
        texto = "CURVA FUERTE"
    else:
        texto = "LINEA NORMAL"

    cv2.putText(
        img,
        texto,
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    return img, thresh, green_mask


# ==========================================================
# LOOP PRINCIPAL
# ==========================================================
while True:
    img = cv2.imread(files[index])

    if img is None:
        print("Error cargando:", files[index])
        break

    processed, mask, green_mask = process_image(img.copy())

    # Número de imagen
    cv2.putText(
        processed,
        f"IMG: {index + 1}/{len(files)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (33, 49, 231),
        2
    )

    cv2.imshow("Robot View", processed)
    cv2.imshow("Mask B/N", mask)
    cv2.imshow("Green Mask", green_mask)

    # 2 = siguiente
    # 1 = anterior
    # q = salir
    key = cv2.waitKey(0) & 0xFF

    if key == ord('2'):
        index = min(index + 1, len(files) - 1)

    elif key == ord('1'):
        index = max(index - 1, 0)

    elif key == ord('q'):
        break

cv2.destroyAllWindows()