import cv2
import os
import glob
import numpy as np

CARPETA = "Python/Tests/test_images"
THRESHOLD_BLACK = 80 # Umbral para la línea negra

TOP_RATIO = 0.80 # Parte inferior de la imagen

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
    # Convertir a blanco y negro
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(
        gray,
        THRESHOLD_BLACK,
        255,
        cv2.THRESH_BINARY_INV
    )

    h, w = thresh.shape
    y_top = int(h * TOP_RATIO)

    # Estados
    interseccion = False
    fin_linea = False
    curva_fuerte = False
    cuadro_naranja = False

    # ------------------------------------------------------
    # Buscar contornos
    # ------------------------------------------------------
    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Si no hay contornos
    if len(contours) == 0:
        fin_linea = True

    else:
        # Contorno más grande
        c = max(contours, key=cv2.contourArea)

        # --------------------------------------------------
        # Centro global del contorno
        # --------------------------------------------------
        M = cv2.moments(c)

        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # Punto rojo
            cv2.circle(img, (cx, cy), 6, (0, 0, 255), -1)

        # --------------------------------------------------
        # Dibujar contorno en máscara
        # --------------------------------------------------
        contour_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(contour_mask, [c], -1, 255, 3)

        ys, xs = np.where(contour_mask > 0)

        # Pintar contorno:
        # Azul arriba
        # Celeste abajo
        for y, x in zip(ys, xs):
            if y < y_top:
                img[y, x] = (222, 0, 0)
            else:
                img[y, x] = (255, 255, 0)

        # --------------------------------------------------
        # Obtener la parte inferior de la línea
        # --------------------------------------------------
        mask_bottom = ys >= y_top
        xs_bot = xs[mask_bottom]
        ys_bot = ys[mask_bottom]

        # Si no hay suficientes puntos abajo
        if len(xs_bot) <= 5:
            fin_linea = True

        else:
            cuadro_naranja = True

            # --------------------------------------------------
            # Rectángulo naranja (parte inferior)
            # --------------------------------------------------
            x_min = np.min(xs_bot)
            x_max = np.max(xs_bot)
            y_min = np.min(ys_bot)
            y_max = np.max(ys_bot)

            expand = 10

            x1 = max(0, x_min - expand)
            y1 = max(0, y_min)
            x2 = min(w - 1, x_max + expand)
            y2 = min(h - 1, y_max)

            # --------------------------------------------------
            #  Dibujar dos líneas verticales:
            #  en x1 y otra en x2, de la linea segura(la inferior)
            # --------------------------------------------------
            cv2.line(
                img,
                (x1, 0),
                (x1, h),
                (0, 100, 255),
                2
            )

            cv2.line(
                img,
                (x2, 0),
                (x2, h),
                (0, 100, 255),
                2
            )

            # Un rectángulo naranja alrededor de la parte inferior
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
            # Si hay muchos píxeles fuera del ancho
            # de la línea inferior, es intersección.
            margen = 40

            izquierda = np.sum(xs < x1 - margen)
            derecha = np.sum(xs > x2 + margen)

            cv2.putText(
                img,
                f"L:{izquierda} R:{derecha}",
                (20, 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2
            )

            if izquierda > 300 and derecha > 300:
                interseccion = True

                # ==================================================
                # CUADROS VERDES (SOLO SI HAY INTERSECCIÓN)
                # ==================================================
                puntos_laterales = (
                    (xs < x1 - margen) |
                    (xs > x2 + margen)
                )

                ys_laterales = ys[puntos_laterales]

                if len(ys_laterales) > 0:

                    # Parte más baja de la barra horizontal negra
                    y_barra = np.max(ys_laterales)

                    # Tamaño de los cuadros verdes
                    box_width = 120
                    box_height = 120

                    # Separación vertical (eje Y)
                    separacion_y = -1

                    # Separación horizontal (eje X)
                    separacion_x = 50

                    # Posición vertical de los cuadros
                    gy1 = min(h - 1, y_barra + separacion_y)
                    gy2 = min(h - 1, gy1 + box_height)

                    # ==================================================
                    # CUADRO IZQUIERDO
                    # ==================================================
                    # Se mueve más hacia la izquierda con separacion_x
                    gx1_left = max(0, x1 - box_width // 2 - separacion_x)
                    gx2_left = min(w - 1, x1 + box_width // 2 - separacion_x)

                    # ==================================================
                    # CUADRO DERECHO
                    # ==================================================
                    # Se mueve más hacia la derecha con separacion_x
                    gx1_right = max(0, x2 - box_width // 2 + separacion_x)
                    gx2_right = min(w - 1, x2 + box_width // 2 + separacion_x)

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

                    # Contar píxeles negros
                    roi_left = thresh[gy1:gy2, gx1_left:gx2_left]
                    roi_right = thresh[gy1:gy2, gx1_right:gx2_right]

                    pix_left = cv2.countNonZero(roi_left)
                    pix_right = cv2.countNonZero(roi_right)

                    # Mostrar conteos
                    cv2.putText(
                        img,
                        f"L:{pix_left} R:{pix_right}",
                        (20, 145),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )
            # --------------------------------------------------
            # Cuadro morado arriba (para detectar línea o todo blanco)
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

            # --------------------------------------------------
            # Revisar si hay línea en el cuadro morado o esta vacio(Blanco)
            # --------------------------------------------------
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

            # --------------------------------------------------
            # Fin de línea o curva fuerte
            # --------------------------------------------------
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

    # Mostrar estado
    cv2.putText(
        img,
        texto,
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    return img, thresh


# ==========================================================
# LOOP PRINCIPAL
# ==========================================================
while True:
    img = cv2.imread(files[index])

    if img is None:
        print("Error cargando:", files[index])
        break

    processed, mask = process_image(img.copy())

    # Mostrar número de imagen
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

    # Teclas:
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