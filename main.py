import cv2
import numpy as np
import pandas as pd
from datetime import datetime

points = []
img = cv2.imread("petri.jpg")
clone = img.copy()

# ---------------------------
# Click de puntos
# ---------------------------
def select_points(event, x, y, flags, param):
    global points, img

    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 3:
        points.append((x, y))

        # Dibujar punto
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)

        # Numerar
        cv2.putText(img, str(len(points)), (x+10, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

# ---------------------------
# Ordenar puntos
# ---------------------------
def ordenar_puntos(pts):
    pts = np.array(pts)

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    top_left = pts[np.argmin(s)]
    bottom_right = pts[np.argmax(s)]
    top_right = pts[np.argmin(diff)]

    return np.array([top_left, top_right, bottom_right], dtype="float32")

# ---------------------------
# Selección
# ---------------------------
cv2.namedWindow("Selecciona 3 marcas (ENTER para continuar)")
cv2.setMouseCallback("Selecciona 3 marcas (ENTER para continuar)", select_points)

while True:
    cv2.imshow("Selecciona 3 marcas (ENTER para continuar)", img)
    key = cv2.waitKey(1)

    if key == 13 and len(points) == 3:  # ENTER
        break

cv2.destroyAllWindows()

# ---------------------------
# Alineación
# ---------------------------
src_pts = ordenar_puntos(points)

dst_pts = np.array([
    [100, 100],
    [500, 100],
    [100, 500]
], dtype="float32")

M = cv2.getAffineTransform(src_pts, dst_pts)
aligned = cv2.warpAffine(clone, M, (700, 700))

# ---------------------------
# Detección de hongo
# ---------------------------
hsv = cv2.cvtColor(aligned, cv2.COLOR_BGR2HSV)

lower = np.array([10, 50, 50])
upper = np.array([40, 255, 255])

mask = cv2.inRange(hsv, lower, upper)

kernel = np.ones((5,5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

area_total = 0

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 50:
        area_total += area
        cv2.drawContours(aligned, [cnt], -1, (0,255,0), 2)

# ---------------------------
# Color promedio
# ---------------------------
mean_color = cv2.mean(aligned, mask=mask)

# ---------------------------
# Guardar en Excel
# ---------------------------
data = {
    "fecha": [datetime.now()],
    "area": [area_total],
    "B": [mean_color[0]],
    "G": [mean_color[1]],
    "R": [mean_color[2]]
}

df = pd.DataFrame(data)

file = "crecimiento_hongos.xlsx"

try:
    df_old = pd.read_excel(file)
    df = pd.concat([df_old, df], ignore_index=True)
except:
    pass

df.to_excel(file, index=False)

# ---------------------------
# Mostrar resultado
# ---------------------------
cv2.imshow("Alineado", aligned)
cv2.imshow("Mascara", mask)
cv2.waitKey(0)
cv2.destroyAllWindows()