import cv2
import numpy as np


# --------------------------------------------------
# Calibración HSV con ROI dibujado por el usuario
# --------------------------------------------------
def calibrate_skin_hsv(cap, samples=30):
    print("Dibuja un rectangulo sobre la piel y presiona ENTER")

    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("No se pudo capturar frame")

    frame = cv2.flip(frame, 1)

    # Selección manual del ROI
    roi = cv2.selectROI(
        "Seleccion de piel",
        frame,
        fromCenter=False,
        showCrosshair=True
    )

    cv2.destroyWindow("Seleccion de piel")

    x, y, w, h = roi
    if w == 0 or h == 0:
        raise RuntimeError("ROI invalido")

    hsv_samples = []

    print("Tomando muestras HSV...")
    for _ in range(samples):
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        roi_frame = frame[y:y+h, x:x+w]
        hsv_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)

        hsv_samples.append(hsv_roi.reshape(-1, 3))
        cv2.waitKey(30)

    hsv_samples = np.vstack(hsv_samples)

    lower = np.percentile(hsv_samples, 5, axis=0)
    upper = np.percentile(hsv_samples, 95, axis=0)

    lower = np.clip(lower - [5, 30, 30], 0, 255).astype(np.uint8)
    upper = np.clip(upper + [5, 30, 30], 0, 255).astype(np.uint8)

    print("HSV calibrado:")
    print("Lower:", lower)
    print("Upper:", upper)

    return lower, upper


# --------------------------------------------------
# Máscara de piel
# --------------------------------------------------
def skin_mask(frame, lower, upper):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.GaussianBlur(mask, (7, 7), 0)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    return mask


# --------------------------------------------------
# Contorno más grande
# --------------------------------------------------
def get_largest_contour(mask):
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


# --------------------------------------------------
# Detección de gestos
# --------------------------------------------------
def detect_gesture(contour):
    epsilon = 0.01 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    if len(approx) < 5:
        return "Unknown"

    hull = cv2.convexHull(approx, returnPoints=False)
    if hull is None or len(hull) < 3:
        return "Unknown"

    defects = cv2.convexityDefects(approx, hull)
    if defects is None:
        return "Unknown"

    finger_count = 0

    for i in range(defects.shape[0]):
        s, e, f, depth = defects[i, 0]
        depth /= 256.0

        start = approx[s][0]
        end = approx[e][0]
        far = approx[f][0]

        a = np.linalg.norm(end - start)
        b = np.linalg.norm(far - start)
        c = np.linalg.norm(end - far)

        angle = np.arccos((b*b + c*c - a*a) / (2*b*c + 1e-5))

        if depth > 15 and angle < np.pi / 2:
            finger_count += 1

    if finger_count == 0:
        x, y, w, h = cv2.boundingRect(approx)
        return "Index Finger Up" if h / float(w) > 1.3 else "Thumb Up"
    elif finger_count == 1:
        return "Two Fingers"
    elif finger_count >= 3:
        return "Open Palm"

    return "Unknown"


# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: no se pudo abrir la camara")
        return

    # === Calibración manual con ROI ===
    lower_hsv, upper_hsv = calibrate_skin_hsv(cap)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        mask = skin_mask(frame, lower_hsv, upper_hsv)
        contour = get_largest_contour(mask)

        gesture = "No Hand Detected"

        if contour is not None and cv2.contourArea(contour) > 3000:
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
            gesture = detect_gesture(contour)

        cv2.putText(frame, f"Gesture: {gesture}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 255, 0), 2)

        cv2.imshow("Skin Mask", mask)
        cv2.imshow("Gesture Detection", frame)

        key = cv2.waitKey(30) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('c'):
            lower_hsv, upper_hsv = calibrate_skin_hsv(cap)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
