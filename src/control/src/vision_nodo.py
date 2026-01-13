#!/usr/bin/env python3
import cv2
import numpy as np
import yaml
import os
import math

# --- IMPORTS DE ROS ---
import rospy
from geometry_msgs.msg import Pose 
# Importamos transformaciones para que el robot mire hacia abajo correctamente
from tf.transformations import quaternion_from_euler
from math import pi

class DiceDetectorPro:
    def __init__(self, cam_index=4, calib_file="ost.yaml"):
        # ============================================================
        # 0. INICIAR ROS
        # ============================================================
        rospy.init_node('dice_vision_node', anonymous=True)
        
        # El publisher envía mensajes tipo Pose
        self.pub = rospy.Publisher('/ur_move_to_pose', Pose, queue_size=10)
        
        # --- CONFIGURACIÓN DE MOVIMIENTO ROBOT ---
        # Altura en metros a la que se enviará el robot (para no chocar con la mesa)
        self.Z_HOVER = 0.15 
        
        # Orientación de la garra: Girada 180 grados en X para mirar hacia abajo
        # (Ajusta estos valores según la base de tu UR)
        q = quaternion_from_euler(pi, 0, 0) 
        self.target_orientation = q # Guardamos el cuaternión [x, y, z, w]

        # ============================================================
        # --- DIMENSIONES REALES ---
        # ============================================================
        self.RECT_WIDTH_CM = 40.0   
        self.RECT_HEIGHT_CM = 24.5  
        
        self.click_point = None 
        print(f"✅ Nodo ROS iniciado.\n   -> Altura Z: {self.Z_HOVER}m\n   -> Orientación 'Mirar Abajo' configurada.")

        # ============================================================
        # 1. CONFIGURACIÓN CÁMARA
        # ============================================================
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.calib_path = os.path.join(script_dir, calib_file)
        
        self.lente_calibrada = False
        self.mtx = None
        self.dist = None
        self.newcameramtx = None
        
        if os.path.exists(self.calib_path):
            try:
                with open(self.calib_path, 'r') as f:
                    data = yaml.safe_load(f)
                    self.mtx = np.array(data['camera_matrix']['data']).reshape(3, 3)
                    self.dist = np.array(data['distortion_coefficients']['data'])
                    self.lente_calibrada = True
                    print("✅ Calibración de lente cargada.")
            except Exception as e: 
                print(f"⚠️ Error cargando yaml: {e}")
        else:
            print("⚠️ No se encontró ost.yaml")

        self.cap = cv2.VideoCapture(cam_index)
        self.clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4,4))

        # ============================================================
        # 2. CONFIGURACIÓN ARUCO
        # ============================================================
        self.use_new_api = False
        try:
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_100)
            self.aruco_params = cv2.aruco.DetectorParameters()
            self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
            self.use_new_api = True
        except AttributeError:
            self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_100)
            self.aruco_params = cv2.aruco.DetectorParameters_create()
            self.use_new_api = False

        self.escala_cm_por_pixel = None 
        self.origen_px = (0, 0)
        self.roi_polygon = None 

        self.THRESH_FONDO = 170
        self.AREA_MIN = 100
        self.AREA_MAX = 5000
        self.samples = []

    def corregir_imagen(self, img):
        if self.lente_calibrada:
            h, w = img.shape[:2]
            if self.newcameramtx is None:
                self.newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (w,h), 1, (w,h))
            return cv2.undistort(img, self.mtx, self.dist, None, self.newcameramtx)
        return img

    def ordenar_puntos_roi(self, pts):
        pts = sorted(pts, key=lambda k: k[1]) 
        top = sorted(pts[:2], key=lambda k: k[0])
        bottom = sorted(pts[2:], key=lambda k: k[0], reverse=True)
        return np.array([top[0], top[1], bottom[0], bottom[1]], dtype=np.int32)

    def procesar_arucos_y_escala(self, img, gray):
        if self.use_new_api:
            corners, ids, rejected = self.detector.detectMarkers(gray)
        else:
            corners, ids, rejected = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
        
        self.escala_cm_por_pixel = None 
        self.roi_polygon = None
        self.origen_px = (0,0)

        if ids is not None and len(ids) >= 4:
            centers = []
            for marker in corners:
                c = np.mean(marker[0], axis=0)
                centers.append([int(c[0]), int(c[1])])
            
            roi_pts = self.ordenar_puntos_roi(centers)
            self.roi_polygon = roi_pts 
            
            tl = roi_pts[0]
            bl = roi_pts[3]
            self.origen_px = tuple(tl)
            
            dist_px_vertical = np.linalg.norm(tl - bl)
            if dist_px_vertical > 0:
                self.escala_cm_por_pixel = self.RECT_HEIGHT_CM / dist_px_vertical
                
                # Visualización
                cv2.polylines(img, [roi_pts], True, (255, 0, 0), 2)
                cv2.circle(img, self.origen_px, 8, (0, 0, 255), -1)
                # cv2.putText(img, "ORIGEN", (tl[0], tl[1]-25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        return img

    def pixel_a_robot_pose(self, cx, cy):
        if self.escala_cm_por_pixel is None: return None, None, None, None
        dx_px = cx - self.origen_px[0] 
        dy_px = cy - self.origen_px[1]
        real_x_cm = dx_px * self.escala_cm_por_pixel
        real_y_cm = dy_px * self.escala_cm_por_pixel
        
        # Conversión a metros y sistema de coordenadas del robot
        # NOTA: Verifica los signos según tu setup físico
        x_meters = (real_x_cm / 100.0) * -1 
        y_meters = (real_y_cm / 100.0)      
        return x_meters, y_meters, real_x_cm, real_y_cm

    def configurar_sistema(self):
        cv2.namedWindow("CONFIGURACION")
        cv2.createTrackbar("Umbral Fondo", "CONFIGURACION", 170, 255, lambda x: None)
        cv2.setMouseCallback("CONFIGURACION", self.mouse_click_config)
        
        print("--- CALIBRANDO (Espacio para guardar, ESC para salir) ---")
        detected_areas = []

        while not rospy.is_shutdown():
            ret, frame = self.cap.read()
            if not ret: break
            frame = self.corregir_imagen(frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            disp = frame.copy()
            disp = self.procesar_arucos_y_escala(disp, gray)

            blur = cv2.GaussianBlur(gray, (5,5), 0)
            th_val = cv2.getTrackbarPos("Umbral Fondo", "CONFIGURACION")
            _, binaria = cv2.threshold(blur, th_val, 255, cv2.THRESH_BINARY)
            
            # Solo mostrar mascara dentro de la ROI si existe
            if self.roi_polygon is not None:
                mask_roi_vis = np.zeros_like(gray)
                cv2.fillPoly(mask_roi_vis, [self.roi_polygon], 255)
                binaria_vis = cv2.bitwise_and(binaria, binaria, mask=mask_roi_vis)
                cv2.imshow("Mascara ROI", binaria_vis)
            else:
                cv2.imshow("Mascara ROI", binaria)

            cnts, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Chequear clicks de usuario para muestrear áreas
            for pt in self.samples:
                for c in cnts:
                    if cv2.pointPolygonTest(c, pt, False) >= 0:
                        area = cv2.contourArea(c)
                        cv2.drawContours(disp, [c], -1, (0, 255, 0), 2)
                        # Evitar duplicados exactos
                        if not any(abs(a - area) < 50 for a in detected_areas):
                            detected_areas.append(area)
                        break
            
            txt = f"Areas guardadas: {len(detected_areas)}"
            cv2.putText(disp, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            cv2.imshow("CONFIGURACION", disp)
            
            key = cv2.waitKey(1)
            if key == 32: # Espacio
                if detected_areas:
                    self.THRESH_FONDO = th_val
                    self.AREA_MIN = min(detected_areas) * 0.7
                    self.AREA_MAX = max(detected_areas) * 1.3
                    print(f"✅ CONFIGURADO. Area min: {self.AREA_MIN:.0f}, Max: {self.AREA_MAX:.0f}")
                    break
            elif key == 27:
                rospy.signal_shutdown("Cancelado por usuario")
                break
        
        cv2.destroyWindow("CONFIGURACION")
        cv2.destroyWindow("Mascara ROI")

    def mouse_click_config(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN: self.samples.append((x, y))

    def mouse_click_run(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.click_point = (x, y)
            print(f"🖱️ Objetivo marcado en pantalla: ({x}, {y})")

    def analizar_puntos_v18(self, gray_full, contour, box):
        # ... (Misma lógica de detección de puntos de dado que tenías) ...
        mask = np.zeros_like(gray_full)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        die_area = cv2.contourArea(contour)
        mask_sin_bordes = cv2.erode(mask, np.ones((3,3), np.uint8), iterations=(1 if die_area<1000 else 3))
        
        x, y, w, h = cv2.boundingRect(contour)
        roi_gray = gray_full[y:y+h, x:x+w]
        roi_mask = mask_sin_bordes[y:y+h, x:x+w]
        
        if roi_gray.size == 0: return 0, None
        roi_enhanced = self.clahe.apply(roi_gray)
        otsu_val, _ = cv2.threshold(roi_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, bin_dots = cv2.threshold(roi_enhanced, otsu_val * 0.85, 255, cv2.THRESH_BINARY_INV)
        bin_dots = cv2.bitwise_and(bin_dots, bin_dots, mask=roi_mask)
        cnts_dots, _ = cv2.findContours(bin_dots, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        puntos_validos = 0
        puntos_feos = 0 
        
        debug_roi = cv2.cvtColor(roi_enhanced, cv2.COLOR_GRAY2BGR)
        
        for cd in cnts_dots:
            area_dot = cv2.contourArea(cd)
            if (die_area * 0.005) < area_dot < (die_area * 0.25): 
                perim = cv2.arcLength(cd, True)
                if perim == 0: continue
                circularity = 4 * np.pi * area_dot / (perim * perim)
                
                if circularity > 0.40:
                    puntos_validos += 1
                    cv2.drawContours(debug_roi, [cd], -1, (0, 255, 0), -1)
                elif circularity > 0.25: 
                    puntos_feos += 1
                    cv2.drawContours(debug_roi, [cd], -1, (0, 255, 255), -1)

        resultado = puntos_validos
        if puntos_validos < 6 and (puntos_validos + puntos_feos) >= 6:
            resultado = 6
            cv2.putText(debug_roi, "FIX", (5, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1)

        return resultado, cv2.resize(debug_roi, (100, 100))

    def run(self):
        self.configurar_sistema()
        
        window_name = "Sistema Vision Dados"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.mouse_click_run)
        
        while not rospy.is_shutdown():
            ret, frame = self.cap.read()
            if not ret: break
            
            frame = self.corregir_imagen(frame)
            disp = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            disp = self.procesar_arucos_y_escala(disp, gray)
            
            if self.roi_polygon is not None:
                mask_roi = np.zeros_like(gray)
                cv2.fillPoly(mask_roi, [self.roi_polygon], 255)
                
                blur = cv2.GaussianBlur(gray, (5,5), 0)
                _, binaria = cv2.threshold(blur, self.THRESH_FONDO, 255, cv2.THRESH_BINARY)
                binaria_roi = cv2.bitwise_and(binaria, binaria, mask=mask_roi)
                
                cnts, _ = cv2.findContours(binaria_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                collage_debug = None
                
                for c in cnts:
                    area = cv2.contourArea(c)
                    if self.AREA_MIN < area < self.AREA_MAX:
                        hull = cv2.convexHull(c)
                        solidity = area / float(cv2.contourArea(hull))
                        if solidity > 0.85:
                            rect = cv2.minAreaRect(c)
                            box = np.int32(cv2.boxPoints(rect))
                            (cx_float, cy_float), _, _ = rect
                            
                            num, img_dbg = self.analizar_puntos_v18(gray, c, box)
                            
                            if 1 <= num <= 6:
                                r_xm, r_ym, r_x_cm, r_y_cm = self.pixel_a_robot_pose(cx_float, cy_float)
                                
                                # Texto informativo en pantalla
                                if r_x_cm is not None:
                                    coord_text = f"Val:{num} | X:{r_x_cm:.1f} Y:{r_y_cm:.1f}"
                                    cv2.putText(disp, coord_text, (box[1][0], box[1][1]-10), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

                                color_box = (0, 255, 0)
                                should_send = False
                                
                                # Verificar si el usuario hizo click dentro de este dado
                                if self.click_point is not None:
                                    if cv2.pointPolygonTest(box, self.click_point, False) >= 0:
                                        should_send = True
                                        color_box = (0, 0, 255)
                                
                                cv2.drawContours(disp, [box], 0, color_box, 2)
                                
                                if should_send and r_xm is not None:
                                    # --- CREACIÓN Y ENVÍO DEL POSE ---
                                    msg = Pose()
                                    msg.position.x = r_xm
                                    msg.position.y = r_ym
                                    msg.position.z = self.Z_HOVER # Usamos altura segura
                                    
                                    # Asignamos la orientación pre-calculada "Mirar Abajo"
                                    msg.orientation.x = self.target_orientation[0]
                                    msg.orientation.y = self.target_orientation[1]
                                    msg.orientation.z = self.target_orientation[2]
                                    msg.orientation.w = self.target_orientation[3]
                                    
                                    self.pub.publish(msg)
                                    rospy.loginfo(f"🚀 ENVIADO POSE: X={r_xm:.3f} Y={r_ym:.3f} Z={self.Z_HOVER} | Dado: {num}")
                                    
                                    self.click_point = None # Reset click
                                    cv2.putText(disp, "ENVIADO!", (int(cx_float), int(cy_float)), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

                                # Collage de depuración para ver los puntos detectados
                                if collage_debug is None: collage_debug = img_dbg
                                else: collage_debug = np.hstack((collage_debug, img_dbg))
                
                if collage_debug is not None: cv2.imshow("Debug Puntos", collage_debug)
            else:
                cv2.putText(disp, "BUSCANDO 4 ARUCOS...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            cv2.imshow(window_name, disp)
            if cv2.waitKey(1) == 27: break 
            
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        DiceDetectorPro().run()
    except rospy.ROSInterruptException:
        pass