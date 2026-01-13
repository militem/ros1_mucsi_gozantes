#!/usr/bin/env python3
import cv2
import numpy as np
import yaml
import os
import math

# --- IMPORTS DE ROS ---
import rospy
from geometry_msgs.msg import Pose 

class DiceDetectorPro:
    def __init__(self, cam_index=4, calib_file="ost.yaml"):
        # ============================================================
        # 0. INICIAR ROS
        # ============================================================
        rospy.init_node('dice_vision_node', anonymous=True)
        self.pub = rospy.Publisher('/ur_move_to_pose', Pose, queue_size=10)
        
        # --- CONSTANTES ---
        self.ROBOT_Z_METERS = 0.0
        self.DIST_REAL_CENTROS_CM = 24.5  
        self.click_point = None 
        
        print("✅ Nodo ROS iniciado. Modo ALTA PRECISIÓN (2 ArUcos) activado.")

        # ============================================================
        # 1. CONFIGURACIÓN CÁMARA (NUEVA RUTA)
        # ============================================================
        
        # 1. Obtenemos la ruta donde está este script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 2. Construimos la ruta: script_dir/calibrationdata/ost.yaml
        self.calib_path = os.path.join(script_dir, "calibrationdata", calib_file)
        
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
                    print(f"✅ Calibración cargada desde: {self.calib_path}")
            except Exception as e:
                print(f"⚠️ Error leyendo archivo de calibración: {e}")
        else:
            print(f"⚠️ NO se encontró calibración en: {self.calib_path}")
            print("   El sistema funcionará, pero las medidas serán menos precisas (efecto ojo de pez).")

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

        self.MEDIDA_ARUCO_INDIVIDUAL_CM = 5.0
        self.escala_cm_por_pixel = None 
        self.origen_px = (0, 0) 

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

    def procesar_arucos_y_escala(self, img, gray):
        if self.use_new_api:
            corners, ids, rejected = self.detector.detectMarkers(gray)
        else:
            corners, ids, rejected = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
        
        self.escala_cm_por_pixel = None 
        roi_coords = None 

        if ids is not None and len(ids) > 0:
            centers = []
            for marker in corners:
                c = np.mean(marker[0], axis=0)
                centers.append(c)
            
            centers = np.array(centers)
            idx_min_y = np.argmin(centers[:, 1]) 
            self.origen_px = (int(centers[idx_min_y][0]), int(centers[idx_min_y][1]))

            if len(ids) >= 2:
                # MODO 2 ARUCOS (Alta precisión)
                p1 = centers[0]
                p2 = centers[1]
                dist_px = np.linalg.norm(p1 - p2)
                
                if dist_px > 0:
                    self.escala_cm_por_pixel = self.DIST_REAL_CENTROS_CM / dist_px
                    
                    pt1 = (int(p1[0]), int(p1[1]))
                    pt2 = (int(p2[0]), int(p2[1]))
                    cv2.line(img, pt1, pt2, (255, 0, 255), 2)
                    mid_x = int((pt1[0]+pt2[0])/2)
                    mid_y = int((pt1[1]+pt2[1])/2)
                    cv2.putText(img, f"{self.DIST_REAL_CENTROS_CM}cm", (mid_x, mid_y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255), 2)
            else:
                # MODO 1 ARUCO (Fallback)
                aruco_perimeter = cv2.arcLength(corners[0], True)
                pixel_side = aruco_perimeter / 4.0
                if pixel_side > 0:
                    self.escala_cm_por_pixel = self.MEDIDA_ARUCO_INDIVIDUAL_CM / pixel_side
                    cv2.putText(img, "CALIB: 1 MARKER (BAJA PRECISION)", (20, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            all_points = []
            for marker in corners:
                for pt in marker[0]: all_points.append(pt)
            
            all_points = np.array(all_points)
            min_x = int(np.min(all_points[:,0]))
            min_y = int(np.min(all_points[:,1]))
            max_x = int(np.max(all_points[:,0]))
            max_y = int(np.max(all_points[:,1]))
            
            roi_x1 = max_x + 20
            roi_y1 = max(0, min_y - 20)
            roi_x2 = min(img.shape[1], roi_x1 + 800)
            roi_y2 = min(img.shape[0], max_y + 20)
            
            if roi_x2 > roi_x1 and roi_y2 > roi_y1:
                roi_coords = (roi_x1, roi_y1, roi_x2, roi_y2)
                cv2.rectangle(img, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 255, 0), 2)

            cv2.circle(img, self.origen_px, 5, (0, 0, 255), -1)
            cv2.putText(img, "ORIGEN (0,0)", (self.origen_px[0], self.origen_px[1]-20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

        return img, roi_coords

    def pixel_a_robot_pose(self, cx, cy):
        if self.escala_cm_por_pixel is None: return None, None
        
        distancia_x_pixels = cx - self.origen_px[0] 
        distancia_y_pixels = cy - self.origen_px[1]

        real_x_cm = distancia_x_pixels * self.escala_cm_por_pixel
        real_y_cm = distancia_y_pixels * self.escala_cm_por_pixel
        
        x_meters = (real_x_cm / 100.0) * -1
        y_meters = (real_y_cm / 100.0)
        
        return x_meters, y_meters

    def configurar_sistema(self):
        cv2.namedWindow("CONFIGURACION")
        cv2.createTrackbar("Umbral Fondo", "CONFIGURACION", 170, 255, lambda x: None)
        cv2.setMouseCallback("CONFIGURACION", self.mouse_click_config)
        
        print("--- CALIBRACIÓN ---")
        detected_areas = []

        while not rospy.is_shutdown():
            ret, frame = self.cap.read()
            if not ret: break
            frame = self.corregir_imagen(frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5,5), 0)
            
            th_val = cv2.getTrackbarPos("Umbral Fondo", "CONFIGURACION")
            _, binaria = cv2.threshold(blur, th_val, 255, cv2.THRESH_BINARY)
            
            disp = frame.copy()
            cnts, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for pt in self.samples:
                for c in cnts:
                    if cv2.pointPolygonTest(c, pt, False) >= 0:
                        area = cv2.contourArea(c)
                        cv2.drawContours(disp, [c], -1, (0, 255, 0), 2)
                        if not any(abs(a - area) < 50 for a in detected_areas):
                            detected_areas.append(area)
                        break
            
            txt = f"Detectados: {len(detected_areas)}"
            if detected_areas: txt += f" | Area: {int(min(detected_areas))}-{int(max(detected_areas))}"
            cv2.putText(disp, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            cv2.imshow("CONFIGURACION", disp)
            cv2.imshow("Mascara", binaria)
            
            key = cv2.waitKey(1)
            if key == 32: # Espacio
                if detected_areas:
                    self.THRESH_FONDO = th_val
                    self.AREA_MIN = min(detected_areas) * 0.7
                    self.AREA_MAX = max(detected_areas) * 1.3
                    print(f"Calibrado. Area min: {self.AREA_MIN:.0f}, Max: {self.AREA_MAX:.0f}")
                    break
            elif key == 27:
                rospy.signal_shutdown("Usuario canceló")
                break
        
        cv2.destroyWindow("CONFIGURACION")
        cv2.destroyWindow("Mascara")

    def mouse_click_config(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.samples.append((x, y))

    def mouse_click_run(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.click_point = (x, y)
            print(f"Click detectado en ({x}, {y})")

    def analizar_puntos_v18(self, gray_full, contour, box):
        mask = np.zeros_like(gray_full)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        die_area = cv2.contourArea(contour)
        
        iteraciones = 1 if die_area < 1000 else 3
        mask_sin_bordes = cv2.erode(mask, np.ones((3,3), np.uint8), iterations=iteraciones)
        
        x, y, w, h = cv2.boundingRect(contour)
        roi_gray = gray_full[y:y+h, x:x+w]
        roi_mask = mask_sin_bordes[y:y+h, x:x+w]
        
        if roi_gray.size == 0: return 0, None

        roi_enhanced = self.clahe.apply(roi_gray)
        otsu_val, _ = cv2.threshold(roi_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        _, bin_dots = cv2.threshold(roi_enhanced, otsu_val * 0.85, 255, cv2.THRESH_BINARY_INV)
        bin_dots = cv2.bitwise_and(bin_dots, bin_dots, mask=roi_mask)
        
        cnts_dots, _ = cv2.findContours(bin_dots, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        puntos = 0
        debug_roi = cv2.cvtColor(roi_enhanced, cv2.COLOR_GRAY2BGR)
        
        for cd in cnts_dots:
            area_dot = cv2.contourArea(cd)
            if (die_area * 0.005) < area_dot < (die_area * 0.20): 
                perim = cv2.arcLength(cd, True)
                if perim == 0: continue
                circularity = 4 * np.pi * area_dot / (perim * perim)
                
                es_punto_valido = False
                if circularity > 0.50: es_punto_valido = True
                elif circularity > 0.40 and area_dot > (die_area * 0.02): es_punto_valido = True
                
                if es_punto_valido:
                    puntos += 1
                    cv2.drawContours(debug_roi, [cd], -1, (0, 255, 0), -1)
                else:
                    cv2.drawContours(debug_roi, [cd], -1, (0, 0, 255), 1)

        return puntos, cv2.resize(debug_roi, (100, 100))

    def run(self):
        self.configurar_sistema()
        
        window_name = "Sistema Final ROS"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.mouse_click_run)
        
        while not rospy.is_shutdown():
            ret, frame = self.cap.read()
            if not ret: break
            
            frame = self.corregir_imagen(frame)
            disp = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            disp, roi_coords = self.procesar_arucos_y_escala(disp, gray)
            collage_debug = None
            
            if roi_coords is not None:
                x1, y1, x2, y2 = roi_coords
                roi_gray_process = gray[y1:y2, x1:x2]
                
                blur = cv2.GaussianBlur(roi_gray_process, (5,5), 0)
                _, binaria = cv2.threshold(blur, self.THRESH_FONDO, 255, cv2.THRESH_BINARY)
                
                cnts, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for c_local in cnts:
                    area = cv2.contourArea(c_local)
                    if self.AREA_MIN < area < self.AREA_MAX:
                        c_global = c_local + [x1, y1]
                        hull = cv2.convexHull(c_global)
                        solidity = area / float(cv2.contourArea(hull))
                        
                        if solidity > 0.85:
                            rect = cv2.minAreaRect(c_global)
                            box = np.int32(cv2.boxPoints(rect))
                            (cx_float, cy_float), _, _ = rect
                            
                            num, img_dbg = self.analizar_puntos_v18(gray, c_global, box)
                            
                            if 1 <= num <= 6:
                                color_box = (0, 255, 0)
                                should_send = False
                                
                                if self.click_point is not None:
                                    if cv2.pointPolygonTest(box, self.click_point, False) >= 0:
                                        should_send = True
                                        color_box = (0, 0, 255)
                                
                                cv2.drawContours(disp, [box], 0, color_box, 2)
                                r_xm, r_ym = self.pixel_a_robot_pose(cx_float, cy_float)

                                if should_send and r_xm is not None:
                                    msg = Pose()
                                    msg.position.x = r_xm
                                    msg.position.y = r_ym
                                    msg.position.z = self.ROBOT_Z_METERS
                                    msg.orientation.w = 1.0

                                    self.pub.publish(msg)
                                    rospy.loginfo(f"ENVIADO POSE! Dado: {num} | X:{r_xm:.3f} Y:{r_ym:.3f}")
                                    
                                    self.click_point = None
                                    cv2.putText(disp, "ENVIADO!", (int(cx_float), int(cy_float)), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

                                info_txt = f"Val: {num}"
                                cv2.putText(disp, info_txt, (int(cx_float)-20, int(cy_float)), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                                
                                coords_txt = f"X:{r_xm:.3f}m Y:{r_ym:.3f}m"
                                cv2.putText(disp, coords_txt, (int(cx_float)-80, int(cy_float)+40), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
                                cv2.line(disp, (int(cx_float), int(cy_float)), self.origen_px, (0, 255, 255), 1)

                                if collage_debug is None: collage_debug = img_dbg
                                else: collage_debug = np.hstack((collage_debug, img_dbg))
            else:
                cv2.putText(disp, "BUSCANDO ARUCOS...", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            cv2.putText(disp, "HAZ CLICK PARA MOVER ROBOT", (20, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            cv2.imshow("Sistema Final ROS", disp)
            if collage_debug is not None:
                cv2.imshow("Debug Puntos", collage_debug)
            
            if cv2.waitKey(1) == 27: break 
            
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        DiceDetectorPro().run()
    except rospy.ROSInterruptException:
        pass