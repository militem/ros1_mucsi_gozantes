#!/usr/bin/env python3
import cv2
import numpy as np
import yaml
import os
import time
import statistics
import math
from collections import Counter

# --- IMPORTS DE ROS ---
import rospy
from geometry_msgs.msg import Pose 

# ==========================================================
# ESTADOS DEL JUEGO
# ==========================================================
STATE_IDLE = "ESPERANDO_INICIO"
STATE_INIT_DEALER = "INICIAL_DEALER"
STATE_INIT_PLAYER = "INICIAL_JUGADOR"
STATE_PLAYER_CHOICE = "DECISION_JUGADOR"
STATE_DEALER_LOOP = "BUCLE_DEALER"
STATE_GAME_OVER = "FIN_JUEGO"

class BlackjackRobotico:
    def __init__(self, cam_index=0, calib_file="ost.yaml"):
        # --- ROS ---
        rospy.init_node('dice_blackjack_master', anonymous=True)
        self.pub = rospy.Publisher('/ur_move_to_pose', Pose, queue_size=10)
        
        # --- CONFIGURACION FISICA ---
        self.RECT_HEIGHT_CM = 24.5  
        self.RECT_WIDTH_CM = 40.0  
        self.CMD_THROW_FLAG = 7.0
        
        # Distancia mínima entre dados para considerarlos distintos (evitar duplicados)
        self.MIN_DIST_DADOS_CM = 1.1 
        
        # --- VISION ---
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.calib_path = os.path.join(script_dir, calib_file)
        
        # Inicializar cámara
        self.cap = cv2.VideoCapture(cam_index, cv2.CAP_V4L2) 
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(cam_index)

        self.clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4,4))
        
        # Cargar calibracion lente
        self.mtx, self.dist, self.newcameramtx = None, None, None
        self.lente_calibrada = False
        if os.path.exists(self.calib_path):
            try:
                with open(self.calib_path, 'r') as f:
                    data = yaml.safe_load(f)
                    self.mtx = np.array(data['camera_matrix']['data']).reshape(3, 3)
                    self.dist = np.array(data['distortion_coefficients']['data'])
                    self.lente_calibrada = True
            except Exception: pass

        # ArUco
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_100)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        # Variables de Calibración
        self.escala_cm_pixel = None
        self.origen_px = (0,0)
        self.roi_polygon = None
        self.marker_corners = [] 
        
        self.thresh_fondo = 170
        self.area_min = 100
        self.area_max = 5000
        
        self.lower_skin = None
        self.upper_skin = None

        # Variables Juego
        self.state = STATE_IDLE
        self.player_score = 0
        self.dealer_score = 0

    # ============================================================
    # 1. UTILIDADES VISIÓN
    # ============================================================
    def corregir_imagen(self, img):
        if self.lente_calibrada:
            h, w = img.shape[:2]
            if self.newcameramtx is None:
                self.newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (w,h), 1, (w,h))
            return cv2.undistort(img, self.mtx, self.dist, None, self.newcameramtx)
        return img

    def procesar_tablero_aruco(self, img, gray):
        corners, ids, _ = self.detector.detectMarkers(gray)
        self.roi_polygon = None
        self.marker_corners = []
        
        if ids is not None and len(ids) >= 4:
            self.marker_corners = corners
            
            centers = [np.mean(m[0], axis=0).astype(int) for m in corners]
            pts = sorted(centers, key=lambda k: k[1])
            top = sorted(pts[:2], key=lambda k: k[0])
            bot = sorted(pts[2:], key=lambda k: k[0], reverse=True)
            
            tl, tr = top[0], top[1]
            br, bl = bot[0], bot[1]
            
            h_px_left = np.linalg.norm(tl - bl)
            h_px_right = np.linalg.norm(tr - br)
            avg_h_px = (h_px_left + h_px_right) / 2.0
            
            w_px_top = np.linalg.norm(tl - tr)
            w_px_bot = np.linalg.norm(bl - br)
            avg_w_px = (w_px_top + w_px_bot) / 2.0
            
            if avg_h_px > 0 and avg_w_px > 0:
                scale_h = self.RECT_HEIGHT_CM / avg_h_px
                scale_w = self.RECT_WIDTH_CM / avg_w_px
                self.escala_cm_pixel = (scale_h + scale_w) / 2.0
                
                # Expandir ROI 2 cm
                px_margin = int(2.0 / self.escala_cm_pixel)
                tl[1] = max(0, tl[1] - px_margin)
                tr[1] = max(0, tr[1] - px_margin)
                bl[1] = min(img.shape[0], bl[1] + px_margin)
                br[1] = min(img.shape[0], br[1] + px_margin)
                
                roi = np.array([tl, tr, br, bl], dtype=np.int32)
                self.roi_polygon = roi
                self.origen_px = tuple(tl)
                
                cv2.polylines(img, [roi], True, (0, 255, 0), 2)
                cv2.putText(img, f"Escala: {self.escala_cm_pixel:.4f} cm/px", (roi[0][0], roi[0][1]-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                           
        return img

    def px_to_robot(self, cx, cy):
        if self.escala_cm_pixel is None: return None, None
        dx = (cx - self.origen_px[0]) * self.escala_cm_pixel
        dy = (cy - self.origen_px[1]) * self.escala_cm_pixel
        return (-dx/100.0), (dy/100.0)
    
    def distancia_euclidiana(self, p1, p2):
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    # ============================================================
    # 2. CALIBRACIÓN Y UI
    # ============================================================
    def esperar_confirmacion_usuario(self, mensaje="PULSA ESPACIO PARA CONTINUAR"):
        """ Función bloqueante para modo paso a paso """
        print(f"⏸️  ESPERANDO USUARIO: {mensaje}")
        while True:
            ret, frame = self.cap.read()
            if not ret: break
            frame = self.corregir_imagen(frame)
            self.dibujar_hud(frame)
            
            # Caja de texto centrada
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (50, h//2 - 40), (w-50, h//2 + 40), (0,0,0), -1)
            cv2.putText(frame, mensaje, (70, h//2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
            
            cv2.imshow("Sistema Blackjack", frame)
            if cv2.waitKey(1) == 32: # Espacio
                print("▶️  Continuando...")
                break

    def configurar_sistema_completo(self):
        print("🔧 INICIANDO PROTOCOLO DE CALIBRACIÓN...")
        cv2.namedWindow("1. ENCUADRE", cv2.WINDOW_NORMAL)
        
        # --- A) Calibración PIEL ---
        print("🖐️  PASO 1: Calibrar Color de Piel")
        while True:
            print("   -> Encuadra tu mano en el rectángulo y pulsa ENTER.")
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    time.sleep(1); continue
                frame = self.corregir_imagen(frame)
                h, w = frame.shape[:2]
                cx, cy = w//2, h//2
                cv2.rectangle(frame, (cx-50, cy-50), (cx+50, cy+50), (0,255,0), 2)
                cv2.putText(frame, "PULSA ENTER", (cx-60, cy-70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                cv2.imshow("1. ENCUADRE", frame)
                if cv2.waitKey(1) == 13: 
                    roi = frame[cy-50:cy+50, cx-50:cx+50]
                    break
            
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            self.lower_skin = np.array([hsv_roi[:,:,0].min(), hsv_roi[:,:,1].min(), hsv_roi[:,:,2].min()]) - [10, 40, 40]
            self.upper_skin = np.array([hsv_roi[:,:,0].max(), hsv_roi[:,:,1].max(), hsv_roi[:,:,2].max()]) + [10, 40, 40]
            self.lower_skin = np.clip(self.lower_skin, 0, 255)
            self.upper_skin = np.clip(self.upper_skin, 0, 255)
            
            print("   -> Verificando... Pulsa ESPACIO para aceptar.")
            accepted = False
            while True:
                ret, check_frame = self.cap.read()
                if not ret: break
                check_frame = self.corregir_imagen(check_frame)
                hsv_check = cv2.cvtColor(check_frame, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv_check, self.lower_skin, self.upper_skin)
                res = check_frame.copy()
                res[mask > 0] = (0, 255, 0) 
                cv2.imshow("1. ENCUADRE", res)
                k = cv2.waitKey(1)
                if k == 32: 
                    accepted = True
                    break
            if accepted: break
        cv2.destroyAllWindows()

        # --- B) Calibración DADOS ---
        cv2.namedWindow("3. DADOS", cv2.WINDOW_NORMAL)
        print("🎲 PASO 2: Calibrar Dados.")
        umbral = 170
        while True:
            ret, frame = self.cap.read()
            if not ret: continue
            frame = self.corregir_imagen(frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.procesar_tablero_aruco(frame, gray)
            
            blur = cv2.GaussianBlur(gray, (5,5), 0)
            _, bin_img = cv2.threshold(blur, umbral, 255, cv2.THRESH_BINARY)
            
            if self.roi_polygon is not None:
                mask = np.zeros_like(gray)
                cv2.fillPoly(mask, [self.roi_polygon], 255)
                if self.marker_corners:
                    for mc in self.marker_corners:
                        cv2.fillPoly(mask, [mc[0].astype(np.int32)], 0)
                bin_img = cv2.bitwise_and(bin_img, bin_img, mask=mask)
            
            cv2.putText(frame, f"Umbral: {umbral} ('u'/'d'). ESPACIO=Fin", (20, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            cv2.imshow("3. DADOS", bin_img)
            k = cv2.waitKey(1)
            if k == ord('u'): umbral += 2
            elif k == ord('d'): umbral -= 2
            elif k == 32: 
                self.thresh_fondo = umbral
                break
        cv2.destroyAllWindows()

    # ============================================================
    # 3. LÓGICA DE DETECCIÓN MEJORADA
    # ============================================================
    def contar_puntos_blobe(self, gray, contour):
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        x,y,w,h = cv2.boundingRect(contour)
        roi = gray[y:y+h, x:x+w]
        enhanced = self.clahe.apply(roi)
        # Umbral algo agresivo para detectar el negro
        _, dots = cv2.threshold(enhanced, 180, 255, cv2.THRESH_BINARY) 
        
        cnts_d, _ = cv2.findContours(dots, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        validos = 0
        area_dado = cv2.contourArea(contour)
        for c in cnts_d:
            if (area_dado * 0.02) < cv2.contourArea(c) < (area_dado * 0.25):
                validos += 1
        return max(1, min(validos, 6))

    def analisis_robusto_dados(self, num_frames=15):
        lecturas_puntuacion = []
        lecturas_coords = []
        
        print(f"👀 Analizando mesa ({num_frames} frames)...")
        for _ in range(num_frames):
            ret, frame = self.cap.read()
            if not ret: continue
            frame = self.corregir_imagen(frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            disp = frame.copy()
            self.procesar_tablero_aruco(disp, gray)
            
            if self.roi_polygon is None: continue
            
            # Mascara y borrado de Arucos
            mask_roi = np.zeros_like(gray)
            cv2.fillPoly(mask_roi, [self.roi_polygon], 255)
            if self.marker_corners:
                for mc in self.marker_corners:
                    cv2.fillPoly(mask_roi, [mc[0].astype(np.int32)], 0) 
            
            blur = cv2.GaussianBlur(gray, (5,5), 0)
            _, binaria = cv2.threshold(blur, self.thresh_fondo, 255, cv2.THRESH_BINARY)
            binaria = cv2.bitwise_and(binaria, binaria, mask=mask_roi)
            
            cnts, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # --- FILTRADO DE DADOS EN ESTE FRAME ---
            dados_frame = [] # Lista de tuplas: (puntos, rx, ry)
            
            for c in cnts:
                area = cv2.contourArea(c)
                if self.area_min < area < self.area_max:
                    hull = cv2.convexHull(c)
                    if (area / float(cv2.contourArea(hull))) > 0.75:
                        rect = cv2.minAreaRect(c)
                        rx, ry = self.px_to_robot(rect[0][0], rect[0][1])
                        
                        if rx is None: continue

                        # --- CHEQUEO DE DUPLICADOS (DISTANCIA) ---
                        es_duplicado = False
                        for d in dados_frame:
                            dist = math.sqrt((rx - d[1])**2 + (ry - d[2])**2)
                            # Convertimos MIN_DIST_CM a metros para comparar (rx, ry están en metros)
                            if dist < (self.MIN_DIST_DADOS_CM / 100.0):
                                es_duplicado = True
                                break
                        
                        if not es_duplicado:
                            puntos = self.contar_puntos_blobe(gray, c)
                            dados_frame.append((puntos, rx, ry))
                            
                            # Debug visual
                            box = np.int32(cv2.boxPoints(rect))
                            cv2.drawContours(disp, [box], 0, (0,0,255), 2)
                            cv2.putText(disp, str(puntos), (box[0][0], box[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)

            # --- CORRECCION DE DADOS PERDIDOS (SIEMPRE SON 2) ---
            num_detectados = len(dados_frame)
            puntos_frame_total = sum([d[0] for d in dados_frame])
            coords_frame_final = [(d[1], d[2]) for d in dados_frame]
            
            # Regla de seguridad: Si faltan dados, asumimos que son 6
            dados_faltantes = 2 - num_detectados
            if dados_faltantes > 0:
                compensacion = dados_faltantes * 6
                puntos_frame_total += compensacion
                # Visualmente avisamos en pantalla
                cv2.putText(disp, f"FALTAN {dados_faltantes} -> +{compensacion}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            # Solo guardamos si detectamos algo o si asumimos los 6
            if num_detectados > 0 or dados_faltantes == 2:
                lecturas_puntuacion.append(puntos_frame_total)
                lecturas_coords.append(coords_frame_final)
            
            cv2.imshow("Sistema Blackjack", disp)
            cv2.waitKey(30)
            
        if not lecturas_puntuacion: return 0, []
        
        # Moda de la puntuación total calculada
        moda_suma = statistics.mode(lecturas_puntuacion)
        
        # Devolver las coordenadas del frame que dio esa puntuación
        # (Si hay varios frames con la misma puntuación, cogemos el primero que coincida)
        try:
            idx_match = lecturas_puntuacion.index(moda_suma)
            coords_finales = lecturas_coords[idx_match]
            return moda_suma, coords_finales
        except:
            return 0, []

    # ============================================================
    # 4. ROBOT Y GESTOS
    # ============================================================
    def detectar_gesto_robusto(self):
        gestos = []
        for _ in range(10):
            ret, frame = self.cap.read()
            if not ret: break
            frame = self.corregir_imagen(frame)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, self.lower_skin, self.upper_skin)
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)
            
            g = "Ninguno"
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                c = max(cnts, key=cv2.contourArea)
                if cv2.contourArea(c) > 3000:
                    g = self.clasificar_mano(c)
                    cv2.drawContours(frame, [c], -1, (0,255,0), 2)
            
            gestos.append(g)
            self.dibujar_hud(frame)
            cv2.imshow("Sistema Blackjack", frame)
            cv2.waitKey(10)
        
        if not gestos: return "Ninguno"
        return Counter(gestos).most_common(1)[0][0]

    def clasificar_mano(self, c):
        epsilon = 0.02 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        hull = cv2.convexHull(approx, returnPoints=False)
        if hull is None or len(hull) < 3: return "Ninguno"
        try: defects = cv2.convexityDefects(approx, hull)
        except: return "Ninguno"
        dedos = 0
        if defects is not None:
            for i in range(defects.shape[0]):
                s,e,f,d = defects[i,0]
                if (d/256.0) > 15:
                    start = approx[s][0]; end = approx[e][0]; far = approx[f][0]
                    angle = np.arccos(np.dot(start-far, end-far) / (np.linalg.norm(start-far)*np.linalg.norm(end-far)))
                    if angle < np.pi/2: dedos += 1
        
        _,_,w,h = cv2.boundingRect(approx)
        ar = h / float(w)
        if dedos >= 3: return "Opened Palm"
        if dedos == 0: return "Index Finger" if ar > 1.5 else "Thumb Up"
        if dedos == 1: return "Index Finger"
        return "Desconocido"

    def accion_robot_lanzar(self):
        msg = Pose()
        msg.position.z = self.CMD_THROW_FLAG
        msg.orientation.w = 1.0
        self.pub.publish(msg)
        print("🤖 ACCION: Lanzar cubilete.")
        self.esperar_tiempo(4.0)

    def accion_robot_recoger(self, coords):
        if not coords: 
            print("⚠️ No hay coordenadas para recoger.")
            return
        
        for (rx, ry) in coords:
            msg = Pose()
            msg.position.x = rx
            msg.position.y = ry
            msg.orientation.w = 1.0
            self.pub.publish(msg)
            print(f"🤖 ACCION: Recoger en {rx:.2f}, {ry:.2f}")
            self.esperar_tiempo(2.5)
        self.esperar_tiempo(1.0)

    def esperar_tiempo(self, segundos):
        start = time.time()
        while (time.time() - start) < segundos:
            ret, frame = self.cap.read()
            if ret:
                frame = self.corregir_imagen(frame)
                self.dibujar_hud(frame)
                cv2.putText(frame, "ROBOT MOVIENDOSE...", (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                cv2.imshow("Sistema Blackjack", frame)
                cv2.waitKey(1)

    def dibujar_hud(self, img):
        cv2.putText(img, f"JUGADOR: {self.player_score}", (30, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
        cv2.putText(img, f"DEALER: {self.dealer_score}", (30, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        cv2.putText(img, f"ESTADO: {self.state}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

    def ciclo_completo_lanzamiento(self):
        # 1. Esperar confirmación antes de lanzar
        self.esperar_confirmacion_usuario("ESPACIO para LANZAR")
        self.accion_robot_lanzar()
        
        # 2. Esperar confirmación antes de leer/recoger (opcional, útil para debug)
        self.esperar_confirmacion_usuario("ESPACIO para LEER DADOS")
        
        pts, coords = self.analisis_robusto_dados()
        print(f"🎲 Puntos totales (ajustados): {pts}")
        
        self.accion_robot_recoger(coords)
        return pts

    def run(self):
        self.configurar_sistema_completo()
        print("🎮 SISTEMA LISTO. ESPERANDO THUMB UP.")
        while not rospy.is_shutdown():
            ret, frame = self.cap.read()
            if not ret: break
            frame = self.corregir_imagen(frame)
            gesto = self.detectar_gesto_robusto()
            self.dibujar_hud(frame)
            
            if self.state == STATE_IDLE:
                cv2.putText(frame, "Haz THUMB UP para empezar", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                if gesto == "Thumb Up":
                    self.player_score = 0; self.dealer_score = 0
                    self.state = STATE_INIT_DEALER
            elif self.state == STATE_INIT_DEALER:
                pts = self.ciclo_completo_lanzamiento()
                self.dealer_score += pts
                self.state = STATE_INIT_PLAYER
            elif self.state == STATE_INIT_PLAYER:
                pts = self.ciclo_completo_lanzamiento()
                self.player_score += pts
                self.state = STATE_PLAYER_CHOICE
            elif self.state == STATE_PLAYER_CHOICE:
                cv2.putText(frame, "'Index'=Pedir | 'Palm'=Plantarse", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
                if self.player_score >= 21: self.state = STATE_DEALER_LOOP
                elif gesto == "Index Finger":
                    pts = self.ciclo_completo_lanzamiento()
                    self.player_score += pts
                elif gesto == "Opened Palm": self.state = STATE_DEALER_LOOP
            elif self.state == STATE_DEALER_LOOP:
                if self.dealer_score < 17:
                    pts = self.ciclo_completo_lanzamiento()
                    self.dealer_score += pts
                else: self.state = STATE_GAME_OVER
            elif self.state == STATE_GAME_OVER:
                if self.player_score > 21: res = "GANA DEALER (Te pasaste)"
                elif self.dealer_score > 21: res = "GANAS TU (Dealer se paso)"
                elif self.player_score > self.dealer_score: res = "GANAS TU!"
                elif self.dealer_score > self.player_score: res = "GANA DEALER!"
                else: res = "EMPATE"
                cv2.rectangle(frame, (50, 150), (590, 350), (0,0,0), -1)
                cv2.putText(frame, res, (80, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                cv2.putText(frame, "Reiniciar: THUMB UP", (180, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
                if gesto == "Thumb Up": self.state = STATE_IDLE

            cv2.imshow("Sistema Blackjack", frame)
            if cv2.waitKey(1) == 27: break
        
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    BlackjackRobotico().run()