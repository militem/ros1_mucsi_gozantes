import cv2

def probar_camaras():
    # Basado en tu ls, tus indices reales son 0 y 2
    indices_a_probar = [0, 2, 4] 
    
    for index in indices_a_probar:
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            print(f"❌ Cámara {index}: No detectada o ocupada.")
            continue
        
        ret, frame = cap.read()
        if ret:
            print(f"✅ Cámara {index}: FUNCIONA. Mira la ventana emergente.")
            cv2.putText(frame, f"CAMARA INDICE {index}", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Mostramos la camara hasta que presiones una tecla
            while True:
                cv2.imshow(f"Prueba Camara {index}", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                ret, frame = cap.read()
                if not ret: break
            
            cap.release()
            cv2.destroyWindow(f"Prueba Camara {index}")
        else:
            print(f"⚠️ Cámara {index}: Detectada pero no da imagen.")

if __name__ == "__main__":
    print("Presiona 'q' en la ventana de imagen para pasar a la siguiente cámara.")
    probar_camaras()