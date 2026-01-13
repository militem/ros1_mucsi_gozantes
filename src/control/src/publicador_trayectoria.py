#!/usr/bin/env python3
import rospy
import time
from sensor_msgs.msg import JointState
from control_msgs.msg import GripperCommand

class SecuenciaVertido:
    def __init__(self):
        rospy.init_node('secuencia_vertido_con_garra', anonymous=True)
        
        # --- PUBLISHERS ---
        self.pub_arm = rospy.Publisher('/ur_move_to_joints', JointState, queue_size=1)
        self.pub_gripper = rospy.Publisher('/ur_move_gripper', GripperCommand, queue_size=1)
        
        # --- DEFINICIÓN DE POSICIONES (RADIANES) ---
        # Pos 1: Suelo (Base)
        self.pos_suelo = [-1.07, -2.56, -2.30, 1.24, 1.14, 0.14]

        # Pos 2: Elevada (Muñeca fija respecto a suelo)
        # Usamos pos_suelo[5] para mantener la rotación de la muñeca estable
        self.pos_elevada = [-1.07, -2.64, -1.37, 0.72, 2.6, self.pos_suelo[5]]

        # Pos 3: Giro (Verter)
        # Copiamos elevada y rotamos la última articulación 180º (pi)
        self.pos_giro = list(self.pos_elevada)
        self.pos_giro[5] = self.pos_elevada[5] + 3.14 

        rospy.sleep(1)
        rospy.loginfo("✅ Nodo de secuencia iniciado.")

    def crear_mensaje_joints(self, valores_articulaciones):
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 
                    'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        msg.position = valores_articulaciones
        msg.velocity = []
        msg.effort = []
        return msg

    def mover_pinza(self, posicion, fuerza=10.0):
        cmd = GripperCommand()
        cmd.position = posicion
        cmd.max_effort = fuerza
        self.pub_gripper.publish(cmd)
        rospy.sleep(1) # Espera técnica para la acción mecánica

    def ejecutar_secuencia(self):
        rospy.loginfo("--- INICIANDO SECUENCIA DE VERTIDO ---")

        # 1. Preparar Garra (Agarrar objeto)
        rospy.loginfo("✊ Cerrando garra...")
        self.mover_pinza(42.3) 
        rospy.sleep(1)

        # 2. Definir secuencia de movimientos
        trayectoria = [
            ("Suelo", self.pos_suelo),
            ("Subir", self.pos_elevada),
            ("Verter", self.pos_giro),
            ("Bajar", self.pos_suelo)
        ]

        # 3. Ejecutar bucle
        for nombre, articulaciones in trayectoria:
            if rospy.is_shutdown(): break
            
            rospy.loginfo(f"🦾 Ejecutando paso: {nombre}")
            msg = self.crear_mensaje_joints(articulaciones)
            self.pub_arm.publish(msg)
            
            # Tiempo de espera para completar el movimiento
            # (Ajustar según velocidad real del robot)
            rospy.sleep(6)

        # 4. Soltar al finalizar
        rospy.loginfo("👐 Abriendo garra (Finalizando)...")
        self.mover_pinza(50.0)
        rospy.sleep(1)
        
        rospy.loginfo("✨ ¡Proceso completado!")

if __name__ == '__main__':
    try:
        app = SecuenciaVertido()
        app.ejecutar_secuencia()
    except rospy.ROSInterruptException:
        pass