#!/usr/bin/env python3
import rospy
import time
from geometry_msgs.msg import Pose
from control_msgs.msg import GripperCommand
from tf.transformations import quaternion_from_euler
from math import pi

class XYZMover:
    def __init__(self):
        # 1. Iniciar nodo (una sola vez)
        rospy.init_node('emisor_coordenadas_xyz', anonymous=True)
        
        # 2. Publishers
        self.pub_pose = rospy.Publisher('/ur_move_to_pose', Pose, queue_size=10)
        self.pub_gripper = rospy.Publisher('/ur_move_gripper', GripperCommand, queue_size=1)
        
        # 3. Configuración de Orientación "Mirar Abajo"
        # Esto evita que el robot llegue a la posición con la muñeca torcida.
        # Rotación de 180º en X suele orientar la herramienta hacia la mesa en robots UR.
        q = quaternion_from_euler(pi, 0, 0)
        self.target_orientation = q
        
        # Esperar conexión
        rospy.sleep(1)
        rospy.loginfo("🤖 Control XYZ Iniciado. Listo para enviar comandos.")

    def controlar_pinza(self, posicion, fuerza=10.0):
        """Envía comando a la pinza."""
        cmd = GripperCommand()
        cmd.position = posicion
        cmd.max_effort = fuerza
        self.pub_gripper.publish(cmd)
        rospy.sleep(0.5) # Breve espera para asegurar envío

    def abrir_pinza(self):
        rospy.loginfo("👐 Abriendo pinza...")
        self.controlar_pinza(50.0) # Ajusta valor apertura

    def cerrar_pinza(self):
        rospy.loginfo("✊ Cerrando pinza...")
        self.controlar_pinza(16.8) # Ajusta valor cierre

    def mover_a(self, x, y, z):
        """Envía al robot a una coordenada (m) con orientación fija hacia abajo."""
        msg = Pose()
        msg.position.x = x
        msg.position.y = y
        msg.position.z = z
        
        # Aplicamos la orientación calculada (mirando abajo)
        msg.orientation.x = self.target_orientation[0]
        msg.orientation.y = self.target_orientation[1]
        msg.orientation.z = self.target_orientation[2]
        msg.orientation.w = self.target_orientation[3]

        rospy.loginfo(f"✈️ Moviendo a: X={x:.3f}, Y={y:.3f}, Z={z:.3f}")
        self.pub_pose.publish(msg)
        
        # Tiempo prudencial para completar movimiento (bloqueante simple)
        # En un sistema pro, escucharías el 'status' del robot.
        rospy.sleep(6) 

    def run_test_sequence(self):
        """Ejecuta una secuencia de prueba limpia."""
        # Ejemplo: Bajar a coger algo y subir
        
        # 1. Aproximación (Aire)
        self.abrir_pinza()
        self.mover_a(-0.169, 0.1, 0.15) # Z=15cm altura seguridad
        
        # 2. Bajar (Mesa)
        # Asumiendo Z=0.0 como superficie o Z=-0.06 como fondo caja
        self.mover_a(-0.169, 0.1, 0.0)  
        
        # 3. Agarrar
        self.cerrar_pinza()
        rospy.sleep(1)
        
        # 4. Subir
        self.mover_a(-0.169, 0.1, 0.15)

if __name__ == '__main__':
    try:
        robot = XYZMover()
        # Descomenta para ejecutar:
        robot.run_test_sequence()
        
        # O movimientos sueltos:
        # robot.mover_a(-0.1, 0.1, 0.2)
        
    except rospy.ROSInterruptException:
        pass