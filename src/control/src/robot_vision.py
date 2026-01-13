#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Pose
from tf.transformations import quaternion_from_euler
from math import pi

class RobotTrajectoryTester:
    def __init__(self):
        rospy.init_node('nodo_vision_tester', anonymous=True)
        self.pub_robot = rospy.Publisher('/ur_move_to_pose', Pose, queue_size=10)
        
        # --- CONFIGURACIÓN DE PRUEBA ---
        self.center_x = 0.5   # Centro del cuadrado en X (metros)
        self.center_y = 0.0   # Centro del cuadrado en Y (metros)
        self.size = 0.2       # Tamaño del lado del cuadrado (metros)
        self.z_height = 0.25  # Altura de seguridad
        
        # Pre-calcular orientación (Mirar hacia abajo)
        # Roll=180, Pitch=0, Yaw=0 es típico para UR mirando a mesa
        q = quaternion_from_euler(pi, 0, 0)
        self.orientation_q = q
        
        rospy.loginfo("🤖 Tester Iniciado. Publicando en /ur_move_to_pose")
        rospy.loginfo(f"   Cuaternión Objetivo: {q}")

    def get_square_points(self):
        """Genera 4 puntos alrededor del centro definido."""
        half = self.size / 2.0
        return [
            (self.center_x + half, self.center_y + half), # P1
            (self.center_x + half, self.center_y - half), # P2
            (self.center_x - half, self.center_y - half), # P3
            (self.center_x - half, self.center_y + half)  # P4
        ]

    def create_pose_msg(self, x, y):
        pose_msg = Pose()
        
        # 1. Posición
        pose_msg.position.x = x
        pose_msg.position.y = y
        pose_msg.position.z = self.z_height
        
        # 2. Orientación (Usando el cálculo Euler -> Cuaternión)
        pose_msg.orientation.x = self.orientation_q[0]
        pose_msg.orientation.y = self.orientation_q[1]
        pose_msg.orientation.z = self.orientation_q[2]
        pose_msg.orientation.w = self.orientation_q[3]
        
        return pose_msg

    def run(self):
        points = self.get_square_points()
        idx = 0
        
        # Dar tiempo a ROS para conectar el publisher
        rospy.sleep(1) 
        
        while not rospy.is_shutdown():
            target_x, target_y = points[idx]
            
            msg = self.create_pose_msg(target_x, target_y)
            self.pub_robot.publish(msg)
            
            rospy.loginfo(f"📍 Punto {idx+1}/4 -> X: {target_x:.2f}, Y: {target_y:.2f}, Z: {self.z_height}")
            
            idx = (idx + 1) % 4
            
            # Esperar antes del siguiente movimiento
            rospy.sleep(5)

if __name__ == '__main__':
    try:
        tester = RobotTrajectoryTester()
        tester.run()
    except rospy.ROSInterruptException:
        pass