#!/usr/bin/python3

import sys
import rospy
import copy
from typing import List
from moveit_commander import MoveGroupCommander, RobotCommander, roscpp_initialize, PlanningSceneInterface
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, PoseStamped

# --- IMPORTANTE: Añadimos el tipo de mensaje que usa tu visión ---
from std_msgs.msg import Float64MultiArray 
# ---------------------------------------------------------------

from control_msgs.msg import GripperCommandActionGoal, GripperCommand, GripperCommandGoal, GripperCommandAction
from actionlib import SimpleActionClient

TOPICO_DIRECTO = "/gripper_controller/gripper_cmd"

class ControlRobot:
    
    def __init__(self) -> None:
        roscpp_initialize(sys.argv)
        rospy.init_node("control_robot_bridge", anonymous=True)
        
        self.robot = RobotCommander()
        self.scene = PlanningSceneInterface()
        self.group_name = "robot" 
        self.move_group = MoveGroupCommander(self.group_name)
        self.planning_frame = self.move_group.get_planning_frame()
        self.move_group.set_planning_time(20.0)
        self.move_group.set_goal_position_tolerance(0.001)
        self.move_group.set_goal_orientation_tolerance(0.01)
        self.move_group.set_max_velocity_scaling_factor(1.0)
        self.move_group.set_max_acceleration_scaling_factor(1.0)
        
        self.pose_origen = None 

        # Pinza
        self.gripper_pub = rospy.Publisher(TOPICO_DIRECTO, GripperCommandActionGoal, queue_size=1)
        self.gripper_action_client = SimpleActionClient("rg2_action_server", GripperCommandAction)
        self.pinza_activa = True 

        self.añadir_suelo()

        self.target_joint_pub = rospy.Publisher('/ur_target_positions', JointState, queue_size=10)
        
        # --- CAMBIO CLAVE AQUÍ ---
        # Ahora nos suscribimos esperando un Float64MultiArray (lo que envía tu visión)
        self.vision_sub = rospy.Subscriber('/ur_move_to_pose', Float64MultiArray, self.callback_vision_array)
        
        self.zero_sub = rospy.Subscriber('/ur_set_zero', Pose, self.callback_set_zero)
        self.gripper_sub = rospy.Subscriber('/ur_move_gripper', GripperCommand, self.gripper_callback)
        
        rospy.loginfo("✅ Nodo Bridge LISTO. Esperando arrays de la cámara...")

    def callback_set_zero(self, msg: Pose):
        self.pose_origen = self.pose_actual()
        rospy.loginfo("✅ Nuevo ORIGEN (ArUco) establecido.")

    def callback_vision_array(self, msg: Float64MultiArray):
        """
        Recibe [x, y, z, rx, ry, rz] desde la visión.
        """
        # 1. Seguridad: Si no hay origen, usamos donde estemos ahora
        if self.pose_origen is None:
            rospy.logwarn("⚠️ Origen no definido. Usando posición actual como referencia.")
            self.pose_origen = self.pose_actual()

        # 2. Extraer datos del array
        # Tu visión manda: [r_xm, r_ym, ROBOT_Z_METERS, ...]
        x_rel = msg.data[0]
        y_rel = msg.data[1]
        z_rel = msg.data[2] # Altura que definiste en vision (0.25)

        rospy.loginfo(f"📷 Recibido de Visión: X={x_rel:.3f}, Y={y_rel:.3f}, Z={z_rel:.3f}")

        # 3. Calcular destino
        objetivo = copy.deepcopy(self.pose_origen)
        
        # Sumamos al origen del ArUco
        objetivo.position.x = self.pose_origen.position.x + x_rel
        objetivo.position.y = self.pose_origen.position.y + y_rel
        
        # NOTA: ¿Quieres sumar Z o ir a esa Z absoluta respecto al origen?
        # Normalmente en pick&place queremos ir a una altura fija sobre la mesa.
        # Si tu origen está en la mesa, Z=0.25 significa "25cm sobre la mesa".
        objetivo.position.z = self.pose_origen.position.z + z_rel 

        # Mantenemos la orientación con la que grabaste el cero (pinza mirando abajo)
        objetivo.orientation = self.pose_origen.orientation

        # 4. Ejecutar
        self.mover_trayectoria([objetivo])

    def mover_trayectoria(self, poses: List[Pose], wait: bool = True) -> bool:
        poses_aux = copy.deepcopy(poses)
        poses_aux.insert(0, self.pose_actual())
            
        (plan, fraction) = self.move_group.compute_cartesian_path(poses_aux, 0.01)

        if fraction < 0.5:
            rospy.logwarn(f"⚠️ Ruta imposible ({fraction*100}%).")
            return False
        
        # Suavizado de trayectoria (Retime)
        try:
            plan = self.move_group.retime_trajectory(
                self.robot.get_current_state(), 
                plan, 
                velocity_scaling_factor=0.5,
                acceleration_scaling_factor=0.5
            )
        except:
            pass

        return self.move_group.execute(plan, wait=wait)

    # --- RESTO DE FUNCIONES IGUALES ---
    def pose_actual(self) -> Pose:
        return self.move_group.get_current_pose().pose

    def gripper_callback(self, msg: GripperCommand):
        self.mover_pinza_blind(msg.position, msg.max_effort)

    def mover_pinza_blind(self, anchura, fuerza):
        goal_msg = GripperCommandActionGoal()
        goal_msg.goal.command.position = anchura
        goal_msg.goal.command.max_effort = fuerza
        self.gripper_pub.publish(goal_msg)
        rospy.sleep(0.5)

    def añadir_suelo(self):
        pose_suelo = Pose()
        pose_suelo.position.z = -0.026
        box_pose = PoseStamped()
        box_pose.header.frame_id = self.planning_frame
        box_pose.pose = pose_suelo
        self.scene.add_box("suelo", box_pose, size=(2, 2, 0.05))

if __name__ == '__main__':
    try:
        ControlRobot()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass