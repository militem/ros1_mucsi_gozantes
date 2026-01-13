#!/usr/bin/python3

import sys
import rospy
import copy
from typing import List
from moveit_commander import MoveGroupCommander, RobotCommander, roscpp_initialize, PlanningSceneInterface
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, PoseStamped

import math
# IMPORTANTE: Usamos el mensaje "ActionGoal" para publicar directamente
# sin necesitar cliente de acción.
from control_msgs.msg import GripperCommandActionGoal, GripperCommand, GripperCommandGoal, GripperCommandAction
from actionlib import SimpleActionClient

# Configuración detectada anteriormente
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
        self.move_group.set_goal_position_tolerance(0.001)   # 1 mm
        self.move_group.set_goal_orientation_tolerance(0.01)
        self.move_group.set_num_planning_attempts(10)
        self.move_group.set_max_velocity_scaling_factor(1.0)
        self.move_group.set_max_acceleration_scaling_factor(1.0)
        
        self.pose_origen = None 

        self.gripper_pub = rospy.Publisher(
            TOPICO_DIRECTO, 
            GripperCommandActionGoal, 
            queue_size=1
        )
        self.gripper_action_client = SimpleActionClient("rg2_action_server", GripperCommandAction)
        rospy.loginfo(f" Publicando en {TOPICO_DIRECTO}")
        self.pinza_activa = True 

        self.añadir_suelo()

        # Publicadores y Suscriptores del Brazo
        self.target_joint_pub = rospy.Publisher('/ur_target_positions', JointState, queue_size=10)
        self.target_pose_pub = rospy.Publisher('/ur_target_pose', PoseStamped, queue_size=10)
        
        self.joint_goal_sub = rospy.Subscriber('/ur_move_to_joints', JointState, self.joint_goal_callback)
        
        # Suscriptor para movimiento relativo
        self.pose_goal_sub = rospy.Subscriber('/ur_move_to_pose', Pose, self.pose_goal_callback)
        
        # Suscriptor para movimiento del cubilete
        self.pose_goal_sub = rospy.Subscriber('/ur_move_cubilete', Pose, self.pose_cubilete_callback)
        
        # Suscriptor para establecer el CERO (origen)
        self.zero_sub = rospy.Subscriber('/ur_set_zero', Pose, self.callback_set_zero)
        
        # --- SUSCRIPTOR DE LA PINZA ---
        self.gripper_sub = rospy.Subscriber('/ur_move_gripper', GripperCommand, self.gripper_callback)
        
        rospy.loginfo(" Nodo Bridge LISTO. Usa /ur_set_zero para marcar origen.")

    def joint_goal_callback(self, msg: JointState):
        self.target_joint_pub.publish(msg)
        self.move_group.go(msg.position, wait=True)

    def callback_set_zero(self, msg: Pose):
        """
        Establece la posicion actual del robot como el punto (0,0,0)
        """
        self.pose_origen = self.pose_actual()
        rospy.loginfo(" Nuevo ORIGEN establecido.")
        rospy.loginfo(f" Posicion absoluta: {self.pose_origen.position}")


    def pose_cubilete_callback(self, msg: Pose):
        """
        Secuencia Final con zona de lanzamiento FIJA por ángulos:
        1. Mueve servos a posición de agarre (Tus valores fijos).
        2. Baja, agarra y sube.
        3. Se mueve a la zona de lanzamiento (Tus nuevos valores fijos).
        4. Gira la última articulación 180º para vaciar.
        """
        rospy.loginfo("INICIANDO SECUENCIA CUBILETE (Posiciones Fijas)")

        # --- PASO 1: ABRIR PINZA ---
        self.mover_pinza(60.0, 10.0)
        
        # --- PASO 2: POSICIÓN DE ALINEACIÓN (Valores de agarre) ---
        grados_agarre = [-30.81, -63.61, -138.44, -46.92, 88.58, 19.08]
        rads_agarre = [math.radians(g) for g in grados_agarre]

        rospy.loginfo("Moviendo a posición de agarre...")
        self.move_group.go(rads_agarre, wait=True)
        self.move_group.stop()
        
        # --- PASO 3: BAJAR Y AGARRAR ---
        pose_referencia = self.pose_actual()
        
        pose_bajar = copy.deepcopy(pose_referencia)
        pose_bajar.position.z -= 0.05 # Bajar 5cm

        rospy.loginfo("Bajando...")
        self.mover_trayectoria([pose_bajar])

        rospy.loginfo("Cerrando pinza...")
        self.mover_pinza(41.0, 10.0)
        rospy.sleep(0.5)

        self.mover_trayectoria([pose_referencia])

        # --- PASO 5: IR A LANZAMIENTO (NUEVOS VALORES) ---
        # Valores proporcionados: -19.81, -67.03, -127.78, -26.57, 93.86, 0.10
        grados_lanzamiento = [-8.18, -51.00, -147.97, 13.48, 96.42, 0.11]
        rads_lanzamiento = [math.radians(g) for g in grados_lanzamiento]

        rospy.loginfo("Yendo a posición de lanzamiento fija...")
        self.move_group.go(rads_lanzamiento, wait=True)
        self.move_group.stop()

        # --- PASO 6: GIRO FINAL 180 GRADOS ---
        rospy.loginfo("Girando 180 grados (Wrist 3)...")
        
        # Obtenemos dónde estamos (que será la posición de lanzamiento)
        joints_now = self.move_group.get_current_joint_values()
        
        # Giramos la última articulación
        joints_now[5] += math.pi 
        
        self.move_group.go(joints_now, wait=True)
        self.move_group.stop()

        rospy.loginfo("DADOS LANZADOS")
        
        rospy.sleep(1.0)
        joints_now[5] -= math.pi
        self.move_group.go(joints_now, wait=True)

        pose_entrega = copy.deepcopy(self.pose_origen)
        pose_entrega.position.x += 0.0
        pose_entrega.position.y += 0.12
        pose_entrega.position.z += msg.position.z
        pose_entrega.orientation = self.pose_origen.orientation

        rospy.loginfo("Yendo a zona de entrega (0.0, 0.2)")
        self.mover_trayectoria([pose_entrega])

        rospy.loginfo("Moviendo a posición de agarre...")
        self.move_group.go(rads_agarre, wait=True)
        self.move_group.stop()

        rospy.loginfo("Abriendo pinza a 25.0")
        self.mover_pinza(50.0, 10.0)
        rospy.sleep(0.5)

        rospy.loginfo("Yendo a zona de entrega (0.0, 0.2)")
        self.mover_trayectoria([pose_entrega])


    def pose_goal_callback(self, msg: Pose):
        """
        Secuencia Completa:
        1. Ir a coordenada de visión (Hover)
        2. Bajar Z - 0.06
        3. Abrir Pinza (25)
        4. Cerrar Pinza (16.8, F=8)
        5. Subir (Seguridad)
        6. Ir a zona de entrega (0.0, 0.2, 0.0)
        """

        if self.pose_origen is None:
            rospy.logwarn("Origen no definido. Usando posición actual.")
            self.pose_origen = self.pose_actual()

        rospy.loginfo("INICIANDO SECUENCIA DE PICK & PLACE")


        pose_hover = copy.deepcopy(self.pose_origen)
        pose_hover.position.x += msg.position.x
        pose_hover.position.y += msg.position.y
        pose_hover.position.z += msg.position.z 
        pose_hover.orientation = self.pose_origen.orientation

        pose_agarre = copy.deepcopy(pose_hover)
        pose_agarre.position.z -= 0.05

        pose_entrega = copy.deepcopy(self.pose_origen)
        pose_entrega.position.x += 0.0
        pose_entrega.position.y += 0.12
        pose_entrega.position.z += msg.position.z 
        pose_entrega.orientation = self.pose_origen.orientation

        # --- PASO 2: EJECUTAR SECUENCIA ---
        rospy.loginfo("Abriendo pinza a 25.0")
        self.mover_pinza(50.0, 10.0)
        rospy.sleep(0.5) 

        rospy.loginfo(f"Yendo a vertical: X={msg.position.x:.2f}, Y={msg.position.y:.2f}")
        exito = self.mover_trayectoria([pose_hover])
        if not exito: return

        rospy.loginfo("Bajando Z -0.06m...")
        self.mover_trayectoria([pose_agarre])

        rospy.loginfo("Cerrando pinza (Agarre)")
        self.mover_pinza(16.2, 8.0)
        rospy.sleep(1.0)

        rospy.loginfo("Subiendo...")
        self.mover_trayectoria([pose_hover])

        rospy.loginfo("Yendo a zona de entrega (0.0, 0.2)")
        self.mover_trayectoria([pose_entrega])

        rospy.loginfo("Abriendo pinza a 50.0")
        self.mover_pinza(50.0, 10.0)
        rospy.sleep(0.5)

        
        rospy.loginfo(" SECUENCIA COMPLETADA")

    def gripper_callback(self, msg: GripperCommand):
        """
        Recibe el mensaje desde 'secuencia_vertido.py' y lo reenvía a la pinza.
        """
        anchura = msg.position
        fuerza = msg.max_effort
        
        rospy.loginfo(f" Ejecutando pinza -> Pos: {anchura}, F: {fuerza}")
        self.mover_pinza(anchura, fuerza)

    def mover_pinza_blind(self, anchura: float, fuerza: float):
        """
        Construye el mensaje complejo manualmente y lo lanza.
        """
        goal_msg = GripperCommandActionGoal()
        goal_msg.header.stamp = rospy.Time.now()
        goal_msg.goal_id.stamp = rospy.Time.now()
        goal_msg.goal_id.id = "manual_command_" + str(rospy.Time.now().to_sec())
        
        goal_msg.goal.command.position = anchura
        goal_msg.goal.command.max_effort = fuerza
        
        self.gripper_pub.publish(goal_msg)
        rospy.sleep(0.5)

    def añadir_suelo(self) -> None:
        pose_suelo = Pose()
        pose_suelo.position.z = -0.026
        box_pose = PoseStamped()
        box_pose.header.frame_id = self.planning_frame
        box_pose.pose = pose_suelo
        self.scene.add_box("suelo", box_pose, size=(2, 2, 0.05))

    def mover_pinza(self, anchura_dedos: float, fuerza: float) -> bool:
        goal = GripperCommandGoal()
        goal.command.position = anchura_dedos
        goal.command.max_effort = fuerza
        self.gripper_action_client.send_goal(goal)
        self.gripper_action_client.wait_for_result()
        result = self.gripper_action_client.get_result()
        
        return result.reached_goal
    
    def mover_a_pose(self, pose_goal: Pose, wait: bool=True) -> bool:
        self.move_group.set_pose_target(pose_goal)
        result = self.move_group.go(wait=wait)
        self.move_group.stop() 
        self.move_group.clear_pose_targets()
        return result
    
    def mover_trayectoria(self, poses: List[Pose], wait: bool = True) -> bool:
        poses_aux = copy.deepcopy(poses)
        poses_aux.insert(0, self.pose_actual())
            
        (plan, fraction) = self.move_group.compute_cartesian_path(poses_aux, 0.01)

        if fraction < 0.5:
            rospy.logwarn(f"Trayectoria incompleta ({fraction*100}%). Abortando.")
            return False
        
        try:
            plan = self.move_group.retime_trajectory(
                self.robot.get_current_state(), 
                plan, 
                velocity_scaling_factor=0.5,
                acceleration_scaling_factor=0.5
            )
        except Exception as e:
            rospy.logwarn(f"No se pudo aplicar retime (quizás versión antigua de MoveIt): {e}")

        rospy.loginfo(f"Ejecutando trayectoria al {fraction*100}%...")
        return self.move_group.execute(plan, wait=wait)
    
    def pose_actual(self) -> Pose:
        return self.move_group.get_current_pose().pose

if __name__ == '__main__':
    try:
        control = ControlRobot()
        
        # Ejemplo inicial (opcional, se puede comentar)
        # control.mover_pinza(50.0,40.0)
        # pose_actual = control.pose_actual()
        # poses = []
        # pose_actual.position.x += 0.1
        # poses.append(copy.deepcopy(pose_actual))
        # control.mover_trayectoria(poses)
        
        rospy.spin()
    except rospy.ROSInterruptException:
        pass