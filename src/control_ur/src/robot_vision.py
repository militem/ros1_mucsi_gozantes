#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Pose, Quaternion
# Si usas transformaciones de Euler a Quaternion (opcional pero recomendado):
from tf.transformations import quaternion_from_euler
from math import pi

def obtener_coordenadas_camara():
    """
    AQUÍ VA TU LÓGICA DE VISIÓN ARTIFICIAL.
    Por ahora simulamos que la cámara detectó algo.
    Retorna: (x, y) en metros con respecto a la base del robot.
    """
    # Ejemplo: Detectó un objeto en X=0.4m, Y=0.1m
    # En un caso real, aquí llamas a OpenCV / YOLO / etc.
    return 0.4, 0.1

def crear_pose_objetivo(x, y, z_altura):
    """
    Convierte X,Y en un mensaje ROS Pose completo.
    """
    pose_msg = Pose()

    # 1. POSICIÓN (Lo que te da la cámara + altura fija)
    pose_msg.position.x = x
    pose_msg.position.y = y
    pose_msg.position.z = z_altura  # Altura de seguridad o de agarre

    # 2. ORIENTACIÓN (CRÍTICO)
    # El robot necesita saber cómo rotar la muñeca.
    # Generalmente queremos que la pinza apunte hacia abajo (perpendicular a la mesa).
    
    # Opción A: Usar valores fijos si ya conoces el cuaternión de "mirar abajo"
    # (Valores típicos para UR mirando abajo)
    # pose_msg.orientation.x = 0.0
    # pose_msg.orientation.y = 1.0
    # pose_msg.orientation.z = 0.0
    # pose_msg.orientation.w = 0.0
    
    # Opción B: Calcular desde ángulos de Euler (Roll, Pitch, Yaw)
    # Rotar 180 grados (pi) en el eje Y suele hacer que apunte abajo.
    q = quaternion_from_euler(pi, 0, 0) # Roll=180°, Pitch=0, Yaw=0
    pose_msg.orientation.x = q[0]
    pose_msg.orientation.y = q[1]
    pose_msg.orientation.z = q[2]
    pose_msg.orientation.w = q[3]

    return pose_msg

def main():
    # 1. Iniciar este nodo
    rospy.init_node('nodo_vision_camara', anonymous=True)

    # 2. Crear el publicador que hablará con tu script ControlRobot
    # Nota: Debe coincidir con el tópico que definiste: '/ur_move_to_pose'
    pub_robot = rospy.Publisher('/ur_move_to_pose', Pose, queue_size=10)

    rate = rospy.Rate(1) # Frecuencia de chequeo (1 Hz)

    rospy.loginfo("Cámara iniciada. Esperando detección...")

    cuadrado = [
        (0.4, 0.1),  # Esquina 1
        (0.4, -0.1), # Esquina 2
        (0.6, -0.1), # Esquina 3
        (0.6, 0.1)   # Esquina 4
    ]
    
    indice = 0

    while not rospy.is_shutdown():
        # Obtener la siguiente esquina
        x_target, y_target = cuadrado[indice]
        
        # Crear mensaje
        mensaje_pose = crear_pose_objetivo(x_target, y_target, 0.3)
        
        rospy.loginfo(f"Moviendo a esquina {indice+1}: X={x_target}, Y={y_target}")
        pub_robot.publish(mensaje_pose)

        # Avanzar al siguiente punto (circularmente)
        indice = (indice + 1) % 4
        
        # Esperar 5 segundos para que le de tiempo a llegar
        rospy.sleep(5)

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass