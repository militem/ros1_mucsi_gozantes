#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Pose
from control_msgs.msg import GripperCommand

# Función auxiliar para crear el mensaje de la pinza más fácilmente
def crear_comando_pinza(posicion, fuerza):
    cmd = GripperCommand()
    cmd.position = posicion
    cmd.max_effort = fuerza
    return cmd

pub_gripper = rospy.Publisher('/ur_move_gripper', GripperCommand, queue_size=1)

def enviar_coordenada(x_m, y_m, z_m):
    """
    Envía coordenadas relativas al origen (ArUco).
    Las unidades son METROS.
    """
    rospy.init_node('emisor_coordenadas', anonymous=True)
    pub = rospy.Publisher('/ur_move_to_pose', Pose, queue_size=10)
    
    # Esperamos un instante a que el publisher conecte con el bridge
    rospy.sleep(0.5)

    msg = Pose()
    msg.position.x = x_m
    msg.position.y = y_m
    msg.position.z = z_m
    
    # La orientación se ignora en el bridge, pero rellenamos w=1 por protocolo
    msg.orientation.w = 1.0

    rospy.loginfo(f" Enviando orden: X={x_m}, Y={y_m}, Z={z_m}")
    pub.publish(msg)
    
    # Damos tiempo a que el mensaje salga
    rospy.sleep(0.5)

if __name__ == '__main__':
    try:
        # --- EJEMPLOS DE USO ---
        # -0.06 en z es el nivel que estan los dados en la caja
        # 1. Subir 5 cm sobre el ArUco
        #enviar_coordenada(-0.1, 0.1, -0.06)
        #enviar_coordenada(-0.1, 0.1, 0.0)
        #rospy.sleep(8)
        enviar_coordenada(-0.169, 0.1, 0.0)
        rospy.sleep(8)
        
        """
        enviar_coordenada(0.0, 0.0, 0.0)
        rospy.sleep(8)
        
        enviar_coordenada(-0.1, 0.1, 0.0)
        rospy.sleep(8)

        enviar_coordenada(-0.1, 0.1, -0.06)
        rospy.sleep(8)

        cmd_abrir = crear_comando_pinza(25, 8.0) 
        pub_gripper.publish(cmd_abrir) 

        rospy.sleep(2)
        cmd_cerrar = crear_comando_pinza(16.8, 8.0) 
        pub_gripper.publish(cmd_cerrar) 

        enviar_coordenada(0.0, 0.1, 0.0)
        rospy.sleep(8)

        cmd_abrir = crear_comando_pinza(25, 8.0) 
        pub_gripper.publish(cmd_abrir) 
        rospy.sleep(2)

        enviar_coordenada(0.0, 0.0, 0.0)
        rospy.sleep(8)
        """
        
        # 2. Moverse 10 cm a la derecha (Y) y 5 cm arriba
        # enviar_coordenada(0.0, 0.1, 0.05)

        # 3. Volver a tocar el ArUco
        # enviar_coordenada(0.0, 0.0, 0.0)
        
    except rospy.ROSInterruptException:
        pass