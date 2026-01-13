#!/usr/bin/env python3
import rospy
import time
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32  # <-- Necesario para la pinza
from control_msgs.msg import GripperCommand

# Función auxiliar para crear el mensaje de la pinza más fácilmente
def crear_comando_pinza(posicion, fuerza):
    cmd = GripperCommand()
    cmd.position = posicion
    cmd.max_effort = fuerza
    return cmd

def crear_mensaje_joints(valores_articulaciones):
    msg = JointState()
    msg.header.stamp = rospy.Time.now()
    msg.name = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 
                'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
    msg.position = valores_articulaciones
    msg.velocity = []
    msg.effort = []
    return msg

def main():
    rospy.init_node('secuencia_vertido_con_garra', anonymous=True)
    
    # 1. Publicador para mover el BRAZO
    pub_arm = rospy.Publisher('/ur_move_to_joints', JointState, queue_size=1)
    
    # 2. Publicador para mover la PINZA (Nuevo)
    pub_gripper = rospy.Publisher('/ur_move_gripper', GripperCommand, queue_size=1)
    
    # Esperamos a que ROS conecte
    rospy.sleep(1)

    # --- DEFINICIÓN DE POSICIONES (EN RADIANES) ---

    # Posición 1: SUELO (Base)
    # Tu valor original: [..., 0.14]
    pos_suelo = [-1.07, -2.56, -2.30, 1.24, 1.14, 0.14]

    # Posición 2: ELEVADA
    # IMPORTANTE: He cambiado el último valor (6.22) por pos_suelo[5] (0.14).
    # Esto garantiza que la muñeca NO gire nada mientras sube.
    pos_elevada = [-1.07, -2.64, -1.37, 0.72, 2.6, pos_suelo[5]]

    # Posición 3: GIRO (Verter)
    # Copiamos la elevada y sumamos PI (3.14) a la última articulación
    pos_giro = list(pos_elevada)
    pos_giro[5] = pos_elevada[5] + 3.14 # Gira 180 grados desde donde esté

    # Posición 4: FINAL (Volver al suelo)
    pos_final = list(pos_suelo)

    # Lista de la secuencia del brazo
    lista_movimientos = [pos_suelo, pos_elevada, pos_giro, pos_suelo]

    rospy.loginfo("--- INICIANDO SECUENCIA DE VERTIDO ---")

    # 1. Aseguramos que la garra empiece CERRADA (opcional, ajusta si quieres)
    rospy.loginfo("Cerrando garra para agarrar objeto...")
    cmd_cerrar = crear_comando_pinza(42.3, 10.0) 
    pub_gripper.publish(cmd_cerrar)
    rospy.sleep(2)

    # 2. Bucle de movimiento del brazo
    for i, articulaciones in enumerate(lista_movimientos):
        if rospy.is_shutdown():
            break

        nombres_pasos = ["Suelo", "Subir (Muñeca fija)", "Girar/Verter", "Bajar a inicio"]
        rospy.loginfo(f"Ejecutando paso {i+1}: {nombres_pasos[i]}")
        
        # Enviar orden al brazo
        mensaje = crear_mensaje_joints(articulaciones)
        pub_arm.publish(mensaje)
        
        # Esperar a que llegue (ajusta el tiempo si va muy lento/rápido)
        rospy.sleep(8)

    # 3. AL FINALIZAR: ABRIR LA GARRA
    rospy.loginfo("Secuencia de movimiento terminada. Abriendo garra...")
    cmd_abrir = crear_comando_pinza(50, 10.0) 
    pub_gripper.publish(cmd_abrir)
    rospy.sleep(2)

    rospy.loginfo("¡Proceso completado!")

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass