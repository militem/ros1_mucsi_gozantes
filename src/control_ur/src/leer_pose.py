#!/usr/bin/env python3
import sys
import rospy
from moveit_commander import MoveGroupCommander, roscpp_initialize

def obtener_datos():
    roscpp_initialize(sys.argv)
    rospy.init_node("lector_de_pose", anonymous=True)

    group_name = "robot"  # Asegúrate que coincide con tu config
    move_group = MoveGroupCommander(group_name)

    # Obtener la pose actual (Posición + Orientación)
    pose_actual = move_group.get_current_pose().pose

    print("\n" + "="*40)
    print(" DATOS DE LA POSICIÓN ACTUAL (Copiar esto)")
    print("="*40)
    print(f"POSICIÓN (XYZ):")
    print(f"  x: {pose_actual.position.x}")
    print(f"  y: {pose_actual.position.y}")
    print(f"  z: {pose_actual.position.z}")
    print("-" * 20)
    print(f"ORIENTACIÓN (Quaternion) -> ¡Usa esto en tu script!")
    print(f"  x: {pose_actual.orientation.x}")
    print(f"  y: {pose_actual.orientation.y}")
    print(f"  z: {pose_actual.orientation.z}")
    print(f"  w: {pose_actual.orientation.w}")
    print("="*40 + "\n")

if __name__ == '__main__':
    obtener_datos()