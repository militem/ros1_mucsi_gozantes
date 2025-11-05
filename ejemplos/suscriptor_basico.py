#!/usr/bin/env python
import rospy
from std_msgs.msg import String


# Función de callback que se ejecuta cada vez que se recibe un mensaje
def callback(data: String) -> None:    
    print(f"He recibido: {data.data}")


# Inicialización del nodo suscriptor
rospy.init_node('suscriptor_basico', anonymous=True)

# Creación de un suscriptor que escucha Strings en el topic 'primer_topic' y que ejecuta la función de callback
rospy.Subscriber('primer_topic', String, callback)

# Mantener el nodo en ejecución hasta que se detenga el maestro
rospy.spin()