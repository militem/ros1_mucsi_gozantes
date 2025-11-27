#!/usr/bin/python3
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from copy import deepcopy
import cv2

class NodoCamara:
    def __init__(self) -> None:
        rospy.init_node('nodo_camara', anonymous=True)
        self.bridge = CvBridge()
        rospy.Subscriber('/usb_cam/image_raw', Image, self.callback)
        rospy.loginfo(f"Esperando imagenes...")
        
    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            ...
            ...
            ...
            cv2.imshow("Vista camara del robot", cv_image)
            cv2.waitKey(1)
        except CvBridgeError as e:
            rospy.logerr(f"Error al conventir la imagen {e}")
    
if __name__=="__main__":
    nodo = NodoCamara()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Cerrando nodo")
        cv2.destroyAllWindows()