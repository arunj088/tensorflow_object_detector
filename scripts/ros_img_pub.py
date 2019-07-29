#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge, CvBridgeError
import tensorflow as tf


if __name__ == '__main__':
    rospy.init_node('image_publisher')
    f_rate = 30
    rate = rospy.Rate(f_rate)
    bridge = CvBridge()
    pub = rospy.Publisher("test_image", Image, queue_size=1)
    while not rospy.is_shutdown():
        img = cv2.imread('/home/aj/Curved-Lane-Lines/test_images/prescan3.PNG')
        img = cv2.resize(img,(640,480))
        try:
            image_out = bridge.cv2_to_imgmsg(img, "bgr8")
        except CvBridgeError as e:
            print(e)
        pub.publish(image_out)
        rate.sleep()
