#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from pimage_lib import pimage as pi
import yaml
from sensor_msgs.msg import CameraInfo
import rospkg
import os

class ImageProcessor:
    def __init__(self):
        
        self.bridge = CvBridge()
    
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('pimage_utils')
        config_file_path = os.path.join(package_path, 'config/ost.yaml')
        rospy.loginfo(f"Using camera calibration from: {config_file_path}")

        self.camera_info_msg = self.load_camera_info(config_file_path)

        self.image_sub = rospy.Subscriber('/input/image', Image, self.image_callback)
        self.image_pub = rospy.Publisher('/output/image', Image, queue_size=1)
        self.camera_info_pub = rospy.Publisher('/output/camera_info', CameraInfo, queue_size=1)

    def load_camera_info(self, yaml_file):
        with open(yaml_file, "r") as file_handle:
            calib_data = yaml.safe_load(file_handle)

        camera_info_msg = CameraInfo()

        camera_info_msg.width = calib_data['image_width']
        camera_info_msg.height = calib_data['image_height']

        camera_info_msg.K = calib_data['camera_matrix']['data']

        camera_info_msg.D = calib_data['distortion_coefficients']['data']

        camera_info_msg.R = calib_data['rectification_matrix']['data']

        camera_info_msg.P = calib_data['projection_matrix']['data']

        camera_info_msg.distortion_model = calib_data['distortion_model']

        return camera_info_msg


    def image_callback(self, msg):

        img_raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

        img_dif = pi.extractDif(img_raw)

        processed_image_msg = self.bridge.cv2_to_imgmsg(img_dif, encoding='bgr8')
        
        processed_image_msg.header = msg.header
        self.camera_info_msg.header = msg.header

        self.camera_info_pub.publish(self.camera_info_msg)
        self.image_pub.publish(processed_image_msg)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('image_processor', anonymous=True)
    processor = ImageProcessor()
    processor.run()


