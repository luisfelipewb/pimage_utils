#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Header
from cv_bridge import CvBridge
import onnxruntime as ort
import numpy as np
import cv2
import yaml
import sensor_msgs.point_cloud2 as pc2
import time

class WasteDetector:
    def __init__(self):
        rospy.init_node('waste_detector', log_level=rospy.INFO)

        self.bridge = CvBridge()

        self.conf_thresh = rospy.get_param('~confidence_thresh', 0.75)
        rospy.loginfo(f"Confidence Threshold for detection: {self.conf_thresh}")

        # Load TensorRT engine
        engine_path = rospy.get_param('~engine')
        self.session, self.input_name, self.output_name = self.load_engine(engine_path)

        # Load camera calibration parameters
        int_path = rospy.get_param('~intrinsics_path')
        rospy.loginfo(f"Using instrincs from file {int_path}")
        ext_path = rospy.get_param('~extrinsics_path')
        rospy.loginfo(f"Using extrinsics from file {ext_path}")

        self.camera_matrix, self.dist_coeffs, self.rvec, self.tvec = self.load_camera_calibration(int_path, ext_path)

        # Prepare fixed projection matrices and values
        self.R_inv, _ = cv2.Rodrigues(-self.rvec)
        self.cam_origin_w = -self.R_inv @ self.tvec
        _, _, self.Z0 = self.cam_origin_w.flatten()

        # Publishers
        self.annotated_image_pub = rospy.Publisher("~image_annotated", Image, queue_size=1)
        self.detected_points_pub = rospy.Publisher("~detections", PointCloud2, queue_size=1)

        # Subscribers
        image_topic = rospy.resolve_name('input_image')
        self.image_sub = rospy.Subscriber(image_topic, Image, self.image_callback)
        rospy.loginfo(f'Subscribed to image topic: {image_topic}')

        rospy.on_shutdown(self.shutdown_hook)


    def load_engine(self, engine_path):
        """ Prepare the ONNX Runtime session """
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1  # Adjust number of threads for CPU
        sess_options.inter_op_num_threads = 4

        # Load the model
        session = ort.InferenceSession(engine_path, sess_options, providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        return session, input_name, output_name


    def load_camera_calibration(self, intrinsics_path, extrinsics_path):
        """ Load intrinsics and extrics matrices from the configuration files """

        with open(intrinsics_path, 'r') as f:
            intrinsics = yaml.safe_load(f)
        with open(extrinsics_path, 'r') as f:
            extrinsics = yaml.safe_load(f)
        
        camera_matrix = np.array(intrinsics['camera_matrix']['data']).reshape(3,3)
        dist_coeffs = np.array(intrinsics['distortion_coefficients']['data']).reshape(1,5)
        rvec = np.array(extrinsics['rvec'])
        tvec = np.array(extrinsics['tvec'])

        return camera_matrix, dist_coeffs, rvec, tvec


    def preprocess_image(self, image, target_size=(640, 640)):
        """ Converts image to format necessary for inference input """

        # Resize image for yolo input
        img_resized = cv2.resize(image, target_size)

        # Normalize and convert BGR to RGB
        img_resized = img_resized[..., ::-1] / 255.0

        # Convert to CHW format and add batch dimension
        img_input = np.transpose(img_resized, (2, 0, 1))[None, ...].astype(np.float32)

        return img_input

    
    def run_inference(self, session, input_name, output_name, img_input):
        """ Run inference on the image and measure time """

        rospy.logdebug(f"Running inference...")
        start_time = time.time()
        outputs = session.run([output_name], {input_name: img_input})
        duration = time.time() - start_time
        rospy.logdebug(f"Inference completed in {duration:.3f}s")
        return outputs[0]

    def convert_xywh_to_xyxy(self, boxes):
        """ Convert bounding boxes from xywh to xyxy format """

        x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2
        return np.column_stack([x1, y1, x2, y2, boxes[:, 4]])
    
    def postprocess_output(self, output, image_size=(1024, 1224)):
        """ Postprocess the raw model output to get bounding boxes."""

        original_boxes = output[output[:, :, 4] > self.conf_thresh]

        # Rescale boxes to original image size
        original_boxes[:, 0] *= image_size[1] / 640  # x1
        original_boxes[:, 1] *= image_size[0] / 640  # y1
        original_boxes[:, 2] *= image_size[1] / 640  # x2
        original_boxes[:, 3] *= image_size[0] / 640  # y2

        converted_boxes = self.convert_xywh_to_xyxy(original_boxes)

        # Apply Non-Maximum Suppression (NMS)
        cv2_boxes = converted_boxes[:, :4]
        cv2_scores = converted_boxes[:, 4]
        IoU_thresh = 0.4
        indices = cv2.dnn.NMSBoxes(cv2_boxes.tolist(), cv2_scores.tolist(), self.conf_thresh, IoU_thresh)

        # Filter selected boxes
        XcYcWH_boxes = original_boxes[indices]
        XYXY_boxes = converted_boxes[indices]

        
        return XcYcWH_boxes, XYXY_boxes
    
    def draw_boxes_on_image(self, img, XcYcWH_boxes, XYXY_boxes, color=(0, 255, 0), marker_color=(0, 0, 255)):
        for box in XYXY_boxes:
            # Draw bounding box
            x1, y1, x2, y2 = box[:4]
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)
            
        for box in XcYcWH_boxes:
            # Draw center
            xc, yc = box[:2]
            cv2.drawMarker(img, (int(xc), int(yc)), marker_color, markerType=cv2.MARKER_CROSS, markerSize=30, thickness=3)
            
        return img
    
    def project_to_world(self, uv_points):
        """ Project from u,v pixel coordinates to x,y,0 point in the robot frame """

        world_coordinates = []

        if len(uv_points > 0):
            # Convert to a float array expected by cv2
            uv_points = np.array([uv_points], dtype=np.float32)
            points_undistorted = cv2.undistortPoints(uv_points, self.camera_matrix, self.dist_coeffs)

            for uv_point in points_undistorted:
                uv1 = np.array([[uv_point[0][0], uv_point[0][1], 1]], dtype=np.float32).T
                ray_direction_w = self.R_inv @ uv1

                _, _, dz = ray_direction_w.flatten()
                if abs(dz) < 0.00001: 
                    raise ValueError("parallel to z=0")
                t = -self.Z0 / dz

                intersection_point = self.cam_origin_w + t * ray_direction_w
                world_coordinates.append(intersection_point.flatten())
        return world_coordinates
    
    def publish_pointcloud(self, points, header):
        
        # Hardcoded and must match the extrinsics calibration
        header_pc = Header()
        header_pc.seq = header.seq
        header_pc.stamp = header.stamp
        header_pc.frame_id = 'base_link'

        point_list = [(p[0], p[1], p[2]) for p in points]
        cloud_msg = pc2.create_cloud_xyz32(header_pc, point_list)

        self.detected_points_pub.publish(cloud_msg)


    def image_callback(self, msg):
        rospy.logdebug('Received image')
        start_time = time.time()

        # Preprocess image
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        img_input = self.preprocess_image(img)

        # Run inference
        inference_output = self.run_inference(self.session, self.input_name, self.output_name, img_input)

        # Threshold detections and NMS
        XcYcWH_boxes, XYXY_boxes = self.postprocess_output(inference_output)

        num_detections = len(XcYcWH_boxes)
        rospy.logdebug(f"Detected {num_detections} objects")

        center_positions = XcYcWH_boxes[:,:2] # get only XcYc positions
        coordinates = self.project_to_world(center_positions)
        self.publish_pointcloud(coordinates, msg.header)

        if self.annotated_image_pub.get_num_connections() > 0:
            labeled_image = self.draw_boxes_on_image(img, XcYcWH_boxes, XYXY_boxes, color=(0, 255, 0), marker_color=(0, 0, 255))
            labeled_image_msg = self.bridge.cv2_to_imgmsg(labeled_image, "passthrough")
            labeled_image_msg.header = msg.header
            self.annotated_image_pub.publish(labeled_image_msg)

        duration = time.time() - start_time
        rospy.logdebug(f"Total pipeline duration: {duration:.2f}s")


    def shutdown_hook(self):
        rospy.loginfo('Shutting down WasteDetector node')


if __name__ == '__main__':
    try:
        publisher = WasteDetector()
        rospy.spin()

    except rospy.ROSInterruptException:
        pass