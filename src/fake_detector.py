#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import PointCloud2, PointField, Image, CameraInfo
from sensor_msgs import point_cloud2
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, PointStamped
import random
import tf
import cv2
import yaml
import numpy as np
from cv_bridge import CvBridge


class FakeDetector:
    def __init__(self):
        rospy.init_node('fake_detector', log_level=rospy.INFO)
        
        self.image_size = (1224, 1024)  # (width, height)
        # Parameters
        # self.world_frame = rospy.get_param('~world_frame', 'world')
        self.robot_frame = rospy.get_param('~robot_frame', 'base_link')
        self.rate = rospy.get_param('~rate', 2.0)

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

        # Subscribers
        self.waste_sub = rospy.Subscriber('/simulated_waste', MarkerArray, self.waste_callback)

        # Publishers
        self.fov_pub = rospy.Publisher('/fov_marker', Marker, queue_size=1)
        self.point_cloud_pub = rospy.Publisher('/waste_point_cloud', PointCloud2, queue_size=1)
        self.annotated_image_pub = rospy.Publisher("~debug_img/image_raw", Image, queue_size=1)
        self.camera_info_pub = rospy.Publisher("~debug_img/camera_info", CameraInfo, queue_size=1)

        # Create a marker template
        self.fov_marker = self.create_fov_marker()
        
        # Create an array of Stamped points
        self.waste_positions = []
        
        self.bridge = CvBridge()

        self.pixel_noise_radius = rospy.get_param('~pixel_noise_radius', 12)

        self.tf_listener = tf.TransformListener()
        rospy.sleep(2.0)  # Allow time for the tf listener to initialize

        rospy.on_shutdown(self.shutdown_hook)

    def create_fov_marker(self):

        fov_marker = Marker()
        fov_marker.header.frame_id = self.robot_frame
        fov_marker.header.stamp = rospy.Time.now()
        fov_marker.ns = "fov"
        fov_marker.id = 0
        fov_marker.type = Marker.LINE_STRIP
        fov_marker.action = Marker.ADD
        fov_marker.scale.x = 0.1
        fov_marker.color.a = 0.8
        fov_marker.color.r = 0.0
        fov_marker.color.g = 1.0
        fov_marker.color.b = 0.0
        fov_marker.lifetime = rospy.Duration(0)
        fov_marker.pose.position.x = 0.0
        fov_marker.pose.position.y = 0.0
        fov_marker.pose.position.z = 0.0
        fov_marker.pose.orientation.x = 0.0
        fov_marker.pose.orientation.y = 0.0
        fov_marker.pose.orientation.z = 0.0
        fov_marker.pose.orientation.w = 1.0

        # Create numpy array with 4 points (shape 5,2,1) with pixel coordinates
        offset = 150
        # image_points = np.array([[0+offset, 0], [self.image_size[1], 0], [self.image_size[1], self.image_size[0]], [0+offset, self.image_size[0]]], dtype=np.float32)
        image_points = np.array([[[0, offset]],
                                 [[0, self.image_size[1]]],
                                 [[self.image_size[0], self.image_size[1]]],
                                 [[self.image_size[0], offset]],
                                 ], dtype=np.float32)

        # Convert to local frame 
        local_points = self.project_to_local(image_points)

        # Create a list of Point objects
        fov_marker.points = []
        for point in local_points:
            p = Point(point[0], point[1], point[2])
            fov_marker.points.append(p)
        return fov_marker

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
    
    def publish_debug_image(self, gt_points, noisy_points):

        image = np.ones((self.image_size[1], self.image_size[0], 3), dtype=np.uint8) * 255
        for point in gt_points:
            cv2.circle(image, (int(point[0][0]), int(point[0][1])), self.pixel_noise_radius, (255, 0, 0), 2)

        for point in noisy_points:
            # Add crosshair
            cv2.drawMarker(image, (int(point[0][0]), int(point[0][1])), (173, 127, 168), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)
        
        # print(image.shape)
        image_msg = self.bridge.cv2_to_imgmsg(image, encoding="bgr8")
        image_msg.header.frame_id = self.robot_frame
        image_msg.header.stamp = rospy.Time.now()
        image_msg.width = self.image_size[0]
        image_msg.height = self.image_size[1]
        image_msg.step = self.image_size[0] * 3
        
        self.annotated_image_pub.publish(image_msg)
        # Also publish camera info 
        camera_info_msg = CameraInfo()
        camera_info_msg.header = image_msg.header
        camera_info_msg.width = self.image_size[0]
        camera_info_msg.height = self.image_size[1]
        camera_info_msg.K = self.camera_matrix.flatten().tolist()
        camera_info_msg.D = self.dist_coeffs.flatten().tolist()
        camera_info_msg.R = np.eye(3).flatten().tolist()
        camera_info_msg.P = self.camera_matrix.flatten().tolist() + [0, 0, 0]
        camera_info_msg.distortion_model = "plumb_bob"
        self.camera_info_pub.publish(camera_info_msg)
        return

    def project_to_image(self, points):
        """ Project 3D points to 2D image coordinates 
        retunrs the points in image coordinates"""

        robot_points = np.array([[p.point.x, p.point.y, p.point.z] for p in points], dtype=np.float32)

        robot_points = robot_points[robot_points[:, 0] > 0] # Drop points behind the robot

        image_points, _ = cv2.projectPoints(robot_points, self.rvec, self.tvec, self.camera_matrix, self.dist_coeffs)

        # Filter points within the image size
        image_points = image_points[image_points[:, 0, 0] >= 0]
        image_points = image_points[image_points[:, 0, 0] < self.image_size[0]]
        image_points = image_points[image_points[:, 0, 1] >= 0]
        image_points = image_points[image_points[:, 0, 1] < self.image_size[1]]

        # Create image and publish it for debugging 
        
        return image_points
    
    def project_to_local(self, uv_points):
        """ Project from u,v pixel coordinates to x,y,0 point in the local (robot) frame """

        local_coordinates = []

        if len(uv_points > 0):
            # Convert to a float array expected by cv2
            # uv_points = np.array([uv_points], dtype=np.float32)
            points_undistorted = cv2.undistortPoints(uv_points, self.camera_matrix, self.dist_coeffs)

            for uv_point in points_undistorted:
                uv1 = np.array([[uv_point[0][0], uv_point[0][1], 1]], dtype=np.float32).T
                ray_direction_w = self.R_inv @ uv1

                _, _, dz = ray_direction_w.flatten()
                if abs(dz) < 0.00001: 
                    raise ValueError("parallel to z=0")
                t = (-self.Z0 -0.15) / dz # There might be a difference of camera hight (water plane=0) from calibration and gazebo.

                intersection_point = self.cam_origin_w + t * ray_direction_w
                local_coordinates.append(intersection_point.flatten())
        return local_coordinates


    def add_detection_noise(self, image_coordinates):

        # skip if image_coordinates is empty
        if len(image_coordinates) == 0:
            return []
        # print(image_coordinates)
        # print(image_coordinates[0])
        noisy_coordinates = np.array(image_coordinates, dtype=np.float32)
        for i in range(image_coordinates.shape[0]): # TODO: convert to vecotrized operation
            # print(f"Coordinate: {coordinate}")
            # Add noise to the point
            # Circle around the position
            noise_radius = random.uniform(0, self.pixel_noise_radius)
            angle = random.uniform(0, 2 * np.pi)
            noise_x = int(noise_radius * np.cos(angle))
            noise_y = int(noise_radius * np.sin(angle))

            noisy_coordinates[i][0][0] += noise_x
            noisy_coordinates[i][0][1] += noise_y

        # clamp to image size (truncate to max and min values)
        noisy_coordinates = np.clip(noisy_coordinates, 0, [self.image_size[0], self.image_size[1]])

        return noisy_coordinates
    
    def stamped_point_to_pcl2(self, stamped_points):

        header = stamped_points[0].header
        # Create a PointCloud2 message
        point_cloud = PointCloud2()
        point_cloud.header = header

        point_cloud.height = 1
        point_cloud.width = len(stamped_points)
        point_cloud.is_dense = True
        point_cloud.is_bigendian = False
        point_cloud.fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1)
        ]
        
        points = [
            [point.point.x, point.point.y, point.point.z]
            for point in stamped_points
        ]

        cloud_msg = point_cloud2.create_cloud_xyz32(header, points)

        return cloud_msg

    def waste_callback(self, msg):
        
        # Skip if the message is empty
        if not msg.markers:
            return

        frame_id = msg.markers[0].header.frame_id
        stamp = msg.markers[0].header.stamp

        # Stores a StampedPoint for each waste marker in the array
        waste_positions = []
        for marker in msg.markers:
            if marker.ns == "waste":
                # Create a StampedPoint for each marker
                stamped_point = PointStamped()
                stamped_point.point.x = marker.pose.position.x
                stamped_point.point.y = marker.pose.position.y
                stamped_point.point.z = marker.pose.position.z
                # Set the header
                stamped_point.header.frame_id = frame_id
                stamped_point.header.stamp = stamp
                waste_positions.append(stamped_point)

        # Convert all the points to the robot frame
        try:
            self.tf_listener.waitForTransform(self.robot_frame, frame_id, stamp, rospy.Duration(1.0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logwarn("TF transform unavailable: %s", e)
            pass

        local_waste_positions = []
        for point in waste_positions:
            local_waste_positions.append(self.tf_listener.transformPoint(self.robot_frame, point))


        # print(f"Transformed waste positions: {local_waste_positions}")
        image_points = self.project_to_image(local_waste_positions)
        # print(f"Image points type {type(image_points)} shape {image_points.shape} \n{image_points}\n")

        noisy_image_points = self.add_detection_noise(image_points)

        self.publish_debug_image(image_points, noisy_image_points)

        local_coordinates = self.project_to_local(noisy_image_points)
        # print(f"Local coordinates: type:{type(local_coordinates)} {local_coordinates}\n")


        # print(f"Image points: type:{type(image_points)} shape:{image_points.shape}\n{image_points}\n")
        # print(f"Noisy image points: type:{type(noisy_image_points)} shape:{noisy_image_points.shape}\n{noisy_image_points}\n\n")

        
        # Create an array of stamped points
        detections = []
        for i, point in enumerate(local_coordinates):
            stamped_point = PointStamped()
            stamped_point.point.x = point[0]
            stamped_point.point.y = point[1]
            stamped_point.point.z = 0.0
            stamped_point.header.frame_id = self.robot_frame
            stamped_point.header.stamp = stamp
            detections.append(stamped_point)
        
        point_cloud = self.stamped_point_to_pcl2(detections)

        # Publish the point cloud and field of view
        self.point_cloud_pub.publish(point_cloud)
        self.fov_pub.publish(self.fov_marker)


    def shutdown_hook(self):
        rospy.loginfo('Shutting down WasteDetector node')


if __name__ == '__main__':
    try:
        publisher = FakeDetector()
        rospy.spin()

    except rospy.ROSInterruptException:
        pass