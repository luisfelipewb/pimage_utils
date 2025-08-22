#!/usr/bin/env python3

import rospy
import rospkg
import tf
import yaml
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs import point_cloud2
import threading


class SimulatedWaste:
    def __init__(self):
        rospy.init_node('simulated_waste', log_level=rospy.INFO)

        # Parameters
        self.frame_id = rospy.get_param('~frame_id', 'world')
        self.rate = rospy.get_param('~rate', 2.0)
        self.distance_threshold = rospy.get_param('~distance_threshold', 0.33)
        self.clean_waste_positions = rospy.get_param('~clean_waste_positions', True)

        self.waste_pub = rospy.Publisher('~simulated_waste', PointCloud2, queue_size=1)
        self.point_sub = rospy.Subscriber('~point', PointStamped, self.point_callback)

        rospack = rospkg.RosPack()


        waypoints_file = rospy.get_param('~config_file', 'empty.yaml')
        file_path = rospack.get_path('kingfisher_experiments') + '/config/' + waypoints_file
        self.waste_points, self.frame_id = self.load_yaml_config(file_path)
        print(f"Loaded waste points: \n {self.waste_points} \n from {file_path} with frame_id {self.frame_id}")

        self.waste_positions = []

        for point in self.waste_points:
            stamped_point = PointStamped()
            stamped_point.header.frame_id = self.frame_id
            stamped_point.header.stamp = rospy.Time.now()
            stamped_point.point.x = point[0]
            stamped_point.point.y = point[1]
            stamped_point.point.z = 0.0
            self.waste_positions.append(stamped_point)
        print(f"Simulated waste positions: {len(self.waste_positions)}")

        # Prepare the point cloud message
        self.point_cloud = PointCloud2()
        self.point_cloud.header.frame_id = "world"
        self.point_cloud.height = 1
        self.point_cloud.width = len(self.waste_positions)
        self.point_cloud.is_dense = True
        self.point_cloud.is_bigendian = False
        self.point_cloud.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)
        ]
        points = [
            [point.point.x, point.point.y, point.point.z]
            for point in self.waste_positions
        ]

        self.cloud_msg = point_cloud2.create_cloud_xyz32(self.point_cloud.header, points)


        # Create a tf buffer and listener
        self.tf_listener = tf.TransformListener()
        # 1s delay to allow the tf listener to initialize
        rospy.sleep(1.0)

        period = 1.0 / self.rate
        self.detection_timer = rospy.Timer(rospy.Duration(period), self.publish_waste)
        self.cleanup_timer = rospy.Timer(rospy.Duration(0.01), self.cleanup)
        self.start_time = rospy.Time.now()

        self.array_lock = threading.Lock()

        rospy.on_shutdown(self.shutdown_hook)

    def load_yaml_config(self, file_path):
        """Load the YAML configuration file."""
        try:
            with open(file_path, 'r') as file:
                config = yaml.safe_load(file)
        except FileNotFoundError:
            rospy.logerr(f"File {file_path} not found")
            exit(1)

        waste_points = config['simulated_waste']
        frame_id = config['frame_id']
        offset = config['offset']
        for point in waste_points:
            point[0] += offset[0]
            point[1] += offset[1]

        return waste_points, frame_id

    def point_callback(self, msg):
        print(f"Received point: {msg.point.x}, {msg.point.y}, {msg.point.z}")
        try:
            # Transform the received point to the world frame
            self.tf_listener.waitForTransform("world", msg.header.frame_id, msg.header.stamp, rospy.Duration(1.0))
            world_point = self.tf_listener.transformPoint("world", msg)
            with self.array_lock:
                self.waste_positions = [world_point]
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logwarn("TF transform unavailable: %s", e)

    def cleanup(self, event):

        if self.clean_waste_positions is False:
            return

        desired_time = rospy.Time.now() # TODO: improve
        try:
            self.tf_listener.waitForTransform(self.frame_id, "base_link", desired_time, rospy.Duration(1.0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logwarn("TF transform unavailable: %s", e)
            pass

        collected_idx = []
        with self.array_lock:
            for i in range(len(self.waste_positions)):
                # Transform the waste positions to the base_link frame
                self.waste_positions[i].header.stamp = desired_time
                point = self.tf_listener.transformPoint("base_link", self.waste_positions[i])
                # compute x, y distance from the base_link
                distance = (point.point.x**2 + point.point.y**2)**0.5
                if distance < self.distance_threshold:
                    collected_idx.append(i)

            # remove the collected waste positions
            for idx in sorted(collected_idx, reverse=True):
                self.waste_positions.pop(idx)


    def publish_waste(self, event):

        # Populate point cloud with simulated waste data
        with self.array_lock:
            self.point_cloud = PointCloud2()
            self.point_cloud.header.frame_id = "world"
            self.point_cloud.height = 1
            self.point_cloud.width = len(self.waste_positions)
            self.point_cloud.is_dense = True
            self.point_cloud.is_bigendian = False
            self.point_cloud.fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)
            ]
            points = [
                [point.point.x, point.point.y, point.point.z]
                for point in self.waste_positions
            ]

        self.cloud_msg = point_cloud2.create_cloud_xyz32(self.point_cloud.header, points)

        self.waste_pub.publish(self.cloud_msg)


    def shutdown_hook(self):
        rospy.loginfo('Shutting down WasteDetector node')


if __name__ == '__main__':
    try:
        publisher = SimulatedWaste()
        rospy.spin()

    except rospy.ROSInterruptException:
        pass