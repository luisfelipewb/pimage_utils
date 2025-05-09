#!/usr/bin/env python3

import rospy
from visualization_msgs.msg import Marker, MarkerArray
import random
import tf
from geometry_msgs.msg import PointStamped
import threading


class SimulatedWaste:
    def __init__(self):
        rospy.init_node('simulated_waste', log_level=rospy.INFO)
        
        # Parameters
        self.frame_id = rospy.get_param('~frame_id', 'world')
        self.rate = rospy.get_param('~rate', 2.0)

        # Create a publisher for the waste detection (visualization marker array)
        self.waste_pub = rospy.Publisher('/simulated_waste', MarkerArray, queue_size=1)

        self.waste_positions = []
        x_values = [0.5, 1, 1.5, 2, 3, 5, 9, 15, 50]
        y_values = range(-5, 6)  # y from -5 to 5 in steps of 1
        # x_values = [2.0]
        # y_values = [-1.0, 1.0]
        for x in x_values:
            for y in y_values:
                stamped_point = PointStamped()
                stamped_point.header.frame_id = self.frame_id
                stamped_point.header.stamp = rospy.Time.now()
                stamped_point.point.x = x
                stamped_point.point.y = y
                stamped_point.point.z = 0.0
                self.waste_positions.append(stamped_point)

        # Create a marker template
        self.marker_template = Marker()
        self.marker_template.header.frame_id = "world"
        self.marker_template.ns = "waste"
        self.marker_template.type = Marker.SPHERE
        self.marker_template.action = Marker.ADD
        self.marker_template.scale.x = 0.1
        self.marker_template.scale.y = 0.1
        self.marker_template.scale.z = 0.1
        self.marker_template.color.a = 1.0
        self.marker_template.color.r = 0.0
        self.marker_template.color.g = 0.0
        self.marker_template.color.b = 1.0
        self.marker_template.lifetime = rospy.Duration(1)
        self.marker_template.pose.position.x = 0.0
        self.marker_template.pose.position.y = 0.0
        self.marker_template.pose.position.z = 0.0
        self.marker_template.pose.orientation.x = 0.0
        self.marker_template.pose.orientation.y = 0.0
        self.marker_template.pose.orientation.z = 0.0
        self.marker_template.pose.orientation.w = 1.0


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
        


    def cleanup(self, event):

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
                if distance < 0.33:
                    collected_idx.append(i)

            # remove the collected waste positions
            for idx in sorted(collected_idx, reverse=True):
                self.waste_positions.pop(idx)


    def publish_waste(self, event):
    
        marker_array = MarkerArray()
        # for i in range(len(self.waste_positions)):
        #     x_offset = random.uniform(-0.1, 0.1)
        #     y_offset = random.uniform(-0.1, 0.1)
        #     self.waste_positions[i] = (self.waste_positions[i][0] + x_offset,
        #                                self.waste_positions[i][1] + y_offset)
        
        # Populate the MarkerArray with simulated waste data
        with self.array_lock:
            for i in range(len(self.waste_positions)):
                marker = Marker()
                marker.header = self.marker_template.header
                marker.ns = self.marker_template.ns
                marker.type = self.marker_template.type
                marker.action = self.marker_template.action
                marker.scale = self.marker_template.scale
                marker.color = self.marker_template.color
                marker.lifetime = self.marker_template.lifetime
                marker.pose.orientation = self.marker_template.pose.orientation
                marker.id = i
                marker.header.stamp = rospy.Time.now()
                marker.pose.position.x = self.waste_positions[i].point.x
                marker.pose.position.y = self.waste_positions[i].point.y
                marker.pose.position.z = 0.0
                marker_array.markers.append(marker)

        self.waste_pub.publish(marker_array)


    def shutdown_hook(self):
        rospy.loginfo('Shutting down WasteDetector node')


if __name__ == '__main__':
    try:
        publisher = SimulatedWaste()
        rospy.spin()

    except rospy.ROSInterruptException:
        pass