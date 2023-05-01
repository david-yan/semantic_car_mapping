import argparse
import json
import numpy as np
import rospy
import tf2_ros

from geometry_msgs.msg import Point, Pose, Quaternion, TransformStamped, Vector3
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from interactive_markers.menu_handler import MenuHandler
from visualization_msgs.msg import InteractiveMarker, InteractiveMarkerControl, Marker, MenuEntry, InteractiveMarkerFeedback
from std_msgs.msg import ColorRGBA


class InteractiveCuboids:

    def __init__(self, box_dim_x=2, box_dim_y=4.5, box_dim_z=2, cuboids_data=None, frame_id="interactive_cuboids_frame", server_name="interactive_cuboids_server", color=(0.5, 0.5, 0.5, 0.7)):
        
        self.box_dim_x = box_dim_x
        self.box_dim_y = box_dim_y
        self.box_dim_z = box_dim_z
        self.cuboid_boxes = []

        self.color = ColorRGBA(*color) # default is gray, alpha=0.7
        self.cuboids_data = cuboids_data
        self.frame_id = frame_id
        self.server_name = server_name

        # init node
        
        

    def visualize(self):
        self.server = InteractiveMarkerServer(self.server_name)

        for i, cuboid in enumerate(self.cuboids_data):
            # Create an interactive marker
            int_marker = InteractiveMarker()
            int_marker.header.frame_id = "odom"
            int_marker.name = "box_%d" % (i)
            int_marker.description = "My Interactive Marker"
            int_marker.pose.position = Point(cuboid['x'], cuboid['y'], cuboid['z'])
            int_marker.pose.orientation = Quaternion(cuboid['qx'], cuboid['qy'], cuboid['qz'], cuboid['w'])
            print(cuboid)

            box_control = self.create_box_marker(x=self.box_dim_x, y=self.box_dim_y, z=self.box_dim_z)
            int_marker.controls.append(box_control)
            self.cuboid_boxes.append(int_marker)

            # Set the callback function for the interactive marker
            self.server.insert(int_marker, self.marker_callback)

            # Publish the interactive marker
            self.server.applyChanges()

        # rospy.spin()

    # Define the callback function for the interactive marker
    def marker_callback(self, feedback):
        # Print the position and orientation of the marker
        print(feedback.pose)

        # Create a transform stamped message
        t = TransformStamped()

        # Set the header
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = "odom"

        # Set the child frame id
        t.child_frame_id = self.frame_id

        # Set the transform translation and rotation
        t.transform.translation = feedback.pose.position
        t.transform.rotation = feedback.pose.orientation

        # Publish the transform
        broadcaster = tf2_ros.StaticTransformBroadcaster()
        broadcaster.sendTransform(t)

    def create_box_marker(self, x=2, y=2, z=2):
        """
        x,y,z: Dimensions of the cube
        len: Dimensions of the 
        """
        scale = Vector3(x, y, z)

        # Create a box control for the interactive marker
        box_control = InteractiveMarkerControl()
        # box_control.interaction_mode = InteractiveMarkerControl.MOVE_ROTATE_3D
        box_control.always_visible = True
        box_control.markers.append(
            Marker(type=Marker.CUBE, scale=scale, color=self.color))

        return box_control


if __name__ == '__main__':
    rospy.init_node('interactive_cuboids')

    # Create an argparse argument parser
    parser = argparse.ArgumentParser(description='Convert a raw text file to JSON')

    # Add arguments
    parser.add_argument('--boxes', help='Path to the boxes file')

    # Parse the arguments
    args = parser.parse_args()
    with open(args.boxes, 'r') as f:
        cuboids_data = json.load(f)

    # Create an instance of the class that contains the main method
    interactive_cuboids = InteractiveCuboids(cuboids_data=cuboids_data)
    interactive_cuboids.visualize()
