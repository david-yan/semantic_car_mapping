from interactive_markers.interactive_marker_server import *
from visualization_msgs.msg import *
from geometry_msgs.msg import Vector3
from std_msgs.msg import ColorRGBA
import copy 
import tf2_ros
from geometry_msgs.msg import TransformStamped, Quaternion, Point
import numpy as np
import pdb
import argparse
import json

# Define the callback function for the interactive marker
def marker_callback( feedback ):
    # Print the position and orientation of the marker
    print(feedback.pose)

    # # Create a transform stamped message
    t = TransformStamped()

    # Set the header
    t.header.stamp = rospy.Time.now()
    t.header.frame_id = "odom"

    # Set the child frame id
    t.child_frame_id = "interactive_cuboids_frame"

    # Set the transform translation and rotation
    t.transform.translation = feedback.pose.position
    t.transform.rotation = feedback.pose.orientation

    # Publish the transform
    broadcaster = tf2_ros.StaticTransformBroadcaster()
    broadcaster.sendTransform(t)

def create_box_marker(int_marker, x=2, y=2, z=2):
    """
    x,y,z: Dimensions of the cube
    len: Dimensions of the 
    """
    scale = Vector3(x, y, z)
    arrow_len = np.max((x,y,z)) * 2
    alpha = 0.7

    # Create a box control for the interactive marker
    box_control = InteractiveMarkerControl()
    # box_control.interaction_mode = InteractiveMarkerControl.MOVE_ROTATE_3D
    box_control.always_visible = True
    box_control.markers.append( Marker(type=Marker.CUBE,
                                        scale=scale,
                                        color=ColorRGBA(0.5, 0.5, 0.5, alpha)))

    # Positive x arrow
    # x_marker = Marker()
    # x_marker.type = Marker.ARROW
    # x_marker.scale = Vector3(arrow_len, arrow_len/10, arrow_len/10)
    # x_marker.color.r = 0.0
    # x_marker.color.g = 1.0
    # x_marker.color.b = 0.0
    # x_marker.color.a = 0.2

    # # Negative x arrow
    # x_marker_neg = copy.deepcopy(x_marker)
    # x_marker_neg.scale.x *= -1

    # # Positive y arrow
    # y_marker = Marker()
    # y_marker.type = Marker.ARROW
    # y_marker.scale = Vector3(arrow_len/10, arrow_len, arrow_len/10)
    # y_marker.color.r = 0.0
    # y_marker.color.g = 1.0
    # y_marker.color.b = 0.0
    # y_marker.color.a = 0.2

    # # Negative y arrow
    # y_marker_neg = copy.deepcopy(y_marker)
    # y_marker_neg.scale.y *= -1

    # # Positive z arrow
    # z_marker = Marker()
    # z_marker.type = Marker.ARROW
    # z_marker.scale = Vector3(arrow_len/10, arrow_len/10, arrow_len)
    # z_marker.color.r = 0.0
    # z_marker.color.g = 1.0
    # z_marker.color.b = 0.0
    # z_marker.color.a = 0.2

    # # Negative z arrow
    # z_marker_neg = copy.deepcopy(z_marker)
    # z_marker_neg.scale.y *= -1

    # # orientation arrow
    # yaw_marker = Marker()
    # yaw_marker.type = Marker.CYLINDER
    # yaw_marker.scale = Vector3(arrow_len, arrow_len, 0.1)
    # yaw_marker.color.r = 0.0
    # yaw_marker.color.g = 0.0
    # yaw_marker.color.b = 1.0
    # yaw_marker.color.a = 0.2
    # # yaw_marker.always_visible = True

    # # create an interactive control for the box marker that allows yaw rotation
    # yaw_control = InteractiveMarkerControl()
    # yaw_control.orientation_mode = InteractiveMarkerControl.FIXED
    # yaw_control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
    # yaw_control.orientation.w = 1
    # yaw_control.orientation.x = 0
    # yaw_control.orientation.y = 1
    # yaw_control.orientation.z = 0
    # yaw_control.markers.append(yaw_marker)
    # int_marker.controls.append(yaw_control)

    # # create an interactive control for the box marker that allows translation along the x-axis
    # control_x = InteractiveMarkerControl()
    # control_x.orientation.w = 1
    # control_x.orientation.x = 1
    # control_x.orientation.y = 0
    # control_x.orientation.z = 0
    # control_x.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
    # control_x.markers.append(x_marker)
    # control_x.markers.append(x_marker_neg)
    # int_marker.controls.append(control_x)

    # # create an interactive control for the box marker that allows translation along the y-axis
    # control_y = InteractiveMarkerControl()
    # control_y.orientation.w = 1
    # control_y.orientation.x = 0
    # control_y.orientation.y = 0
    # control_y.orientation.z = 1
    # control_y.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
    # control_y.markers.append(y_marker)
    # control_y.markers.append(y_marker_neg)
    # int_marker.controls.append(control_y)

    # # create an interactive control for the box marker that allows translation along the z-axis
    # control_z = InteractiveMarkerControl()
    # control_z.orientation.w = 1
    # control_z.orientation.x = 0
    # control_z.orientation.y = 1
    # control_z.orientation.z = 0
    # control_z.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
    # control_z.markers.append(z_marker)
    # control_z.markers.append(z_marker_neg)
    # int_marker.controls.append(control_z)

    return box_control

# Initialize the interactive marker server
def main():
    # Create an argparse argument parser
    parser = argparse.ArgumentParser(description='Convert a raw text file to JSON')

    # Add arguments
    parser.add_argument('--boxes', help='Path to the boxes file')

    # Parse the arguments
    args = parser.parse_args()

    rospy.init_node('interactive_cuboids')
    server = InteractiveMarkerServer("interactive_cuboids_server")

    # FILL OUT THESE PARAMS
    box_dim_x = 2
    box_dim_y = 4.5
    box_dim_z = 2

    with open(args.boxes, 'r') as f:
        cuboids_data = json.load(f)

    cuboid_boxes = []
    for i, cuboid in enumerate(cuboids_data):
        # Create an interactive marker
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = "odom"
        int_marker.name = "box_%d"%(i)
        int_marker.description = "My Interactive Marker"
        int_marker.pose.position = Point(cuboid['x'], cuboid['y'], cuboid['z'])
        int_marker.pose.orientation = Quaternion(cuboid['qx'], cuboid['qy'], cuboid['qz'], cuboid['w'])
        
        box_control = create_box_marker(int_marker, x=box_dim_x, y=box_dim_y, z=box_dim_z)
        int_marker.controls.append(box_control)
        cuboid_boxes.append(int_marker)

        # Set the callback function for the interactive marker
        server.insert(int_marker, marker_callback)

        # Publish the interactive marker
        server.applyChanges()

    rospy.spin()

    

if __name__ == '__main__':
    # rospy.init_node("interactive_marker_node")

    main()
