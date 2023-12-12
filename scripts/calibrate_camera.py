#!/usr/bin/env python3

import rospy
import numpy as np
import cv2
import cv2.aruco as aruco
import os
import tf
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from autolab_core import Point, CameraIntrinsics, RigidTransform

from geometry_msgs.msg import Pose

from autolab_core import RigidTransform

def get_object_center_point_in_world(image_x, image_y, depth_image, namespace, intrinsics):

    object_center = Point(np.array([image_x, image_y]), namespace + '_azure_kinect_overhead')
    object_depth = np.mean(depth_image[image_y-1:image_y+1, image_x-1:image_x+1])

    return intrinsics.deproject_pixel(object_depth, object_center)

def rtvec_to_matrix(rvec, tvec):
    """
    Convert rotation vector and translation vector to 4x4 matrix
    """
    rvec = np.asarray(rvec)
    tvec = np.asarray(tvec).flatten()

    T = np.eye(4)
    R, jac = cv2.Rodrigues(rvec)
    T[:3, :3] = R
    T[:3, 3] = tvec
    return T

def estimatePoseSingleMarkers(corners, marker_size, mtx, distortion):
    '''
    This will estimate the rvec and tvec for each of the marker corners detected by:
       corners, ids, rejectedImgPoints = detector.detectMarkers(image)
    corners - is an array of detected corners for each detected marker in the image
    marker_size - is the size of the detected markers
    mtx - is the camera matrix
    distortion - is the camera distortion matrix
    RETURN list of rvecs, tvecs, and trash (so that it corresponds to the old estimatePoseSingleMarkers())
    '''
    marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, -marker_size / 2, 0],
                              [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
    trash = []
    rvecs = []
    tvecs = []
    for c in corners:
        nada, R, t = cv2.solvePnP(marker_points, c, mtx, distortion, False, cv2.SOLVEPNP_IPPE_SQUARE)
        rvecs.append(R)
        tvecs.append(t)
        trash.append(nada)
    return rvecs, tvecs, trash

def save_camera_intrinsics(file_path, namespace, camera_info):
    intrinsics_str = '{"_frame": "'+namespace+'_azure_kinect_overhead", "_fx": '+ str(camera_info.K[0]) + \
                       ', "_fy": '+ str(camera_info.K[4]) + ', "_cx": '+ str(camera_info.K[2]) + \
                       ', "_cy": '+ str(camera_info.K[5]) + ', "_skew": 0.0, "_height": '+ \
                       str(camera_info.height) + ', "_width": '+ str(camera_info.width) + ', "_K": 0}'
    f = open(file_path, "w")
    f.write(intrinsics_str)
    f.close()

def run():

    rospy.init_node("calibrate_camera")

    namespace = rospy.get_param("calibrate_camera/namespace")
    print(namespace)
    root_pwd = rospy.get_param("calibrate_camera/root_pwd")
    print(root_pwd)
    
    T_aruco_ee = RigidTransform.load(root_pwd+'/config/aruco_ee.tf')

    tf_listener = tf.TransformListener()
    bridge = CvBridge()
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)  # Use 5x5 dictionary to find markers
    parameters = aruco.DetectorParameters()  # Marker detection parameters
    detector = aruco.ArucoDetector(dictionary, parameters)
    camera_info = rospy.wait_for_message('/'+namespace+'/rgb/camera_info', CameraInfo)
    save_camera_intrinsics(root_pwd+'/calib/'+namespace+'_azure_kinect_intrinsics.intr', namespace, camera_info)
    print('Saved camera intrinsics to ' + root_pwd+'/calib/'+namespace+'_azure_kinect_intrinsics.intr')
    intrinsics = CameraIntrinsics.load(root_pwd+'/calib/'+namespace+'_azure_kinect_intrinsics.intr')
    camera_matrix = np.array(camera_info.K).reshape((3,3))

    while True:
        try:
            rgb_image_msg = rospy.wait_for_message('/'+namespace+'/rgb/image_rect_color', Image)
            rgb_cv_image = bridge.imgmsg_to_cv2(rgb_image_msg, desired_encoding='bgr8')
            gray = cv2.cvtColor(rgb_cv_image, cv2.COLOR_BGR2GRAY)  # Change grayscale

            corners, ids, rejected_img_points = detector.detectMarkers(gray)
            print(ids)
            if ids is None:
                continue
            depth_images = []
            for i in range(5):
                depth_image_message = rospy.wait_for_message('/'+namespace+'/depth_to_rgb/hw_registered/image_rect', Image)
                depth_image = bridge.imgmsg_to_cv2(depth_image_message, desired_encoding='passthrough')
                depth_images.append(depth_image)

            centers = []
            corner_points_in_world = []

            for corner in corners:
                rvecs, tvecs, trash = estimatePoseSingleMarkers(corner, 0.1, camera_matrix, np.zeros(5))
                matrix = rtvec_to_matrix(rvecs[0], tvecs[0])
                print(matrix)

                new_corner = corner.reshape((4,2)).astype(int)
                centers.append(np.rint(np.mean(new_corner, axis=0)).astype(int))
                corner_points = []

                for corner_point in range(4):
                  corner_point_in_world = get_object_center_point_in_world(new_corner[corner_point,0], new_corner[corner_point,1], depth_image, namespace, intrinsics)
                  corner_points.append([corner_point_in_world.x, corner_point_in_world.y, corner_point_in_world.z])
                corner_points_in_world.append(corner_points)

            center_points_in_world = []
            for center in centers:
              for depth_image in depth_images:
                center_point_in_world = get_object_center_point_in_world(center[0], center[1], depth_image, namespace, intrinsics)
                center_points_in_world.append([center_point_in_world.x, center_point_in_world.y, center_point_in_world.z])
            center_point_in_world = np.mean(center_points_in_world, axis=0)
            print(center_point_in_world)

            T_aruco_camera = RigidTransform(rotation=np.array(matrix[:3,:3]), 
                                            translation = np.array(center_point_in_world), 
                                            from_frame='aruco', to_frame=namespace + '_azure_kinect_overhead')
            print(T_aruco_camera)

            try:
                (trans,rot) = tf_listener.lookupTransform('/'+namespace+'/base_link', '/'+namespace+'/tool0', rospy.Time(0))
                pose = Pose()
                pose.position.x = trans[0]
                pose.position.y = trans[1]
                pose.position.z = trans[2]
                pose.orientation.x = rot[0]
                pose.orientation.y = rot[1]
                pose.orientation.z = rot[2]
                pose.orientation.w = rot[3]

                T_ee_world = RigidTransform.from_pose_msg(pose, from_frame='ee', to_frame='world')

                T_aruco_world = T_ee_world * T_aruco_ee
                print(T_aruco_world)

                T_camera_world = T_aruco_world * T_aruco_camera.inverse()
                print(T_camera_world)
                T_camera_world.save(root_pwd+'/calib/'+namespace+'_azure_kinect_overhead_world.tf')
                print('Saved camera extrinsics to ' + root_pwd+'/calib/'+namespace+'_azure_kinect_overhead_world.tf')
                break
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
                print(e)

        except CvBridgeError as e:
          print(e)

if __name__ == "__main__":
    run()