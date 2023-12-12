import numpy as np
import rospy
import cv2
import sys
import cv2.aruco as aruco

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose2D
from autolab_core import Point, CameraIntrinsics, RigidTransform

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


def get_object_center_point_in_world(image_x, image_y, depth_image, intrinsics, transform):

    object_center = Point(np.array([image_x, image_y]), 'azure_kinect_overhead')
    object_depth = depth_image[image_y, image_x]

    return transform * intrinsics.deproject_pixel(object_depth, object_center)

print(cv2.__version__)


AZURE_KINECT_INTRINSICS = 'azure_kinect_intrinsics.intr'
AZURE_KINECT_EXTRINSICS = 'azure_kinect_overhead_to_world.tf'

class image_converter:

  def __init__(self):

    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber('/rgb/image_rect_color', Image, self.callback)
    self.matrix_coefficients = np.array([1943.6480712890625, 0.0, 2045.5838623046875, 0.0, 1943.1328125, 1576.270751953125, 0.0, 0.0, 1.0])
    self.distortion_coefficients = np.array([0, 0, 0, 0, 0])
    self.intrinsics = CameraIntrinsics.load(AZURE_KINECT_INTRINSICS)
    self.azure_kinect_to_world_transform = RigidTransform.load(AZURE_KINECT_EXTRINSICS) 
    self.pub = rospy.Publisher('/amr_pose', Pose2D, queue_size=10)

  def callback(self,data):
    try:
        cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
        #bgr_image = cv2.cvtColor(cv_image, cv2.COLOR_BGRA2BGR)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)  # Change grayscale
        #print(gray.shape)
        dictionary = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)  # Use 5x5 dictionary to find markers
        parameters = aruco.DetectorParameters()  # Marker detection parameters
        detector = aruco.ArucoDetector(dictionary, parameters)

        corners, ids, rejected_img_points = detector.detectMarkers(gray)
        depth_image_message = rospy.wait_for_message('/depth_to_rgb/hw_registered/image_rect', Image)
        depth_image = self.bridge.imgmsg_to_cv2(depth_image_message, desired_encoding='passthrough')

        #print(ids)

        #print(corners)

        centers = []

        for corner in corners:
            #rvecs, tvecs, trash = estimatePoseSingleMarkers(corner, 0.1, self.matrix_coefficients, np.zeros(5))
            #matrix = rtvec_to_matrix(rvecs[0], tvecs[0])
            #print(matrix)

            centers.append(np.rint(np.mean(corner.reshape((4,2)), axis=0)).astype(int))


        center_points_in_world = {}
        amr_center = []
        center_idx = 0
        for center in centers:
            center_point_in_world = get_object_center_point_in_world(center[0], center[1], depth_image, self.intrinsics, self.azure_kinect_to_world_transform)
            #print([center_point_in_world.x, center_point_in_world.y, center_point_in_world.z])
            center_points_in_world[ids[center_idx][0]] = [center_point_in_world.x, center_point_in_world.y, center_point_in_world.z]
            center_idx += 1
            amr_center.append([center_point_in_world.x, center_point_in_world.y, center_point_in_world.z])
        #print(center_points_in_world)

        if len(amr_center) == 4:
            true_amr_center = np.mean(np.array(amr_center),axis=0)
            true_amr_center[2] = 0.2
        elif len(amr_center) >= 2:
            if 69 in ids and 45 in ids:
                y = (center_points_in_world[69][1] + center_points_in_world[45][1])/2
                x = (center_points_in_world[69][0] + center_points_in_world[45][0])/2
                z = 0.2
                true_amr_center = np.array([x,y,z])
            if 37 in ids and 58 in ids:
                y = (center_points_in_world[37][1] + center_points_in_world[58][1])/2
                x = (center_points_in_world[37][0] + center_points_in_world[58][0])/2
                z = 0.2
                true_amr_center = np.array([x,y,z])
        print(true_amr_center)
        if 58 in ids:
            adjusted_point_58 = np.array(center_points_in_world[58]) - true_amr_center
            angle = np.arctan2(adjusted_point_58[1], adjusted_point_58[0]) - np.pi/4
            amr_angle = angle * 180 / np.pi
        elif 69 in ids:
            adjusted_point_69 = np.array(center_points_in_world[69]) - true_amr_center
            angle = np.arctan2(adjusted_point_69[1], adjusted_point_69[0]) - np.pi*3/4
            amr_angle = angle * 180 / np.pi
        elif 37 in ids:
            adjusted_point_37 = np.array(center_points_in_world[37]) - true_amr_center
            angle = np.arctan2(adjusted_point_37[1], adjusted_point_37[0]) - np.pi*5/4
            amr_angle = angle * 180 / np.pi
        elif 45 in ids:
            adjusted_point_45 = np.array(center_points_in_world[45]) - true_amr_center
            angle = np.arctan2(adjusted_point_45[1], adjusted_point_45[0]) - np.pi*7/4
            amr_angle = angle * 180 / np.pi
        print(amr_angle)

        if np.abs(amr_angle) > 50 or np.any(np.isnan(true_amr_center)):
          pass
        else:
          pose = Pose2D()
          pose.x = true_amr_center[0]
          pose.y = true_amr_center[1]
          pose.theta = angle
          self.pub.publish(pose)
        # if true_amr_center[2] < 0.2 or np.abs(amr_angle) > 30 or np.any(np.isnan(true_amr_center)):
        #     amr_angle = 90
        # else:
        #     true_amr_centers.append(true_amr_center)
        #     angles.append(angle)


        # # lists of ids and the corners beloning to each id
        # corners, ids, rejected_img_points = aruco.detectMarkers(gray, aruco_dict,
        #                                                         parameters=parameters,
        #                                                         cameraMatrix=self.matrix_coefficients,
        #                                                         distCoeff=self.distortion_coefficients)
        # if np.all(ids is not None):  # If there are markers found by detector
        #     # for i in range(0, len(ids)):  # Iterate in markers
        #     #     # Estimate pose of each marker and return the values rvec and tvec---different from camera coefficients
        #     #     rvec, tvec, markerPoints = aruco.estimatePoseSingleMarkers(corners, 0.0635, self.matrix_coefficients,
        #     #                                                                self.distortion_coefficients)
        #     #     (rvec - tvec).any()  # get rid of that nasty numpy value array error
        #     aruco.drawDetectedMarkers(cv_image, corners)  # Draw A square around the markers
        #         #aruco.drawAxis(cv_image, self.matrix_coefficients, self.distortion_coefficients, rvec, tvec, 0.01)  # Draw Axis
        # # Display the resulting cv_image
        # cv2.imshow('image', cv_image)
    except CvBridgeError as e:
      print(e)

def main(args):
  ic = image_converter()
  rospy.init_node('image_converter', anonymous=True)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
