import numpy as np
import rospy
import cv2
import sys
import cv2.aruco as aruco

from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from autolab_core import Point, CameraIntrinsics

def get_object_center_point_in_world(image_x, image_y, depth_image, intrinsics):

    object_center = Point(np.array([image_x, image_y]), 'azure_kinect_overhead')
    object_depth = depth_image[image_y, image_x]

    return intrinsics.deproject_pixel(object_depth, object_center)

print(cv2.__version__)


AZURE_KINECT_INTRINSICS = 'azure_kinect_intrinsics.intr'

class image_converter:

  def __init__(self):

    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber('/rgb/image_rect_color', Image, self.callback)
    self.matrix_coefficients = np.array([1943.6480712890625, 0.0, 2045.5838623046875, 0.0, 1943.1328125, 1576.270751953125, 0.0, 0.0, 1.0])
    self.distortion_coefficients = np.array([0, 0, 0, 0, 0])
    self.intrinsics = CameraIntrinsics.load(AZURE_KINECT_INTRINSICS)

  def callback(self,data):
    try:
        cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
        #bgr_image = cv2.cvtColor(cv_image, cv2.COLOR_BGRA2BGR)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)  # Change grayscale
        dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)  # Use 5x5 dictionary to find markers
        parameters = aruco.DetectorParameters()  # Marker detection parameters
        detector = aruco.ArucoDetector(dictionary, parameters)

        corners, ids, rejected_img_points = detector.detectMarkers(gray)
        depth_image_message = rospy.wait_for_message('/depth_to_rgb/hw_registered/image_rect', Image)
        depth_image = self.bridge.imgmsg_to_cv2(depth_image_message, desired_encoding='passthrough')

        print(ids)

        print(corners)

        centers = []

        for corner in corners:
            centers.append(np.rint(np.mean(corner.reshape((4,2)), axis=0)).astype(int))


        center_points_in_world = []
        for center in centers:
            center_point_in_world = get_object_center_point_in_world(center[0], center[1], depth_image, self.intrinsics)
            center_points_in_world.append([center_point_in_world.x, center_point_in_world.y, center_point_in_world.z])
        print(center_points_in_world)

        # print(np.mean(np.array(center_points_in_world),axis=0))


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
