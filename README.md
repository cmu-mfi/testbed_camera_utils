# testbed_camera_utils

## Installation Instructions
1. Install autolab_core, pyyaml, rospkg
    ```bash
    pip install autolab_core pyyaml rospkg
    ```
2. Clone into a catkin_ws
    ```bash
    git clone https://github.com/cmu-mfi/testbed_lin_actuator.git
    ```
3. Catkin build
    ```bash
    catkin build
    ```

## Running Instructions
1. Mount the Aruco marker to the force torque sensor in the +x direction
2. Start the cameras using roslaunch yk_god.launch
3. Run the calibrate_camera launch file and specify the type of camera (azure_kinect or realsense) and the namespace
    ```bash
    roslaunch testbed_camera_utils calibrate_camera.launch camera_type:=azure_kinect namespace:=yk_builder
    ```
4. Detect amrs by running the detect_amr.alunch file and specify the namespace
    ```bash
    roslaunch testbed_camera_utils detect_amr.launch namespace:=yk_builder
    ``` 
