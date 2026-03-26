# Crossing Detection Package

[![Build Test](https://github.com/DHBW-Smart-Rollerz/ros2_example_package/actions/workflows/build-test.yaml/badge.svg)](https://github.com/DHBW-Smart-Rollerz/ros2_example_package/actions/workflows/build-test.yaml)


## Usage

This package implements an intersection detector working on grayscale Cv2 images.

To build the package run the following command.

```bash
source smarty_workspace/scripts/build.sh
```
To run the detector node, run:

Calibrate cam:
```bash
ros2 run camera_preprocessing camera_calibration_node --ros-args -p chessboard_path:='/home/smartrollerz/Downloads/chessboard_crossing.png'

ros2 run camera_preprocessing camera_calibration_node --ros-args -p chessboard_path:='/home/smartrollerz/Downloads/chessboard_crossing.png' -p calibration_images_path:='/home/smartrollerz/Desktop/smartrollers/smarty_workspace/src/camera_preprocessing/img/calib/new_lens'

ros2 run camera_preprocessing camera_calibration_node --ros-args -p chessboard_path:='/home/smartrollerz/Downloads/crossing-bag-new/ROS_Bags_2026-03-12-crossing/chessboard.png' -p calibration_images_path:='/home/smartrollerz/Desktop/smartrollers/smarty_workspace/src/camera_preprocessing/img/calib/new_lens'


```

Run the preprocessing node:
```bash
ros2 run camera_preprocessing camera_preprocessing_node --remap /camera/image_raw:=/camera/image/raw
```

Run the bag:
```bash
ros2 bag play /home/smartrollerz/Downloads/crossing/rosbag2_2025_11_11-17_04_16_0.mcap
ros2 bag play /path/to/bag

ros2 bag play /home/smartrollerz/Downloads/crossing-bag-new/ROS_Bags_2026-03-12-crossing/rosbag2_2026_03_12-14_10_34/rosbag2_2026_03_12-14_10_34_0.mcap

```

Run the crossing node:
```bash
ros2 run crossing_detection crossing_detection_node

# to run with debug config
ros2 launch crossing_detection crossing_detection.launch.py config_file:=$(ros2 pkg prefix crossing_detection)/share/crossing_detection/config/debug.crossing_detection.yaml
# for competition config
ros2 launch crossing_detection crossing_detection.launch.py

```

Die crossing detection braucht "scikit-learn". Also `pip install scikit-learn`.



## Structure

- `config/`: All configurations (most of the time yaml files)
- `launch/`: Contains all launch files. Launch files can start multiple nodes with yaml-configurations
- `models/`: Contains all models (optional) and only necessary for machine learning nodes
- `resource/`: Contains the package name (required to build with colcon)
- `crossing_detection`: Contains all nodes and sources for the ros package
- `test/`: Contains all tests
- `package.xml`: Contains metadata about the package
- `setup.py`: Used for Python package configuration
- `setup.cfg`: Additional configuration for the package
- `requirements.txt`: Python dependencies

## Contributing

Thank you for considering contributing to this repository! Here are a few guidelines to get you started:

1. Fork the repository and clone it locally.
2. Create a new branch for your contribution.
3. Make your changes and ensure they are properly tested.
4. Commit your changes and push them to your forked repository.
5. Submit a pull request with a clear description of your changes.

We appreciate your contributions and look forward to reviewing them!

## License

This repository is licensed under the MIT license. See [LICENSE](LICENSE) for details.

## Author

Reinhold Brant (reinhold.bra5@gmail.com) for SmartRollerz.
