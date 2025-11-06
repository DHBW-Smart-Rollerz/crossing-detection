# Crossing Detection Package

[![Build Test](https://github.com/DHBW-Smart-Rollerz/ros2_example_package/actions/workflows/build-test.yaml/badge.svg)](https://github.com/DHBW-Smart-Rollerz/ros2_example_package/actions/workflows/build-test.yaml)


## Usage

This package implements an intersection detector working on grayscale Cv2 images.

To build the package run the following command.

```bash
colcon build --symlink-install --packages-select my_package
```
To run the detector node, run:

```bash
ros2 run crossing_detection crossing_detection_node
```


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
