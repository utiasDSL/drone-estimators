# estimators
Drone state estimators @ LSY


[![Python Version]][Python Version URL] [![Ruff Check]][Ruff Check URL] [![Tests]][Tests URL]

[Python Version]: https://img.shields.io/badge/python-3.10+-blue.svg
[Python Version URL]: https://www.python.org

[Ruff Check]: https://github.com/utiasDSL/estimators/actions/workflows/ruff.yml/badge.svg?style=flat-square
[Ruff Check URL]: https://github.com/utiasDSL/estimators/actions/workflows/ruff.yml

[Tests]: https://github.com/utiasDSL/estimators/actions/workflows/testing.yml/badge.svg
[Tests URL]: https://github.com/utiasDSL/estimators/actions/workflows/testing.yml

## Installation
Clone repository

`pip install -e .`

## Usage
Either use the estimators directly:

`from lsy_estimators.estimator import KalmanFilter`

or run the ROS2 node with:

`python lsy_estimators/ros_nodes/ros2_node.py`

For the latter, you need to add all drones you want to estimate to the `estimators.toml` file, or create your own file and call

`python lsy_estimators/ros_nodes/ros2_node.py --settings <your_estimators.toml>`