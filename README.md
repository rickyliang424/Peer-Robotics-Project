# Peer Robotics Project

The objective of this assignment is to develop a pallet detection & ground segmentation application in ROS2 for a manufacturing or warehousing environment. The solution is optimized for deployment on edge devices like the NVIDIA Jetson AGX Orin, ensuring real-time performance suitable for mobile robotics applications.

## 1. Demo

Download the rosbag from the [link](https://drive.google.com/drive/folders/1xSqKa55QrNGufLRQZAbp0KFGYr9ecqgT?usp=sharing)

Open a new terminal and run the rosbag:
``` bash
ros2 bag play -l ./bag/bag.db3
```

In a new terminal, cd to `prediction` folder.
``` bash
python3 ./InferenceNode.py
```

Then open another terminal, open visualization tool:
``` bash
rviz2 -d ./config.rviz
```

## 2. Docker

Build docker image:
``` bash
docker build -t peer_robotics .
```

Run the docker image, copy and play ros2 bag
``` bash
docker run -it --rm --gpus all --net=host --env="DISPLAY" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" peer_robotics
docker cp path/to/bag ${container_id}:/home/bag
ros2 bag play -l /home/bag/bag.db3
```

Open a new terminal, launch Python node in background
``` bash
docker exec -it ${containe_id} /bin/bash
python3 /home/src/InferenceNode.py
```

Open a new terminal, launch RViz2 in foreground (blocking)
``` bash
docker exec -it ${containe_id} /bin/bash
rviz2 -d config.rviz
```

Result:
<div align="center">
  <img src="asset/demo_0.gif" width="40%" style="display: inline-block; margin-right: 5%;">
  <img src="asset/demo_1.gif" width="40%" style="display: inline-block;">
</div>

## Install

This project was developed under Ubuntu 20.04 and ROS2 foxy environment with Python 3.8.10. 

Check the following ROS2 Dependencies have installed:
``` bash
sudo apt install ros-${ROS_DISTRO}-rclpy ros-${ROS_DISTRO}-cv-bridge
```

Install the required packages:
``` bash
pip install - r requirements.txt
```

## Data preprocessing

1. Download dataset with annotations from this [page](https://github.com/tum-fml/loco) and extract to `data_preprocess`.
2. Run `python3 update_dataset.py` to flattern folder and update json file (only keep class "Pallet").
3. To get the bounding boxes of class "Ground", I used [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) for object detection. After install it, run `inference_on_images.py`.
4. Run `python3 create_txt.py` to generate label files for class "Pallet".
5. Run `python3 merge_txt.py` to combine labels of "Ground" and "Pallet".
6. Run `python3 split_dataset.py` to split training and validation data.
7. The dataset should be processed. To check there is no error, run `python3 show_annot.py` for visualization.

## Training

- Follow `detection/train.ipynb` to train the YOLO model.
- The training results and weight files are stored in `detection/training_res` folder.
- `prediction/optimization.py` converts `.pt` models to `.onnx` and `.engine` (TenserRT format) with quantization, and it shows the inference time of different models.

