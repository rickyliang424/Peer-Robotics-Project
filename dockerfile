FROM ros:humble
WORKDIR /home/src
SHELL ["/bin/bash", "-c"]

# Install the application dependencies
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    ros-humble-cv-bridge \
    ros-humble-rviz2 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment (note: activation must be sourced at runtime)
RUN python3 -m venv /home/venv
RUN /home/venv/bin/pip install --no-cache-dir "numpy<2.0" "ultralytics==8.3.109"

# Auto-source ROS + activate venv in all future shells
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc && \
    echo "source /home/venv/bin/activate" >> ~/.bashrc

# Copy code into container
COPY models/ /home/src/models
COPY prediction/InferenceNode.py /home/src
COPY prediction/config.rviz /home/src

# Default shell
CMD ["bash"]
