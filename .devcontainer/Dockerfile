FROM osrf/ros:noetic-desktop-full

# Set DEBIAN_FRONTEND to noninteractive
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git \
    python3-pip \
    cmake \
    build-essential \
    openssh-client \
    libeigen3-dev \
    libboost-all-dev \
    libopencv-dev \
    libceres-dev \
    libgstreamer1.0-dev \
    python3-catkin-tools \
    python3-osrf-pycommon \
    wget \
    gnupg2 \
    protobuf-compiler \
    libprotoc-dev \
    && rm -rf /var/lib/apt/lists/*


RUN pip3 install gitman

# Install CUDA 11.8
RUN wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb && \
    dpkg -i cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb && \
    cp /var/cuda-repo-ubuntu2004-11-8-local/cuda-368EAC11-keyring.gpg /usr/share/keyrings/ && \
    apt-get update && \
    apt-get -y install cuda

# Set environment variables for CUDA
RUN echo "export CUDA_HOME=/usr/local/cuda-11.8\n\
export PATH=$PATH:$CUDA_HOME/bin\n\
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64\n\
" >> ~/.bashrc 

RUN ["/bin/bash", "-c", "source ~/.bashrc"]

# Install cuDNN for CUDA 11.8 using .deb package
RUN wget https://developer.download.nvidia.com/compute/redist/cudnn/v8.8.0/local_installers/11.8/cudnn-local-repo-ubuntu2004-8.8.0.121_1.0-1_amd64.deb && \
    dpkg -i cudnn-local-repo-ubuntu2004-8.8.0.121_1.0-1_amd64.deb && \
    cp /var/cudnn-local-repo-ubuntu2004-8.8.0.121/cudnn-local-B70907B4-keyring.gpg /usr/share/keyrings/ && \
    apt-get update && \
    apt-get -y install libcudnn8=8.8.0.121-1+cuda11.8 libcudnn8-dev=8.8.0.121-1+cuda11.8

# Install ONNX Runtime
RUN mkdir /opt/tmp/ && \
    cd /opt/tmp && \
    wget https://github.com/microsoft/onnxruntime/releases/download/v1.18.0/onnxruntime-linux-x64-gpu-1.18.0.tgz && \
    tar -xzf onnxruntime-linux-x64-gpu-1.18.0.tgz && \
    mv onnxruntime-linux-x64-gpu-1.18.0 /usr/local && \
    ln -sf /usr/local/onnxruntime-linux-x64-gpu-1.18.0  /usr/local/onnxruntime && \
    cd .. && \
    rm -r tmp

RUN echo "export ONNXRUNTIME_DIR=/usr/local/onnxruntime-linux-x64-gpu-1.18.0\n\
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/onnxruntime-linux-x64-gpu-1.18.0/lib\n\
export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:/usr/local/onnxruntime-linux-x64-gpu-1.18.0/include\n\
source /opt/ros/noetic/setup.bash\n\
source /opt/baza/devel/setup.bash\n\
export LIBGL_ALWAYS_SOFTWARE=1\n\
" >> ~/.bashrc 

WORKDIR /opt/baza

CMD ["/bin/bash"]
