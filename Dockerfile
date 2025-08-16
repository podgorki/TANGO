# Stage 1: Base environment with PyTorch and CUDA
FROM pytorch/pytorch:2.2.2-cuda11.8-cudnn8-devel AS base

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    build-essential \
    cmake \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglu1-mesa-dev \
    libegl1-mesa-dev \
    libjpeg-dev \
    libglm-dev \
    mesa-utils \
    xorg-dev \
    freeglut3-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set up python3.10 as default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Upgrade pip and install cmake version required for habitat-sim
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir cmake==3.14.4
RUN pip install --no-cache-dir "numpy>=1.25,<2"

# Stage 2: Build habitat-sim
FROM base AS habitat-sim

WORKDIR /app

RUN mkdir -p third-party && cd third-party/ && \
    git clone https://github.com/facebookresearch/habitat-sim.git && \
    cd habitat-sim/ && \
    git checkout v0.2.4 && \
    CXXFLAGS="-march=haswell" python setup.py install --cmake && \
    cd ../..

# Stage 3: Install system Python dependencies
FROM habitat-sim AS dependencies

# Install remaining Python packages (PyTorch already included in base)
RUN pip install --no-cache-dir \
    opencv-python \
    matplotlib \
    pyyaml \
    networkx \
    h5py \
    natsort \
    spatialmath-python \
    kornia \
    einops \
    ultralytics \
    numpy-quaternion \
    gitpython \
    imageio \
    imageio-ffmpeg \
    numba

# Stage 4: Final application image
FROM dependencies AS app

# Copy git files and initialize submodules
COPY .git/ ./.git/
COPY .gitmodules ./
RUN git submodule update --init --recursive

# Copy all application files
COPY pyproject.toml ./
COPY Readme.md ./
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY configs/ ./configs/
COPY third_party/ ./third_party/

# Install the project in editable mode with habitat-lab
RUN pip install --no-cache-dir -e ".[habitat-lab]" --extra-index-url https://download.pytorch.org/whl/cu118 --prefer-binary

# Add zoedepth path for Depth Anything
RUN echo "/app/third_party/depth_anything/metric_depth" > \
     $(python -c "import site, sys; print(site.getsitepackages()[0])")/zoedepth_local.pth

# Create directories for external mounts
RUN mkdir -p /app/logs \
    && mkdir -p /app/outputs \
    && mkdir -p /app/models \
    && mkdir -p /app/data

# Set environment variables
ENV PYTHONPATH=/app
ENV MAGNUM_LOG=quiet
ENV HABITAT_SIM_LOG=quiet
ENV CUDA_VISIBLE_DEVICES=0

# Default command
CMD ["python", "scripts/run_goal_control_demo.py", "--config_file", "configs/tango_default.yaml"]