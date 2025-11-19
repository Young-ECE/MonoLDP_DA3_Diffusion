# MonoLDP Docker Image for Training
# Supports GPU training with CUDA

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3.8-dev \
    python3-pip \
    git \
    wget \
    curl \
    vim \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python
RUN ln -s /usr/bin/python3.8 /usr/bin/python && \
    ln -s /usr/bin/pip3 /usr/bin/pip

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch with CUDA 11.8 support
RUN pip install --no-cache-dir \
    torch==1.13.1+cu117 \
    torchvision==0.14.1+cu117 \
    --extra-index-url https://download.pytorch.org/whl/cu117

# Copy requirements file
COPY requirements.txt /workspace/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install Depth Anything V3 from GitHub
# Note: Set HuggingFace mirror for faster download in China
ENV HF_ENDPOINT=https://hf-mirror.com
RUN pip install --no-cache-dir git+https://github.com/ByteDance-Seed/Depth-Anything-3.git

# Copy project files
COPY . /workspace/

# Create directories for data and logs
RUN mkdir -p /workspace/data /workspace/logs /workspace/models

# Set default working directory to DepthEstimation
WORKDIR /workspace/DepthEstimation

# Expose TensorBoard port (optional)
EXPOSE 6006

# Default command (can be overridden)
CMD ["/bin/bash"]

