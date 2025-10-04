# Dockerfile for COLMAP Pipeline
# Provides a reproducible environment with COLMAP and all dependencies

FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    build-essential \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libeigen3-dev \
    libflann-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libsqlite3-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libceres-dev \
    python3 \
    python3-pip \
    python3-dev \
    ffmpeg \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Build and install COLMAP
RUN git clone https://github.com/colmap/colmap.git /tmp/colmap && \
    cd /tmp/colmap && \
    git checkout 3.8 && \
    mkdir build && \
    cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release \
             -DCMAKE_INSTALL_PREFIX=/usr/local \
             -DTESTS_ENABLED=OFF \
             -DGUI_ENABLED=OFF && \
    make -j$(nproc) && \
    make install && \
    cd / && \
    rm -rf /tmp/colmap

# Install Python packages
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# Copy pipeline code
WORKDIR /app
COPY . /app/

# Install pipeline package
RUN pip3 install -e .

# Create directories for data
RUN mkdir -p /data/input /data/output

# Set working directory
WORKDIR /data

# Entry point
ENTRYPOINT ["colmap-reconstruct"]
CMD ["--help"]

