# Use NVIDIA CUDA 12.1 base image
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=UTF-8

# Install Python 3.10 (Ubuntu 22.04 native — Soup supports 3.9+).
# Avoids deadsnakes PPA which has had repeated connection timeouts from GHA runners.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-dev \
    python3-venv \
    python3-pip \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Alias `python` -> `python3`
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# Set working directory
WORKDIR /workspace

# Install Soup from PyPI (always the latest published release, not local source)
RUN pip install --no-cache-dir "soup-cli[serve,data,eval]"

# Default entrypoint and command
ENTRYPOINT ["soup"]
CMD ["--help"]
