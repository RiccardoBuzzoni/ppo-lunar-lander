# --- Base image ---
# python:3.11-slim: a minimal Debian-based image with Python 3.11
# preinstalled. "slim" keeps the image small (no compilers or docs);
# we add back only what is actually needed below. Matches the Python
# version used in the local conda environment, for reproducibility.
FROM python:3.11-slim

# --- System dependencies ---
# box2d-py (pulled in by gymnasium[box2d]) is compiled from source on
# most platforms, and needs SWIG plus a C/C++ build toolchain, the same
# class of issue you hit locally with pygame/SDL2 on WSL2.
# ffmpeg is optional, only needed if you use evaluate.py --render to
# record videos inside the container.
RUN apt-get update && apt-get install -y --no-install-recommends \
    swig \
    build-essential \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*
# "rm -rf /var/lib/apt/lists/*" deletes apt's package index after
# installing; it is not needed at runtime and just adds dead weight
# to the image layer.

# --- Working directory ---
# All subsequent instructions (COPY, RUN, CMD) run relative to this
# path inside the container. It also becomes the container's default
# directory when you start a shell in it.
WORKDIR /app

# --- Dependencies layer (cached separately from source code) ---
# Copying ONLY requirements.txt first, then installing, means Docker
# can reuse this (slow) layer on rebuilds as long as requirements.txt
# hasn't changed, even if you've edited train.py since.
# If we copied the whole project before running pip install, every
# code change would invalidate the cache and force a full reinstall.
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

# --- Source code layer ---
# This changes often during development, so it's copied last,
# keeping the expensive dependency layer above untouched by cache.
COPY src/ ./src/
WORKDIR /app/src

# --- Default command ---
# Runs when the container starts with no other command specified.
# Easy to override at runtime, e.g.:
#   docker run <image> python evaluate.py --episodes 20
CMD ["python", "train.py"]