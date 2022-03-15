FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime

# Avoiding user interaction with libopencv-dev
ENV DEBIAN_FRONTEND=noninteractive

RUN apt update -y && apt install -y \
    git
RUN apt-get update && apt-get install -y \
    build-essential \
    libsndfile1 \
    vim \
    libopencv-dev

# Install python package.
ADD requirements.txt  /tmp/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir mmcv-full==1.4.6 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html && \
    pip install --no-cache-dir openmim==0.1.5 && \
    pip install -r /tmp/requirements.txt

# Language settings
ENV LANG C.UTF-8
ENV LANGUAGE en_US
