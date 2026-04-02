FROM ubuntu:24.04

ARG CUDA_VERSION=11.8.0
ENV CUDA_VERSION=${CUDA_VERSION}
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ca-certificates wget git curl \
       build-essential gcc-11 g++-11 \
       libgl1-mesa-dev libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL -o /tmp/miniconda.sh \
        https://repo.anaconda.com/miniconda/Miniconda3-py311_25.1.1-2-Linux-x86_64.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh
ENV PATH=/opt/conda/bin:$PATH
RUN conda init bash

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
ENV FORCE_CUDA=1

WORKDIR /workspace
COPY . .

RUN bash install_env.sh fullcircle WITH_GCC11

RUN echo "conda activate fullcircle" >> ~/.bashrc
