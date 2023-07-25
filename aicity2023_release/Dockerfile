FROM nvidia/cuda:11.2.1-cudnn8-devel-ubuntu20.04

# Install system-level dependencies if needed

# Install and update pip
RUN apt-get update && \
    apt-get install -y python3-pip && \
    pip3 install --upgrade pip

# Copy environment.yaml file into the container
COPY environment.yaml /tmp/environment.yaml

# Set up conda environment and install packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget && \
    rm -rf /var/lib/apt/lists/* && \
    wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    /opt/conda/bin/conda update -n base -c defaults conda && \
    
    /opt/conda/bin/conda env create -f /tmp/environment.yaml && \
    /opt/conda/bin/conda clean -afy && \
    rm /tmp/environment.yaml

# Set the conda environment as the default
ENV PATH /opt/conda/envs/myenv/bin:$PATH

# Set the working directory
WORKDIR /home/code

# Copy any additional files or scripts into the container if needed

# Set the entrypoint or command for your container
CMD ["bash"]

