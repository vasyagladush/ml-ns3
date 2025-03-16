FROM ubuntu:22.04

# Update apt repositories
RUN apt-get update

# Install minimal requirements for C++ and other dependencies for ns-3 and ns3-gym
RUN apt-get install -y \
    gcc \
    g++ \
    python3 \
    python3-pip \
    cmake \
    wget \
    tar \
    git \
    libzmq5 \
    libzmq3-dev \
    libprotobuf-dev \
    protobuf-compiler \
    pkg-config

# Download and install ns3
RUN wget https://www.nsnam.org/releases/ns-allinone-3.40.tar.bz2 && \
    tar xf ns-allinone-3.40.tar.bz2 && \
    rm ns-allinone-3.40.tar.bz2

WORKDIR /ns-allinone-3.40/ns-3.40

# Clone ns3-gym repository into contrib directory
RUN mkdir contrib && cd contrib && \
    git clone https://github.com/tkn-tub/ns3-gym.git opengym && \
    cd opengym && \
    git checkout app-ns-3.36+

# Configure and build ns-3 project
RUN ./ns3 configure --enable-examples && \
    ./ns3 build

# Install ns3gym python package
WORKDIR /ns-allinone-3.40/ns-3.40/contrib/opengym
RUN python3 -m venv ns3gym-venv && \
    source ./ns3gym-venv/bin/activate && \
    pip3 install --no-cache-dir ./model/ns3gym

# Set working directory for example execution
WORKDIR /ns-allinone-3.40/ns-3.40/contrib/opengym/examples/opengym

# Command to run the example (simple_test.py) when the container starts
CMD ["./simple_test.py"]

# If you want to run interactively or another example, you can override the CMD when running the container.
# For example: docker run -it --rm your-ns3-gym-image bash