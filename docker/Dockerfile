FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu20.04 as builder

ENV DEBIAN_FRONTEND=noninteractive
# 기본 도구 설치
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3-dev \
        python3-pip \
        wget \
        git \
        build-essential \
        cmake \
        libopenblas-dev \
        && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /root

# oneDNN 빌드 및 설치
ENV ONEDNN_VERSION=3.1.1
RUN wget -q https://github.com/oneapi-src/oneDNN/archive/refs/tags/v${ONEDNN_VERSION}.tar.gz && \
    tar xf v${ONEDNN_VERSION}.tar.gz && \
    rm v${ONEDNN_VERSION}.tar.gz && \
    cd oneDNN-${ONEDNN_VERSION} && \
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DONEDNN_LIBRARY_TYPE=STATIC \
          -DONEDNN_BUILD_EXAMPLES=OFF \
          -DONEDNN_BUILD_TESTS=OFF \
          -DONEDNN_ENABLE_WORKLOAD=INFERENCE \
          -DONEDNN_ENABLE_PRIMITIVE="CONVOLUTION;REORDER" \
          -DONEDNN_BUILD_GRAPH=OFF . && \
    make -j$(nproc) install && \
    cd .. && rm -rf oneDNN-${ONEDNN_VERSION}

# OpenMPI 빌드 및 설치
ENV OPENMPI_VERSION=4.1.6
RUN wget -q https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-${OPENMPI_VERSION}.tar.bz2 && \
    tar xf openmpi-${OPENMPI_VERSION}.tar.bz2 && \
    rm openmpi-${OPENMPI_VERSION}.tar.bz2 && \
    cd openmpi-${OPENMPI_VERSION} && \
    ./configure && \
    make -j$(nproc) install && \
    cd .. && rm -rf openmpi-${OPENMPI_VERSION}

# CTranslate2 소스 복사
COPY third_party third_party
COPY cli cli
COPY include include
COPY src src
COPY cmake cmake
COPY python python
COPY CMakeLists.txt .

# 빌드
ENV CTRANSLATE2_ROOT=/opt/ctranslate2
ENV LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH}

RUN mkdir build_tmp && cd build_tmp && \
    cmake -DCMAKE_INSTALL_PREFIX=${CTRANSLATE2_ROOT} \
          -DWITH_CUDA=ON \
          -DCUDA_ARCH_LIST="7.5" \
          -DWITH_CUDNN=ON \
          -DWITH_MKL=OFF \
          -DWITH_DNNL=ON \
          -DOPENMP_RUNTIME=COMP \
          -DCMAKE_BUILD_TYPE=Release \
          -DWITH_TENSOR_PARALLEL=ON .. && \
    make -j$(nproc) install

# 파이썬 wheel 빌드
RUN apt update && apt install -y software-properties-common && add-apt-repository ppa:deadsnakes/ppa
RUN DEBIAN_FRONTEND=noninteractive apt install -y python3.9 python3.9-dev curl && \
  rm -rf /var/lib/apt/lists/*
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python3.9 get-pip.py && rm get-pip.py
RUN cd python && \
    python3.9 -m pip install --no-cache-dir -r install_requirements.txt && \
    python3.9 setup.py bdist_wheel --dist-dir $CTRANSLATE2_ROOT

# 런타임 이미지
FROM nvidia/cuda:12.2.2-base-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libcublas-12-2 \
        libcudnn8=8.9.7.29-1+cuda12.2 \
        libnccl2=2.19.3-1+cuda12.2 \
        libopenmpi3=4.0.3-0ubuntu1 \
        openmpi-bin \
        libgomp1 \
        software-properties-common \
        curl python3.9 python3.9-dev python3.9-distutils libsndfile1-dev && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get purge -y cuda-compat-12-2 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python3.9 get-pip.py && rm get-pip.py

ENV CTRANSLATE2_ROOT=/opt/ctranslate2
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CTRANSLATE2_ROOT/lib

COPY --from=builder $CTRANSLATE2_ROOT $CTRANSLATE2_ROOT

RUN python3.9 -m pip install --no-cache-dir $CTRANSLATE2_ROOT/*.whl

ENTRYPOINT ["/opt/ctranslate2/bin/ct2-translator"]

