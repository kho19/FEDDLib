########
# base #
########
FROM ubuntu:latest AS base

ENV TZ=Europe \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    rm -rf /var/lib/apt/lists/*

RUN add-apt-repository universe
RUN apt-get -y update && apt-get -y install build-essential wget ca-certificates coreutils curl environment-modules gfortran git gpg lsb-release python3 unzip zip cmake openmpi-bin openmpi-common libopenmpi-dev libparmetis-dev libboost-all-dev libhdf5-openmpi-dev libsuitesparse-dev
RUN apt-get -y update && apt-get -y install "liblapack*" "libblas*"
RUN apt-get -y update && apt-get -y install "openmpi*" "libopenmpi*"
############
# trilinos #
############
FROM base AS trilinos

ARG TRILINOS_CONFIGFILE=configure/do-configure-trilinos.sh
ARG TRILINOS_VERSION=15-1-0

WORKDIR /opt/trilinos
COPY $TRILINOS_CONFIGFILE /opt/trilinos/do-configure-trilinos.sh
RUN /bin/bash -xc "\
    wget -nv https://github.com/trilinos/Trilinos/archive/refs/tags/trilinos-release-$TRILINOS_VERSION.tar.gz; \
    mkdir source; \
    tar xfz /opt/trilinos/trilinos-release-$TRILINOS_VERSION.tar.gz -C /opt/trilinos/source --strip-components=1; \
    rm -f /opt/trilinos/trilinos-release-$TRILINOS_VERSION.tar.gz; \
    mkdir build; \
    pushd build; \
    ../do-configure-trilinos.sh; \ 
    popd; \
    rm do-configure-trilinos.sh; \
    cmake --build build --parallel 6; \ 
    cmake --install build --prefix .; \
    rm -rf source; \
    rm -rf build; \
    "

FROM trilinos AS tester

# TODO: kho create a non-root user
# RUN useradd -ms /bin/bash tester
# USER newuser

WORKDIR /usr/src

# Copy any source file(s) required for the action
COPY entrypoint.sh .

# Configure the container to be run as an executable
ENTRYPOINT ["/usr/src/entrypoint.sh"]
