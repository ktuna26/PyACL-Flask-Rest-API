# Dockerfile
# Copyright 2022 Huawei Technologies Co., Ltd
# 
# Usage:
#   $ sudo docker build -t pyacl_flask_rest_api:1.0 \
#                       --build-arg NNRT_PKG=Ascend-cann-nnrt_5.0.2_linux-x86_64.run .
# 
# CREATED:  2021-11-24 15:12:13
# MODIFIED: 2022-02-10 16:48:45


#OS and version number. Change them based on the site requirements.
FROM python:3.7.5-slim

# Set the parameters of the offline inference engine package.
ARG NNRT_PKG

# Set environment variables.
ARG ASCEND_BASE=/usr/local/Ascend
ENV LD_LIBRARY_PATH=\
$ASCEND_BASE/driver/lib64:\
$ASCEND_BASE/driver/lib64/common:\
$ASCEND_BASE/driver/lib64/driver:\
$ASCEND_BASE/nnrt/latest/acllib/lib64:\
$LD_LIBRARY_PATH
ENV PYTHONPATH=$ASCEND_BASE/nnrt/latest/pyACL/python/site-packages/acl:\
$PYTHONPATH
ENV ASCEND_OPP_PATH=$ASCEND_BASE/nnrt/latest/opp
ENV ASCEND_AICPU_PATH=\
$ASCEND_BASE/nnrt/latest/x86_64-linux
RUN echo $LD_LIBRARY_PATH && \
    echo $PYTHONPATH && \
    echo $ASCEND_OPP_PATH &&\
    echo $ASCEND_AICPU_PATH

# Copy the offline inference engine package.
COPY $NNRT_PKG .

# Install the offline inference engine package.
RUN umask 0022 && \
    groupadd -g 183426 HwHiAiUser && \
    useradd -g HwHiAiUser -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash && \
    usermod -u 183426 HwHiAiUser && \
    chmod +x ${NNRT_PKG} && \
    ./${NNRT_PKG} --quiet --install && \
    rm ${NNRT_PKG} && \
    . /usr/local/Ascend/nnrt/set_env.sh

# set workdir
WORKDIR /pyacl_flask_rest_api

# copy the project into docker image
COPY * /pyacl_flask_rest_api

# install the necessary package
RUN cd /pyacl_flask_rest_api && \
    python3 -m pip install --upgrade pip && \
    pip3 install -r requirements.txt && \
    apt-get update -y && \
    apt-get install redis-server -y

# set a port
EXPOSE 8500

# run the api
ENTRYPOINT ["server_run.sh"]