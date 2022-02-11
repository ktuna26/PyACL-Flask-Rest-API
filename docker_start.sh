# Docker Start Comand 
# Copyright 2022 Huawei Technologies Co., Ltd
# 
# Usage:
#   $ sudo docker_start.sh pyacl_flask_rest_api:1.0 <app_cfg>
# 
# CREATED:  2021-11-07 15:12:13
# MODIFIED: 2021-02-07 16:48:45

#!/usr/bin/env bash
docker_image=$1
app_cfg=$(find $PWD -type f | grep "$2")

# read port from app_user.cfg file
port=$((cut -d "=" -f2 <<< $(cat ${app_cfg} | grep port))| sed 's/ //g')

docker run -it \
        -p $port:$port \
        --rm \
        --ipc=host \
        --device=/dev/davinci0 \
        --device=/dev/davinci_manager \
        --device=/dev/devmm_svm \
        --device=/dev/hisi_hdc \
        -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
        -v /usr/local/Ascend/add-ons/:/usr/local/Ascend/add-ons/ \
        -v /var/log/npu/conf/slog/slog.conf:/var/log/npu/conf/slog/slog.conf \
        -v /var/log/npu/slog/:/var/log/npu/slog \
        -v /var/log/npu/profiling/:/var/log/npu/profiling \
        -v /var/log/npu/dump/:/var/log/npu/dump \
        -v /var/log/npu/:/usr/slog \
        -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
        -v ${app_user_cfg}:/pyacl_flask_rest_api/data/app_user.cfg \
        ${docker_image}