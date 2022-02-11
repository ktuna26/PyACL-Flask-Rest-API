# Server Start Comand 
# Copyright 2022 Huawei Technologies Co., Ltd
# 
# Usage:
#   $ server_start.sh <image_name> <app_user_cfg>
# 
# CREATED:  2021-11-07 15:12:13
# MODIFIED: 2021-12-07 16:48:45

#!/usr/bin/env bash
set -e

exec python3 pyacl_app.py &
exec python3 flask_app.py 