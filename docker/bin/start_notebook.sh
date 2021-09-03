#!/bin/bash

bin=`dirname "$0"`
export APP_HOME=`cd "$bin/../.."; pwd`
: ${DATA_DIR=/data}

docker stop finrl
which nvidia-smi > /dev/null
DOCKER=docker
if [[ -$? -eq 0 ]];then
    DOCKER=nvidia-docker
fi
$DOCKER run -it --rm -v ${APP_HOME}:/src/FinRL-Library -w /src -v${DATA_DIR}:/data \
    -d \
    --network host \
    --name finrl \
    registry.cn-hangzhou.aliyuncs.com/all3n/finrl

HOSTNAME=$(hostname)
echo "start notebook http://$HOSTNAME:8888"
