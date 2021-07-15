#!/bin/bash

bin=`dirname "$0"`
export APP_HOME=`cd "$bin/../.."; pwd`
: ${DATA_DIR=/data}

docker stop finrl
docker run -it --rm -v ${APP_HOME}:/home -v${DATA_DIR}:/data \
    -d \
    --network host \
    --name finrl \
    registry.cn-hangzhou.aliyuncs.com/all3n/finrl

HOSTNAME=$(hostname)
echo "start notebook http://$HOSTNAME:8888"
