#!/bin/bash

GIT_BRANCH=$(git symbolic-ref --short -q HEAD)
GIT_HEAD=$(git rev-parse --short HEAD)
GIT_TAG="$(git describe --exact-match --tags HEAD)"

# if no tag, don't build
if [[ ! $? == 0 || $GIT_TAG == fatal* ]]; then
    echo Tag not found, exiting
    exit 0
fi

# image details
REGISTRY=cmiauditor.azurecr.io
IMAGE_NAME=yolov5

DATE=`date "+%Y%m%d.%H%M%S"`
IMAGE_TAG=$DATE-$GIT_TAG.$GIT_HEAD
echo "building" $IMAGE_NAME:$IMAGE_TAG
docker build --build-arg GIT_COMMIT=$GIT_HEAD -t $IMAGE_NAME:$IMAGE_TAG .
docker tag $IMAGE_NAME:$IMAGE_TAG $REGISTRY/$IMAGE_NAME:$IMAGE_TAG
docker push $REGISTRY/$IMAGE_NAME:$IMAGE_TAG
