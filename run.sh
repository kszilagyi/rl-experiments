#!/bin/bash
set -euo pipefail

FILES=$(git status --porcelain)
if [[ $FILES ]]; then
  echo "Uncommitted files"
  echo $FILES
  exit 1
fi

VERSION=$(git rev-parse --short=12 HEAD)
if [[ $(docker image ls | grep $VERSION) ]]; then
  echo "WARNING!!!!"
  echo "Tag already exists!"
  echo "WARNING!!!!"
  exit 1
fi

TAG=us.gcr.io/rl-experiments-296208/rl-experiments:$VERSION
echo "Building $TAG"
docker build -t "$TAG"
echo "Pushing"
docker push $TAG
echo "Pushed $TAG"

