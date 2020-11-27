#!/bin/bash
set -euo pipefail

JOB_SPEC_PATH=$1
echo "JOB_SPEC_PATH=$JOB_SPEC_PATH"
FILES=$(git status --porcelain)
if [[ $FILES ]]; then
  echo "Uncommitted files"
  echo $FILES
  exit 1
fi

VERSION=$(git rev-parse --short=12 HEAD)
if [[ $(docker image ls | grep $VERSION) ]]; then
  echo "WARNING!!!!"
  echo "Tag already exists! SKipping building!!!"
  echo "WARNING!!!!"
else
  TAG=us.gcr.io/rl-experiments-296208/rl-experiments:$VERSION
  echo "Building $TAG"
  docker build -t "$TAG" .
  echo "Pushing"
  docker push $TAG
  echo "Pushed $TAG"
fi

python -m src.submit_jobs --job_spec_path $JOB_SPEC_PATH --docker_image $TAG