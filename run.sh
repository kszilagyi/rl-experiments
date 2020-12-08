#!/bin/bash
set -euo pipefail

JOB_SPEC_PATH=$1
TEST="${2:-}"
TEST_SUFFIX=''
if [ "$TEST" = '--test' ]; then
  TEST_SUFFIX="-test-$(openssl rand -hex 10)"
  echo '***Test mode***'
fi


echo "JOB_SPEC_PATH=$JOB_SPEC_PATH"
if [ ! "$TEST" = '--test' ]; then
  FILES=$(git status --porcelain)
  if [[ $FILES ]]; then
    echo "Uncommitted files"
    echo $FILES
    exit 1
  fi
fi

VERSION="$(git rev-parse --short=12 HEAD)$TEST_SUFFIX"
TAG=us.gcr.io/rl-experiments-296208/rl-experiments:$VERSION

if [[ $(docker image ls | grep $VERSION) ]]; then
  echo "WARNING!!!!"
  echo "Tag already exists! SKipping building!!!"
  echo "WARNING!!!!"
else
  echo "Building $TAG"
  docker build -t "$TAG" .
  echo "Pushing"
  docker push $TAG
  echo "Pushed $TAG"
fi

python -m src.submit_jobs --job_spec_path $JOB_SPEC_PATH --docker_image $TAG $TEST