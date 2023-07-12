#!/usr/bin/env bash
docker build -t hlcv_lens:v$1 .

docker image tag hlcv_lens:v$1 felixceard/hlcv_lens:v$1

docker push felixceard/hlcv_lens:v$1