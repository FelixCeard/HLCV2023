#!/usr/bin/env bash
docker build -t gen_attr:v$1 .

docker image tag gen_attr:v$1 felixceard/gen_attr:v$1

docker push felixceard/gen_attr:v$1