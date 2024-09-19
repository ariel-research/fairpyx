#!/bin/bash

docker stop front-container
docker rm front-container

docker stop back-container
docker rm back-container