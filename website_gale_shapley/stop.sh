#!/bin/bash

cd front/deployment
docker stop front-container
docker rm front-container

cd ../../back/deployment
docker stop back-container
docker rm back-container