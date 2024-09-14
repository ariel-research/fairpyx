#!/bin/bash

kubectl create namespace fairpyx

# Setup backend
cd back
docker build -t backend-image .
cd deployment
sh start.sh

# Setup frontend
cd ../../front
docker build -t frontned-image .
cd deployment
sh start.sh