#!/bin/bash

kubectl create namespace fairpyx

# Setup backend
cd back
docker build -t backend-image .
sh start.sh

# Setup frontend
cd ../front
docker build -t frontned-image .
sh start.sh