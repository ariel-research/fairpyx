#!/bin/bash

kubectl apply -f deployment.yaml -n fairpyx
kubectl apply -f service.yaml -n fairpyx
