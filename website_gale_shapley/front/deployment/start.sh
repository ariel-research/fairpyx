#!/bin/bash

kubectl apply -f frontend-configmap.yaml -n fairpyx
kubectl apply -f frontend-deployment.yaml -n fairpyx
kubectl apply -f frontend-service.yaml -n fairpyx
