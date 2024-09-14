#!/bin/bash

kubectl delete -f frontend-service.yaml -n fairpyx
kubectl delete -f frontend-deployment.yaml -n fairpyx
kubectl delete -f frontend-configmap.yaml -n fairpyx

