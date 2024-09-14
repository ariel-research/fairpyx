#!/bin/bash

kubectl delete -f service.yaml -n fairpyx
kubectl delete -f deployment.yaml -n fairpyx
