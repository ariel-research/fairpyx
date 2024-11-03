#!/bin/sh

cd ..
uvicorn ${SERVICE_FOLDER_NAME}.main:app --host 0.0.0.0
