#!/bin/bash
echo "Starting my Docker container..."

cd Twin4Build

python3 twin4build/api/codes/ml_layer/simulator_api.py

echo "API is running..."