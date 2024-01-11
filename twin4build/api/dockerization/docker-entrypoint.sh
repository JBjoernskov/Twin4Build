#!/bin/bash
echo "Starting twin4build Docker container..."

cd /Twin4Build

pip install -r requirements_linux_no_version.txt

#absolute path of the resulting repo:
cd /Twin4Build/twin4build/api/codes/ml_layer

python3 simulator_api.py

# Alternatively, if the script needs to run in the background, use:
# python3 simulator_api.py &

# If you need the container to stay alive after the script execution,
# you can add a command like 'tail -f /dev/null' at the end of this script.