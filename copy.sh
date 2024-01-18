#!/bin/bash
# Assign the variables.
# scp -rp jabj@cei-datamining.sandbox.tek.sdu.dk:/home/jabj/Twin4Build/twin4build/estimator/tests/generated_files/model_parameters/chain_logs/ "C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/python/BuildingEnergyModel/remote_results/chain_logs/";
USER='jabj';
HOST='cei-datamining.sandbox.tek.sdu.dk';
LOCAL_PATH='C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/python/BuildingEnergyModel/remote_results/chain_logs/';
REMOTE_PATH='/home/jabj/Twin4Build/twin4build/estimator/tests/generated_files/model_parameters/chain_logs/';

# Get the most recent `.pickle` file and assign it to `RECENT`.
RECENT=$(ssh ${USER}@${HOST} ls -lrt ${REMOTE_PATH} | awk '/.pickle/ { f=$NF }; END { print f }');
echo $RECENT
echo ${USER}@${HOST}:${REMOTE_PATH}${RECENT}
echo ${LOCAL_PATH}
# Run the actual SCP command.
# '${RECENT}' ${RECENT}
scp -r ${USER}@${HOST}:${REMOTE_PATH}${RECENT} "${LOCAL_PATH}";