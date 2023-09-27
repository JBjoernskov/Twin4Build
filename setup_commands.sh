#!/bin/sh
pip install -r requirements_linux.txt
sudo add-apt-repository universe
sudo apt update
sudo apt install graphviz