#!/bin/sh
sudo apt update
sudo apt-get install dos2unix
sudo add-apt-repository universe
sudo apt install graphviz -y
pip install -r requirements_linux_no_version.txt
