#!/bin/bash
python -m virtualenv p3env
source p3env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

mkdir datasets
#wget <add link> datasets/.