#!/bin/bash

# Create main folders
mkdir -p src
mkdir -p src/cloud
mkdir -p src/components
mkdir -p src/constants
mkdir -p src/configeration
mkdir -p src/pipeline
mkdir -p src/entity
mkdir -p src/logger
mkdir -p src/exception
mkdir -p src/utils

mkdir -p config_yaml
touch config_yaml/config.yaml
touch config_yaml/param.yaml
touch config_yaml/secrets.yaml


# # Create __init__.py in each folder
touch src/__init__.py
touch src/cloud/__init__.py
touch src/components/__init__.py
touch src/constants/__init__.py
touch src/configeration/__init__.py
touch src/pipeline/__init__.py
touch src/entity/__init__.py
touch src/logger/__init__.py
touch src/logger/logging.py
touch src/exception/__init__.py
touch src/utils/__init__.py


# Create main files
touch app.py
touch requirements.txt
touch notex.txt 
touch setup.py 
touch templates.sh




## bash templates.sh


