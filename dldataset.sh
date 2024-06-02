#!/bin/bash

# https://www.nist.gov/itl/products-and-services/emnist-dataset

# Check if curl is installed
if ! command -v curl &> /dev/null
then
    echo "curl could not be found. Please install curl and try again."
    exit 1
fi

# Create datasets directory and navigate to it
mkdir -p datasets
cd datasets

# Download the EMNIST dataset using curl
echo "Downloading EMNIST dataset..."
curl "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip" --output gzip.zip

# Unzip the downloaded file
echo "Unzipping the dataset..."
unzip gzip.zip

echo "Download and extraction complete."
