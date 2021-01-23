#!/usr/bin/env bash

if [ -d data ]; then
    echo "data directory already present, exiting"
    exit 1
fi

mkdir data
wget --recursive --level=1 --cut-dirs=3 --no-host-di
pytest
jupyter notebook noisy_mnist.ipynb
