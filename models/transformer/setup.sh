#!/bin/bash

if [ ! -d ./cutlass ]; then
  git clone https://github.com/NVIDIA/cutlass.git && cd cutlass && git checkout ed2ed4d6 && cd ..
  echo "Clone cutlass ed2ed4d6 version"
fi

pip install -e .
