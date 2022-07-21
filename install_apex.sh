#!/bin/bash

apex_dir=./apex
if [ ! -d $apex_dir ]; then
  git clone https://github.com/NVIDIA/apex
  cd apex
  git reset --hard ae7576
else
  cd apex
fi
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
