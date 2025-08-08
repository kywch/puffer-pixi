#!/bin/bash

export PATH="$PIXI_PROJECT_ROOT/.pixi/envs/default/bin:$PATH"

export LD_LIBRARY_PATH="$PIXI_PROJECT_ROOT/.pixi/envs/default/lib:$PIXI_PROJECT_ROOT/.pixi/envs/default/lib/python3.12/site-packages/torch/lib"
export CUDA_HOME=$CONDA_PREFIX