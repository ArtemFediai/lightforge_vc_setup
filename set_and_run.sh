#!/bin/bash

# Set environment variables and source the configuration
export NANOMATCH=/home/hk-project-zimnano/nz8308/nanomatch
export NANOVER=V6
source $NANOMATCH/$NANOVER/configs/quantumpatch.config

# Run your Python script
python /hkfs/work/workspace/scratch/nz8308-VC/vdw_materials/light/setup/change_settings.py
