#!/usr/bin/env bash

echo $PWD
singularity build --force ./singularity/jaeger_dev.sif ./singularity/jaeger_dev_singularity.def 
# sync with zeus
rsync -azvh --progress ./singularity/jaeger_dev.sif zeus:/mnt/beegfs/bioinf/wijesekara/jaeger/container/jaeger_dev.sif

#sync with brain 
rsync -azvh --progress ./singularity/jaeger_dev.sif brain:/home/wijesekary/jaeger/container/jaeger_dev.sif