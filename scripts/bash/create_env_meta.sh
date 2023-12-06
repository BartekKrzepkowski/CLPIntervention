#!/bin/bash

srun --mem=8G --cpus-per-task=4 --qos=normal -p cpu --pty /bin/bash
sbatch create_env.sh