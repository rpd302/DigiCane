#!/bin/bash

#SBATCH --job-name=lab1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20GB
#SBATCH --gres=gpu
#SBATCH --time=20:00:00
#SBATCH --output=out.%j


module load jupyter-kernels/py2.7
module load scikit-image/intel/0.13.1


python /home/at3577/mobile-vision/models/research/object_detection/train.py --logtostderr --train_dir=/scratch/at3577/coco_train/training/ --pipeline_config_path=/scratch/at3577/coco_train/ssd_mobilenet_final_v1.config


