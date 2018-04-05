#!/bin/bash

#SBATCH --job-name=lab1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=20GB
#SBATCH --time=18:20:00
#SBATCH --partition=c32_41
#SBATCH --output=out.%j


module purge
module load jupyter-kernels/py2.7
module load scikit-image/intel/0.13.1


python generate_tfrecord_final.py --csv_input=/home/at3577/mobile-vision/models/research/object_detection/object_detection_data/merged_test_labels.csv  --output_path=test.record --image_input=val2017/











