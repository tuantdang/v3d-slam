#!/bin/bash

set -e # Exit on error

# dataset_name="rgbd_dataset_freiburg3_sitting_static"
# dataset_name=rgbd_dataset_freiburg3_sitting_halfsphere
#dataset_name=rgbd_dataset_freiburg3_walking_halfsphere
# dataset_name=rgbd_dataset_freiburg3_sitting_rpy
#dataset_name=rgbd_dataset_freiburg1_xyz
#dataset_name=rgbd_dataset_freiburg3_walking_static
# dataset_name=rgbd_dataset_freiburg3_sitting_static
# dataset_name=rgbd_dataset_freiburg3_sitting_static
# dataset_name=rgbd_dataset_freiburg3_walking_static
dataset_name=rgbd_dataset_freiburg3_walking_rpy
echo "Working on $dataset_name..."

#echo "Generate association file..."
# rm  ~/slam/data/associations/$dataset_name.txt
#python ~/slam/data/tum.py  --name $dataset_name
#echo "===="

path_to_vocabulary="../../Vocabulary/ORBvoc.txt"
path_to_settings="config.yaml"
path_to_sequence="/home/tuandang/slam/data/extract/$dataset_name"
path_to_association="/home/tuandang/slam/data/associations/$dataset_name.txt"
echo $path_to_vocabulary 
echo $path_to_association
echo $path_to_sequence
echo $path_to_association
echo "===="
./slam $path_to_vocabulary $path_to_settings $path_to_sequence $path_to_association
