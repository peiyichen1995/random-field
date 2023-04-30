#!/bin/bash
SAMPLE=500
input="scene_1_suplement.txt"
while IFS= read -r line
do
  echo "====================================="
  echo $SAMPLE
  python non_gaussian_fields_scene_1.py 64 $line $SAMPLE
  (( SAMPLE ++ ))
done < "$input"

# #!/bin/bash
# COUNTER=0
# INLOOPCOUNTER=0
# for L in {0..999}; do
#   python generate_random_field.py 64 4000 1440000 $L > output/command_output/$L.txt &
#   (( COUNTER ++ ))
#   if [ $COUNTER = 8 ]; then
#     wait
#     COUNTER=0
#     (( INLOOPCOUNTER ++ ))
#     echo $INLOOPCOUNTER
#   fi
# done


#!/usr/bin/env bash
# for L in {1..5}; do
#   echo "====================================="
#   echo "running L $L sample $sample"
#   echo "====================================="
#   python generate_random_field.py 64 4000 1440000 $L
# done
