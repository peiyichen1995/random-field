#!/usr/bin/env bash
count = 0
for L in {0..999}; do
  python generate_random_field.py 64 4000 1440000 $L &
  (( count ++ ))
  if (( count = 10 )); then
    wait
    count=0
  fi
done




# for L in {1..5}; do
#   echo "====================================="
#   echo "running L $L sample $sample"
#   echo "====================================="
#   python generate_random_field.py 64 4000 1440000 $L
# done
