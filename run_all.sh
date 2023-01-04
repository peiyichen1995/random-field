#!/usr/bin/env bash
for L in {1..5}; do
  echo "====================================="
  echo "running L $L sample $sample"
  echo "====================================="
  python generate_random_field.py 25 4000 250 $L
done