#!/bin/sh

echo "cd /gscratch/scrubbed/briggs3/get_roco_data/roco-dataset"
cd /gscratch/scrubbed/briggs3/get_roco_data/roco-dataset

echo "grabbing roco data"

python scripts/fetch.py

echo "finished!"
