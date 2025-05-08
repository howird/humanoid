#!/bin/bash
DATASET_PATH="/home/howard/stathletes-multiview-games"
GAME_NAME="02-06-23 MIN @ ARI"

BASE_DIR="${DATASET_PATH}/${GAME_NAME}"
OUTPUT_DIR="${BASE_DIR}/hmr-track"

mkdir -p "${OUTPUT_DIR}"

# within the specified game directory.
# Use xargs to run the python script for each found video file.
find "${BASE_DIR}" -type f -name "*.mp4" -path "*/shots_*/*" | \
xargs -I {} python scripts/hmr_track.py --video.source "{}" --video.output_dir "${OUTPUT_DIR}"
# find "${BASE_DIR}" -type f -name "*.mp4" -path "*/shots_*/*" | \
# xargs -I {} echo python scripts/hmr_track.py --video.source "{}" --video.output_dir "${OUTPUT_DIR}"
