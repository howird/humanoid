#!/bin/bash
DATASET_PATH="/home/howard/stathletes-multiview-games"
MIN_SHOT_NUMBER=000

# Array of game names to process
GAME_NAMES=(
    '02-06-23 MIN @ ARI'
    '03-02-23 MTL @ LAK'
    '03-10-23 ANA @ CGY'
    '03-24-23 ARI @ COL'
    '04-06-23 CHI @ VAN'
    '11-01-22 PHI @ NYR'
    '11-08-22 EDM @ TB'
    '11-17-22 SJS @ MIN'
    '02-19-23 WPG @ NJD'
    '03-02-23 NSH @ FLA'
    '03-23-23 NYR @ CAR'
    '03-30-23 CBJ @ BOS'
    '04-08-23 VGK @ DAL'
    '11-03-22 LAK @ CHI'
    '11-12-22 OTT @ PHI'
    '11-26-22 VAN @ VGK'
    '03-01-23 WSH @ ANA'
    '03-04-23 DET @ NYI'
    '03-23-23 SEA @ NSH'
    '04-01-23 TOR @ OTT'
    '10-29-22 PIT @ SEA'
    '11-05-22 NYI @ DET'
    '11-17-22 NJD @ TOR'
    '12-10-22 BUF @ PIT'
)

# Get total number of games
TOTAL_GAMES=${#GAME_NAMES[@]}

# Verify all game directories exist
echo "Verifying game directories..."
MISSING_DIRS=0
for GAME_NAME in "${GAME_NAMES[@]}"; do
    if [ ! -d "${DATASET_PATH}/${GAME_NAME}" ]; then
        echo "Error: Directory not found: ${DATASET_PATH}/${GAME_NAME}"
        MISSING_DIRS=$((MISSING_DIRS + 1))
    fi
done

if [ $MISSING_DIRS -gt 0 ]; then
    echo "Error: Found $MISSING_DIRS missing directories. Please check the paths and try again."
    exit 1
fi

echo "All game directories verified. Starting processing..."
echo "Processing shots with numbers >= ${MIN_SHOT_NUMBER}"

# Process each game
for i in "${!GAME_NAMES[@]}"; do
    GAME_NAME="${GAME_NAMES[$i]}"
    CURRENT_INDEX=$((i + 1))
    echo "Processing game ${CURRENT_INDEX}/${TOTAL_GAMES}: ${GAME_NAME}"
    
    BASE_DIR="${DATASET_PATH}/${GAME_NAME}"
    OUTPUT_DIR="${BASE_DIR}/hmr-track"

    # Create output directory if it doesn't exist
    mkdir -p "${OUTPUT_DIR}"

    # Find and process all video files in the game directory
    # Using regex to match shot numbers >= MIN_SHOT_NUMBER
    find "${BASE_DIR}" -type f -name "*.mp4" -path "*/shots_*/*" | \
    grep -E "shot[0-9]{3}\.mp4$" | \
    awk -F/ -v min=$MIN_SHOT_NUMBER '{
        match($NF, /shot([0-9]+)\.mp4/, arr);
        if (arr[1] >= min) print $0;
    }' | \
    xargs -I {} python scripts/hmr_track.py "{}" --video.output_dir "${OUTPUT_DIR}"
done

echo "All ${TOTAL_GAMES} games processed!"
