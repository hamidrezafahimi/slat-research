#!/usr/bin/env bash
set -e

# Create a directory to hold all generated scenes and VO outputs
OUTPUT_DIR="batch_results"
mkdir -p "${OUTPUT_DIR}"

# Fixed parameter
TRAJ_N=5

# Iterate over traj_step_size values: 1, 3, 5, 7, 9
for TS in 1 3 5 7 9; do
    echo "=== Generating scene with traj_step_size=${TS} ==="
    python projection.py \
        --traj_step_size "${TS}" \
        --no_imshow \
        --no_plot \
        --traj_N "${TRAJ_N}"

    # projection.py always writes scene_with_traj.json, so rename it per TS
    SCENE_FILE="${OUTPUT_DIR}/scene_ts${TS}.json"
    mv scene_with_traj.json "${SCENE_FILE}"

    # Iterate over noise_std values: 0.5, 1.0, 2.0, 3.0
    for NS in 0.5 1.0 2.0 3.0; do
        echo "--- Running VO on scene_ts${TS}.json with noise_std=${NS} ---"
        OUT_VO="${OUTPUT_DIR}/vo_ts${TS}_ns${NS}.json"
        python sample_vo.py \
            "${SCENE_FILE}" \
            --noise_std "${NS}" \
            --no_imshow \
            --no_plot \
            --out_vo "${OUT_VO}"
    done
done

echo "All runs complete. Results are in the '${OUTPUT_DIR}' directory."
