#!/usr/bin/env bash
set -e

# Directory where VO JSON files live (or use "." if they are in the current directory)
VO_DIR="batch_results"

# Find and sort all vo_ts*_ns*.json files
vo_files=( $(ls "${VO_DIR}"/vo_ts*_ns*.json 2>/dev/null | sort) )
if [ "${#vo_files[@]}" -eq 0 ]; then
    echo "No VO files (vo_ts*_ns*.json) found in ${VO_DIR}."
    exit 1
fi

for vof in "${vo_files[@]}"; do
    echo "---------------------------------------------"
    echo "Plotting '${vof}' (press 'q' in the plot window to continue)..."

    # Optional: customize title here; if you omit --title, the Python script builds one automatically
    TITLE="Plot for ${vof}"
    
    python plot_vo_output.py "${vof}" --title "${TITLE}"
done

echo "All plots completed."
