
## Overall product
```bash
# Find a good spline
python3 search_surfaces.py --cloud pcd_file.pcd --N 1000 --save-best best_ctrl.csv

# (optional - visualize score of what is found)
python3 score3d.py --in pcd_file.pcd --show --ext_mode smoothness --pp_mode score --max_dz 5 --spline_data best_ctrl.csv --plot_spline

# Fine-Tune and find the final answer
# (visualized)
ython3 optim.py --in pcd_file.pcd --spline_data best_ctrl.csv --alpha 0.0001 --show_pp --show_ext --max_dz 5 --verbosity tiny
# (no viz)
python3 optim.py --in pcd_file.pcd --spline_data best_ctrl.csv --alpha 0.0001 --max_dz 5 --verbosity tiny --fast
```

## Demo
```bash
python3 score3d.py --in pcd_file.pcd --show --ext_mode black --pp_mode score --max_dz 3 --spline_data spline_ctrl.csv --plot_spline

# (just demo)
python3 generate_random_surfs.py --cloud pcd_file.pcd --viz
```