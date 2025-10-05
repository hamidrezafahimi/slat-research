
```bash
# 1. Search
python3 spline2d/search_surfaces.py --cloud pcd_file.pcd --N 1000 --save-best best_ctrl.csv

# 2. Coarse Tune the Background Model
# (gui)
python3 spline2d/optim.py --initial_guess best_ctrl.csv --in pcd_file.pcd --iters 30 --max_dz <take-from-prev-code-log> --verbosity tiny --out op1.csv
# (fast)
python3 spline2d/optim.py --initial_guess best_ctrl.csv --in pcd_file.pcd --iters 30 --max_dz <take-from-prev-code-log> --verbosity tiny --fast --out op1.csv

# 3. Shift
python3 spline2d/shift.py --pcd pcd_file.pcd --ctrl op1.csv --out-csv shifted.csv --k 1.1

# 4. Upsample
python3 spline2d/upsample.py --spline_data shifted.csv --samples_u 6 --samples_v 6 --output_csv upsampled.csv

# 5. Fine Tune the Background Model
python3 spline2d/optim.py --initial_guess upsampled.csv --in pcd_file.pcd --iters 50 --max_dz <0.25*shift-val-in-shift-code-log> --verbosity tiny --base_ctrl op1.csv --fast --out final.csv --downsample 0.1 --alpha 10

# For extracting background with great details, repeat an "shift --> upsample --> fine-tune" step, with max_dz less than 1 ...

# Visualize result and save background pcd
python3 spline2d/project2spline.py --pcd pcd_file.pcd --ctrl final.csv --output bg.pcd
```