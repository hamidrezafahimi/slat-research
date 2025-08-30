



## Major Job
```bash
cd launch/local

python map3d_img.py ../../data/kitti_image_sample/metric_depth.csv ../../data/kitti_image_sample/metric_pattern.csv ../../data/kitti_image_sample/color.png ../../data/kitti_image_sample/mask.png
# OR simpler
python process_single.py ../../data/kitti_image_sample/metric_depth.csv ../../data/kitti_image_sample/metric_pattern.csv ../../data/kitti_image_sample/color.png
```

## Tools
```bash
cd tools
```

### Generate Point Cloud Data from Raw Depth Image
```bash
python mapping/depth2pcd.py ../data/kitti_image_sample/metric_depth.csv ../data/kitti_image_sample/color.png

# For visualizing what is saved:
python vis/vis_current_pcd.py output.pcd
```

### Manual Pattern Diffusion
```bash
python diffusion/app.py --grid_w 6 --grid_h 6 --metric_w 10 --metric_h 10 --cloud depth_pcd_file.pcd --step 1
# OR 
python diffusion/app.py --cloud depth_pcd_file.pcd --spline_data spline_ctrl.csv --step 0.05
```

### Generating Depth Data from Spline

```bash
# First, generate a depth data from your spline
python mapping/capture_3dSpline.py spline_ctrl.csv

# Then convert that data to pcd, if desired
python mapping/depth2pcd.py depth_spline.csv ../data/viot_image_sample/color.png

# To check the generated depth image vs its base spline
python vis/vis_3dspline.py --spline_data spline_ctrl.csv --cloud bg_pcd_file.pcd

# Or to check with the main depth image
python vis/vis_3dspline.py --spline_data spline_ctrl.csv --cloud depth_pcd_file.pcd
```
