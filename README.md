



## Major Job
```bash
cd launch/local
python map3d_img.py ../../data/kitti_image_sample/metric_depth.csv ../../data/kitti_image_sample/metric_pattern.csv ../../data/kitti_image_sample/color.png ../../data/kitti_image_sample/mask.png
# OR simpler
python process_single.py ../../data/kitti_image_sample/metric_depth.csv ../../data/kitti_image_sample/metric_pattern.csv ../../data/kitti_image_sample/color.png
```

## Tools
### Generate Point Cloud Data from Raw Depth Image
```bash
cd tools/mapping
python depth2pcd.py ../../data/kitti_image_sample/metric_depth.csv ../../data/kitti_image_sample/color.png

# For visualizing what is saved:
cd tools/mapping
python vis_current_pcd.py output.pcd
```

### Manual Pattern Diffusion
```bash
cd tools/diffusion
python app.py --grid_w 10 --grid_h 10 --metric_w 10 --metric_h 10 --cloud ../mapping/output.pcd --step 1
# OR 
python app.py --cloud ../mapping/output.pcd --spline_data spline_ctrl.csv --step 0.1
```

### Generating Depth Data from Spline

```bash
cd tools/mapping
python capture_3dSpline.py ../diffusion/spline_ctrl.csv
```
