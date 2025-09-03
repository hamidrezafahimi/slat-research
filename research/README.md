

```bash
cd fusion

python unfold_pcd_surface_points.py --pcd ../../data/research/surface_to_unfold.pcd

python depth2unfoldedPixels.py --metric_depth ../../data/research/depth_spline_viot.csv --hfov_deg 66.0 --pose_x 0 --pose_y 0 --pose_z 9.4 --pose_roll 0 --pose_pitch -0.78 --pose_yaw 0 --n_greens 10

python unfold_depth_image_border_points.py --metric_depth ../../data/research/depth_spline_viot.csv --n_greens 32

python unfold_depth_image_borders.py   --metric_depth ../../data/research/depth_spline_viot.csv   --hfov_deg 66.0 --pose_x 0 --pose_y 0 --pose_z 9.4   --pose_roll 0 --pose_pitch -0.78 --pose_yaw 0   --n_greens 32

python unfold_surface.py --metric_depth ../../data/research/depth_spline_viot.csv --hfov_deg 66.0 --pose_x 0 --pose_y 0 --pose_z 9.4   --pose_roll 0 --pose_pitch -0.78 --pose_yaw 0 --n_greens 32
```
