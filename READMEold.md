
[![GT generation](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hamidrezafahimi/depth_pattern_analysis/blob/main/launch/notebook/generate_ground_truth.ipynb)


```bash

cd launch/local

# GOOD
python3 map3d_img.py /media/hamid/856edb17-0012-45a1-8ec5-b69ec208a64d/home/hamid/Downloads/depth_data.csv /media/hamid/Workspace/viot3/bgsub4/outs9/00000001_bgCurve.jpg /media/hamid/Workspace/viot3/data1005/orig/00000001.jpg /home/hamid/w/thing/adaptive_thresh/binary_threshold.png

# BAD
python3 map3d_img_simple2.py /home/hamid/Downloads/00000027.csv /media/hamid/856edb17-0012-45a1-8ec5-b69ec208a64d/home/hamid/pattern.png /media/hamid/856edb17-0012-45a1-8ec5-b69ec208a64d/home/hamid/fasdfdf.png /home/hamid/binary_threshold.png

# EVAL
python3 eval_depth.py /home/hamid/Downloads/00000027.csv /home/hamid/Downloads/interpolated_depth_row201plus.csv
```