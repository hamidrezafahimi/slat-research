:~/w/thing/adaptive_thresh$ rm -r masked_images/ glob_search_output/ loc_search_output/ depth_image_combined_thresh.jpg depth_image_processed_combined_thresh.jpg 
:~/w/thing/adaptive_thresh$ python3 search_thresh.py 
cell info: 88 82 5 4
Mosaic Array: 
[[ 531  391 1055   -1 1144]
 [ 178  103   18   -1  391]
 [   7   86   -1   -1   -1]
 [   0  384   -1   -1  466]]
Saved raw combined threshold => depth_image_combined_thresh.jpg
Saved processed (spline) combined threshold => depth_image_processed_combined_thresh.jpg
No valid candidate found for depth_image.jpg
:~/w/thing/adaptive_thresh$ python3 load_pattern.py depth_image_combined_thresh.jpg 

:~/w/thing/adaptive_thresh$ python3 load_pattern.py depth_image_processed_combined_thresh.jpg
:~/w/thing/adaptive_thresh$ python3 load_pattern.py depth_image_processed_combined_thresh.jpg 0

