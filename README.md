



## Major Job
```bash
python3 launch/do_diffusion.py --dataset data/earth

python3 launch/project_metric.py --dataset data/earth --do_fuse
```

**Scaling Fixed**
Check the scaling and it is now 'MEAN_Z', which finds the scale factor such that the distance of flattened background equals the pose altitude. Must be much better than 'MIN_Z' (the old method) which tried to equalize the minimun z of raw projected depth to pose altitude - prone to great errors when there is tiny a error in ground perception by MDE.

**Shape Enhanced**
The current method for reshaping is: project depth radially, diffuse background spline, reshape depth (currently ndfDrop method is tested) to flatten the background. The shapes ar better than raw projected MDE output (which basically needs pyramid projection). 
*Question:* Wasn't it cleaner if we de-canonicalized the depth pyramid with a simpler geometrical transform? Was its quality equal to what we have now? If ours would be better, then our current math is something worth it! ...


### Evaluation

Provide an automatic evaluation through ground-truth data and our output 3d models. Separately check the following metrics:
- Shape error (Independent of scale and pose)
- Scale error (Independent of shape and pose)
- Pose error (Independent of shape and scale)
- Depth error (calculated in 2.5 depth image space)
- Point cloud error (point-by-point distance)

### Final Products
**Finalize the method**
1. Mean_z scaling + ndfDrop reshape (current state)
2. (above) + Does 'unfold' make the results better?
3. Compare de-canonicalization vs our radialproj-unfold method in terms of shape error

--> The outcome of 1,2,3 is our product-1

**Background RLS**

4. Evaluate the product-1 using a background model fitted overally on the vision stream (in an online scheme: correcting the ground model through sequential vision stream) 

--> product-2 (removes product-1)

**Elevation Fusion**

5. Instead of flat-ground-fusion, perform elevation data fusion with elevation diffused from GT data OR nasa elevation data

--> product-3 (presentable along with product-2)