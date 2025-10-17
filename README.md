



## Major Job
```bash
# Diffuse a dataset in loop
python3 launch/do_diffusion.py --dataset data/earth

# Fuse a dataset in loop
python3 launch/project_metric.py --dataset data/earth --do_fuse
```

**Scaling Fixed**
Check the scaling and it is now 'MEAN_Z', which finds the scale factor such that the distance of flattened background equals the pose altitude. Must be much better than 'MIN_Z' (the old method) which tried to equalize the minimun z of raw projected depth to pose altitude - prone to great errors when there is tiny a error in ground perception by MDE.

**Shape Enhanced**
The current method for reshaping is: project depth radially, diffuse background spline, reshape depth (currently NDFDrop method is tested) to flatten the background. The shapes ar better than raw projected MDE output (which basically needs pyramid projection). 

### Research Trials
**Question:** Wasn't it cleaner if we de-canonicalized the depth pyramid with a simpler geometrical transform? Was its quality equal to what we have now? If ours would be better, then our current math is something worth it! ...
**Update/Answer:** I added the 'depyramidization' method in fusion methods and compared it with our currently active NDFDrop method. Turns out that our method is better. Indeed, to depyramidize the raw depth's pyramid-projection, we need backgroud info. Alternatively, the method is highly dependent to the z-val of flat background used as rotation anchor for any point (See function `fusion.helper.depyramidize_pointCloud` --> i'm talking about the var: `plane_z` in that function). Even determining such a plane as a planar mean of diffused background does not guarantee that the resulting reshaping method (so far, it must have been much more complex-in-implementation than my NDFDrop method!) will be consistent in reshaping the point cloud geometry in different areas (based on the fact that the current point to be depyramidized is below, or above the backgroud-mean-z plane), and probably ruins the local shapes. 
**Conclusion:** So yes! This proves that our flat-ground-fusion method is what it must be


**Question:** The current fine-tunning in diffusion procedure takes too much time. Isn't it better to satisfy with corse-tunning and fit a higher-degree spline surface to a filtered set of points after coarse-tunning, as fine-tunning without GD??
**Update/Answer:** In the previous commint, I implemented the above idea. The `research/diffusion/spline2d/fine_tune_fit.py` proves that surface fitting on filtered data is much weaker than our current iterative tunning method. If fine-tunning takes time, we can simply make the fine-tune spline grid weaker, which is not suggested! Fine-tunning tries to fit the ground patter more, in a tiny optimization radius which does not allow the ctrl points to see the non-ground points of point cloud (i.e. `max_dz` is set tiny in `scorer.reset` in fine-tunning phase of `diffusion.diffusion. ... .diffuse`)
**Conclusion:** So this proves that we are good to iteratively estimate the background pattern in our dataset generation scheme.


## Evaluation

Currently, use something like:
```bash
# in the root of a dataset, after running diffusion and fusion
python3 tools/vis/compare_pcds.py fusion/something.pcd red rawdepth/something.pcd
```

**TODO:**
Provide an automatic evaluation through ground-truth data and our output 3d models. Separately check the following metrics:
- Shape error (Independent of scale and pose)
- Scale error (Independent of shape and pose)
- Pose error (Independent of shape and scale)
- Depth error (calculated in 2.5 depth image space)
- Point cloud error (point-by-point distance)

### Final Products
**Finalize the method**
1. Mean_z scaling + NDFDrop reshape (current state)
2. (above) + Does 'unfold' make the results better?
--> The outcome of 1,2,3 is our product-1

**Background RLS**

4. Evaluate the product-1 using a background model fitted overally on the vision stream (in an online scheme: correcting the ground model through sequential vision stream) 

--> product-2 (removes product-1)

**Elevation Fusion**

5. Instead of flat-ground-fusion, perform elevation data fusion with elevation diffused from GT data OR nasa elevation data

--> product-3 (presentable along with product-2)