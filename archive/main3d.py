import matplotlib.pyplot as plt
from VisualDepthGeometry3D import VisualDepthGeometry3D
from lib.utils import set_axes_equal

cam_point = (0, 0, 9.4)

vdg3d = VisualDepthGeometry3D(cam_point=cam_point,
                              cam_hor_fov_deg=66,
                              fix_roof_z=3,
                              ground_z=0,
                              cam_angle=-0.783653,
                              img_downscale_factor = 0.0625)

vdg3d.registerData('depthpro_output.png')

corrected_tcdps_irn, rdps, geps = vdg3d.estimateRealPoints()

from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting 

# print(geps.shape)
# print(rdps)
# print(corrected_tcdps_irn.shape)

fig = plt.figure()
ax = fig.gca(projection='3d')

for i in range(geps.shape[0]):
    for j in range(geps.shape[1]):
        ax.plot([cam_point[0], geps[i,j,0]], [cam_point[1], geps[i,j,1]], [cam_point[2], geps[i,j,2]], color='blue')

ax.scatter(rdps[:,:,0], rdps[:,:,1], rdps[:,:,2], color = 'red')
ax.scatter(corrected_tcdps_irn[:,:,0], corrected_tcdps_irn[:,:,1], corrected_tcdps_irn[:,:,2], color = 'orange')
set_axes_equal(ax)
plt.show()
