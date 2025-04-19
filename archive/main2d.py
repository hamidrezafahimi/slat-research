import matplotlib.pyplot as plt
import numpy as np
from VisualDepthGeom import VisualDepthGeometry2D
from GroundTruth2D import GroundTruth2D
import numpy as np


depth_data1 = np.array([140 ,139 ,139 ,139 ,138 ,138 ,137 ,135 ,133 ,133 ,133 ,132 ,130 ,129 ,128 ,129 ,127 ,127 ,127 ,127 ,126 ,125 ,120 ,113 ,111 ,112 ,107 ,105 ,98 ,86 ,75 ,63 ,60 ,51 ,42 ,40 ,41 ,42 ,43 ,41 ,43 ,44 ,44 ,45 ,46 ,47 ,48 ,48 ,49 ,50 ,51 ,51 ,52 ,53 ,54 ,55 ,55 ,56 ,57 ,58 ,59 ,60 ,61 ,62 ,63 ,63 ,64 ,65 ,66 ,67 ,68 ,69 ,70 ,71 ,73 ,74 ,76 ,77 ,78 ,79 ,80 ,81 ,82 ,83 ,83 ,84 ,85 ,86 ,87 ,88 ,89 ,89 ,90 ,91 ,92 ,92 ,93 ,94 ,95 ,96 ,96 ,97 ,98 ,99 ,100 ,101 ,102 ,102 ,103 ,104 ,105 ,106 ,107 ,108 ,109 ,110 ,110 ,111 ,112 ,113 ,114 ,115 ,115 ,116 ,117 ,118 ,118 ,119 ,119 ,120 ,121 ,122 ,122 ,123 ,124 ,124 ,125 ,126 ,126 ,127 ,128 ,129 ,129 ,130 ,131 ,131 ,132 ,133 ,134 ,134 ,135 ,136 ,136 ,137 ,138 ,139 ,139 ,140 ,141 ,142 ,142 ,143 ,143 ,144 ,145 ,146 ,146 ,147 ,148 ,149 ,149 ,150 ,151 ,151 ,152 ,153 ,153 ,154 ,154 ,155 ,155 ,156 ,156 ,156 ,157 ,157 ,158 ,158 ,158 ,158 ,159 ,159 ,159 ,159 ,159 ,159 ,159 ,161 ,178 ,182 ,184 ,185 ,185 ,185 ,183 ,183 ,183 ,184 ,184 ,185 ,185 ,185 ,186 ,186 ,186 ,187 ,187 ,187 ,188 ,188 ,189 ,189 ,190 ,190 ,190 ,191 ,192 ,192 ,192 ,193 ,193 ,194 ,194 ,195 ,196 ,196 ,197 ,198 ,198 ,199 ,200 ,200 ,200 ,201 ,201 ,202 ,202 ,202 ,202 ,203 ,203 ,203 ,203 ,204 ,204 ,204 ,204 ,204 ,205 ,205 ,205 ,205 ,205 ,205 ,206 ,206 ,206 ,207 ,207 ,207 ,208 ,208 ,208 ,209 ,209 ,210 ,210 ,210 ,211 ,211 ,212 ,212 ,213 ,213 ,213 ,214 ,214 ,214 ,215 ,215 ,215 ,216 ,216 ,216 ,217 ,218 ,218 ,218 ,219 ,219 ,220 ,220 ,220 ,220 ,221 ,221 ,223 ,224 ,224 ,224 ,225 ,225 ,224 ,224 ,224 ,224 ,224 ,224 ,225 ,225 ,225 ,226 ,226 ,226 ,226 ,227 ,227 ,227 ,227 ,227], dtype=float)

cam_point = (0, 9.4)
num_of_pix = 30
cam_point = cam_point
cam_ver_fov_deg = 50
fix_roof_z = 3
ground_z = 0
cam_angle = -0.783653

def analyze(dd, ax, lab, teps=None):
    vdg2d = VisualDepthGeometry2D(cam_point=cam_point,
                                cam_ver_fov_deg=cam_ver_fov_deg,
                                fix_roof_z=fix_roof_z,
                                ground_z=ground_z,
                                cam_angle=cam_angle)
    vdg2d.registerData(dd)
    # p = vdg2d.estimate()
    gs, reps, dps, hps = vdg2d.getInternalData()
    # ax.plot(e[:,0], e[:,1], color='red', label="")
    print(dps)
    ax.plot(dps[:,0], dps[:,1], label="raw depth estimation for "+lab, linewidth=2, color='black')
    ax.plot(hps[:,0], hps[:,1], label="over-ground normalized depth estimation for "+lab, 
            linewidth=3, color='grey')
    k = 0
    for g, r in zip(gs, reps):
        if k % 20 == 0:
            ax.plot([cam_point[0], r[0]], [cam_point[1], r[1]], ':', color='orange')
            ax.plot([cam_point[0], g[0]], [cam_point[1], g[1]], ':', color='blue')
        k += 1

    if not teps is None:
        ax.plot(teps[:,0], teps[:,1], label="ground_truth for " + lab)


# Create the camera radial lines object
gt2d = GroundTruth2D(cam_point, cam_angle, cam_ver_fov_deg, num_of_pix)
# # gt2d.plot()
depth_data2 = gt2d.get_depth_data()
# real_points = gt2d.get_ground_truth()
# true_end_points = gt2d.get_seen_points()
ground_truth_points = gt2d.get_closest_points()
depth_data2 = np.array(depth_data2, dtype=float)
# print (depth_data)

# find ground truth
# gt_depth = []
# for t in true_end_points:
#     gt_depth.append(np.sqrt((t[0] - cam_point[0])**2 + (t[1] - cam_point[1])**2))

# print(p)
# angles, geps, f_n, h_n, g_n = vdg2d.getInternalData()
# angles, geps, f_n, h_n, g_n, f, h, g = vdg2d.getInternalData()

# real_points_est, rdps, geps = vdg2d.estimateRealPoints()

# f(x) + g(x) = h(x)
# f, h --> regenerateFromGroundTruth --> h
# g = vdg2d.regenerateFromGroundTruth()
# Next, we must do this: f, h' --> reg... --> h, while h' = K and K is known (how to calc?)

# Plot setup
# ax = plt.gca()
# ax.set_aspect('equal', adjustable='box')

# ax.plot(angles, f_n, color='blue')
# ax.plot(angles, h_n, color='red')
# ax.plot(angles, g_n, color='green')
# plt.gca().autoscale()
plt.figure()
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
analyze(depth_data1, ax, "real image")
# analyze(depth_data2, ax2, "gt", ground_truth_points)


# for pt in true_end_points:
#     ax.scatter(pt[0], pt[1], color='blue')

# for pt in g:
#     ax.scatter(pt[0], pt[1], color='red')

# print(g)

# for end in end_points:
#     ax.plot([camera_point[0], end[0]], [camera_point[1], end[1]], color="yellow")



# for end in g:
#     ax.plot([cam_point[0], end[0]], [cam_point[1], end[1]], color="blue")

# # # Plot the red points on each blue line based on the random values
# for point in rdps:
#     ax.plot(point[0], point[1], 'o', color="red", label='rdps')

# for i, point in enumerate(irn_points_2):
#     ax.plot(point[0], point[1], 'o', color="green", label='irn_points')

# for i, point in enumerate(tcdps):
#     ax.plot(point[0], point[1], 'o', color="pink", label='tcdps')

# for i, point in enumerate(corrected_rdp):
#     ax.plot(point[0], point[1], 'o', color="orange", label='corrected_rdp')

# for i, point in enumerate(irn_points_3):
#     ax.plot(point[0], point[1], 'o', color="brown", label='corrected_rdp irned')

# for i, point in enumerate(corrected_tcdps):
#     ax.plot(point[0], point[1], 'o', color="yellow", label='corrected_tcdps')

# for point in real_points_est:
#     ax.plot(point[0], point[1], 'o', color="purple", label='corrected_tcdps irned')

# for i, point in enumerate(tcdp):
#     plt.plot(point[0], point[1], 'o', color="purple")

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
# plt.xlabel("X-axis")
# plt.ylabel("Z-axis")
plt.grid(True)
plt.show()