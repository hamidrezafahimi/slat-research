import cv2
import numpy as np

def arg_min_2d(arr):
    amin = np.argmin(arr)
    w = arr.shape[1]
    return (int(np.floor(amin / w)), (amin + 1) % w - 1)


depth_image = cv2.imread("depth_image.png", cv2.IMREAD_GRAYSCALE).astype(np.float32)
bg_image = cv2.imread("bg_image.png", cv2.IMREAD_GRAYSCALE).astype(np.float32)
gep_image = cv2.imread("gep_image.png", cv2.IMREAD_GRAYSCALE).astype(np.float32)

proximity_image = 255.0 - depth_image
bg_prox = 255.0 - bg_image
gep_prox = 255.0 - gep_image
moved_prox = proximity_image - bg_prox + gep_prox
moved_depth = 255.0 - moved_prox

min_depth_index = arg_min_2d(depth_image)
min_moved_index = arg_min_2d(moved_depth)
min_before = depth_image[min_depth_index]
min_after = moved_depth[min_moved_index]
# print(min_after, min_before)
print(np.min(depth_image))
print(np.max(depth_image))
print(np.min(moved_depth))
max_after = np.max(moved_depth)
print(max_after)

if min_after < 1:
    min_before_ = min_before + 2 * abs(min_after)
    min_after_ = min_after + 2 * abs(min_after)
    moved_depth += 2 * abs(min_after)
    moved_depth += (min_before_ - min_after_) * ((max_after + 2 * abs(min_after) - moved_depth) / \
                    (max_after + 2 * abs(min_after) - min_after_))
    moved_depth -= 2 * abs(min_after)
else:
    moved_depth += (min_before - min_after) * ((max_after - moved_depth) / \
                    (max_after - min_after))

print(f"max: {np.max(depth_image)}, min: {np.min(depth_image)}")
print(f"max: {np.max(moved_depth)}, min: {np.min(moved_depth)}")

cv2.imwrite("depth_image_.png", depth_image.astype(np.uint8))
cv2.imwrite("moved_.png", moved_depth.astype(np.uint8))

assert np.min(moved_depth) >= 0 and np.max(moved_depth) <= 255.0, \
    f"max: {np.max(moved_depth)}, min: {np.min(moved_depth)}"