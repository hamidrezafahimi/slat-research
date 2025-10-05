import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2

metric_depth = np.loadtxt('/home/hamid/w/DATA/272/asfd.csv', delimiter=',', dtype=np.float32)
metric_bg = np.loadtxt('/home/hamid/w/DATA/272/zi.csv', delimiter=',', dtype=np.float32)
bg_image = cv2.imread('/home/hamid/w/DATA/272/pattern.png', cv2.IMREAD_GRAYSCALE).astype(np.float32)
mask_img = 255 - cv2.imread('/home/hamid/w/DATA/272/mask.png', cv2.IMREAD_GRAYSCALE)

metric_depth_scaled = metric_depth / np.max(metric_depth) * 255.0

# 1. interpolated background 
nnn = np.where(np.isnan(metric_bg), bg_image, metric_bg)
metric_bg_scaled = nnn / np.max(nnn) * 255.0
maxbg = np.where(bg_image < metric_bg_scaled, metric_bg_scaled, bg_image)
bg_scaled = np.where(mask_img > 127, maxbg, metric_depth_scaled)
bg_scaled = np.where(bg_scaled < metric_depth_scaled, metric_depth_scaled, bg_scaled)

before = np.where(metric_depth_scaled < bg_image, 255, 0).astype(np.uint8)
cv2.imshow("mask before rescaling", before)
after = np.where(metric_depth_scaled < bg_scaled, 255, 0).astype(np.uint8)
cv2.imshow("mask after rescaling", after)

cv2.imshow('d', metric_depth_scaled.astype(np.uint8))
cv2.imshow('b', metric_bg_scaled.astype(np.uint8))
cv2.imshow('sd', bg_scaled.astype(np.uint8))

# 2. Visual background 
bg_scaled1 = np.where(bg_image < metric_depth_scaled, metric_depth_scaled, bg_image)
cv2.imshow('wb', bg_scaled1.astype(np.uint8))



cv2.waitKey()