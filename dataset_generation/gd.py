
import numpy as np
from lines import calcFlatGroundDepth, expand_segments
import cv2 as cv

gep_depth = calcFlatGroundDepth(-0.2, 1, 5)

cv.imshow("d", gep_depth)
cv.waitKey()
