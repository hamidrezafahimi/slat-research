import numpy as np

class LossCalculator():
    def __init__(self):
        self.splineFitter = VisualSplineFit()
    
    def computeLoss(self, input_image, tmask, output_img):
        thresh_pat = self.splineFitter(output_img)
        # threshold input_image with thresh_pat: means set each pixel in input_image which is
        # more than corresponding pixel in thresh_pat, to zero, and the rest remain as they are
        masked_image = np.where(input_image > thresh_pat, 0, input_image)
        # then compute the overlap between 'tmask' and 'masked_image'. 1 means no overlap. 
        # 0 means they're exactly equal
        return np.count_nonzero(tmask & masked_image) / np.count_nonzero(tmask)