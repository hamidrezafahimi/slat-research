import torch
import torch.nn as nn

class CombinedLoss(nn.Module):
    def __init__(self, lambda_smooth=0.1, lambda_reg=0.1, g_min=-1.0, g_max=1.0):
        super(CombinedLoss, self).__init__()
        self.lambda_smooth = lambda_smooth
        self.lambda_reg = lambda_reg
        self.g_min = g_min
        self.g_max = g_max

    def forward(self, r, e_pred, m):
        g_pred = r - e_pred
        supervised_loss = torch.mean(((m == 0).float() * (e_pred - r)) ** 2)
        reg_loss = torch.mean(
            (m == 1).float() * (torch.clamp(self.g_min - g_pred, min=0) ** 2 +
                                torch.clamp(g_pred - self.g_max, min=0) ** 2)
        )
        e_grad = torch.diff(e_pred, dim=1)
        smooth_loss = torch.mean(e_grad ** 2)
        return supervised_loss + self.lambda_smooth * smooth_loss + self.lambda_reg * reg_loss
