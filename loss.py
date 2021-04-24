import math

import torch
import torch.nn as nn


class WeightAdjustingLoss(nn.Module):

    def __init__(self, weight, prior, tau=1.0, mmi=False):
        super(WeightAdjustingLoss, self).__init__()

        self.tau = tau
        self.mmi = mmi
        self.weights_ce = nn.CrossEntropyLoss(weight=weight)
        self.prior = torch.log(prior + 1e-8).unsqueeze(dim=0)
        self.normal_ce = nn.CrossEntropyLoss()

    def forward(self, y_pred, y_true, weights=None):
        if self.mmi is True:
            y_pred = y_pred + self.tau * self.prior
        return self.weights_ce(y_pred, y_true)


class FocalLoss(nn.Module):
    """Multi-class Focal Loss Implementation"""

    def __init__(self, gamma=2, alpha=0.8, weight=None, reduction='mean', ignore_index=-100, smooth=1):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.weight = weight
        self.smooth = smooth
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, input, target):
        """
        input: [N, C]
        target: [N, ]
        """
        log_pt = torch.log_softmax(input, dim=1)
        pt = torch.exp(log_pt)
        # let us keep value of alpha = 0.8
        # Try setting alpha = 1.0 to check if it works better
        log_pt = self.alpha * (1 - pt) ** self.gamma * log_pt
        loss = torch.nn.functional.nll_loss(log_pt, target, self.weight, reduction=self.reduction,
                                            ignore_index=self.ignore_index)
        return loss


class ComboLoss(nn.Module):

    def __init__(self, alpha=0.25, ratio=0.5, beta=0.6, smooth=1, weight=None, size_average=True):
        """

        The loss function aims to penalize False Positives and False Negatives
        """
        super(ComboLoss, self).__init__()

        self.alpha = alpha  # < 0.5 penalises FP more, > 0.5 penalises FN more
        self.beta = beta
        self.smooth = smooth
        self.ratio = ratio

    def forward(self, input, target):
        input = input.view(-1)
        target = target.view(-1)

        intersection = (input * target).sum()
        dice = (2. * intersection + self.smooth) / (input.sum() + target.sum() + self.smooth)

        input = torch.clamp(input, math.exp(1), 1.0 - math.exp(1))
        out = - (self.alpha * ((target * torch.log(input)) + ((1 - self.alpha) * (1 - target) * torch.log(1 - input))))
        weighted_ce = out.mean(-1)
        combo = (self.ratio * weighted_ce) - ((1 - self.ratio) * dice)
        return combo
