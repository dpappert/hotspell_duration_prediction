"""
Custom loss functions for the torch-based models (CNN, MLP).

`FocalLoss` was verified byte-identical across all 21 CNN scripts and all
4 MLP scripts before extraction -- no behavioural ambiguity here.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # ensure correct shape
        targets = targets.float().unsqueeze(1) if targets.dim() == 1 else targets.float()

        # BCE with logits (no reduction!)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        # compute pt = p_t
        pt = torch.exp(-bce_loss)

        # focal loss
        loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
