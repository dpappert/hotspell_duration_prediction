"""
Custom loss functions for the torch-based models (CNN, MLP).

`FocalLoss` below matches the CNN scripts' implementation, verified
byte-identical across all 21 CNN scripts before extraction.

CORRECTION: an earlier check claimed this was also byte-identical to the
MLP scripts' FocalLoss -- that check had a bug. The MLP scripts (all 4,
byte-identical to each other) use a simpler version: no configurable
`reduction` param (hardcoded `.mean()`), no defensive `unsqueeze` on 1D
targets. Verified this produces numerically identical output to the
version below for how both scripts actually call it -- targets are always
already 2D (`.unsqueeze(1)`'d before reaching the loss) and only the
default mean-reduction path is ever used -- so this single implementation
is reused for MLP too rather than keeping two near-duplicate classes.
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
