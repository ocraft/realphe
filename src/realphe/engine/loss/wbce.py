from typing import cast

import torch
from torch import Tensor
from torch.nn import Module
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence


class WeightedBCEWithLogitsLoss(Module):
    """
    Weighted BCEWithLogits loss for PackedSequence inputs.

    Assumptions:
    - y_pred contains logits
    - y_true contains binary labels
    - y_pred and y_true share identical PackedSequence layout

    Reduction:
        sum(loss * weight) / sum(weight)
    """

    def __init__(
        self,
        ramp_min: float = 0.3,
        label_smoothing: float = 0.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()

        self.label_smoothing = label_smoothing
        self.ramp_min = ramp_min
        self.eps = eps

    @staticmethod
    def _build_ramp_weights(
        batch_sizes: Tensor,
        ramp_min: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:

        T = batch_sizes.numel()

        timestep_weights = torch.linspace(
            1.0,
            ramp_min,
            T,
            device=device,
            dtype=dtype,
        )

        return torch.repeat_interleave(
            cast(Tensor, timestep_weights).to(dtype=dtype, device=device),
            batch_sizes.to(device=device),
        )

    def forward(
        self,
        y_pred: PackedSequence,
        y_true: PackedSequence,
    ) -> Tensor:
        if not torch.equal(y_pred.batch_sizes, y_true.batch_sizes):
            raise ValueError(
                "y_pred and y_true must have identical PackedSequence layout"
            )

        logits = y_pred.data.float()
        target = y_true.data.float()

        # 0 -> s/2
        # 1 -> 1 - s/2
        if self.label_smoothing > 0.0:
            s = self.label_smoothing
            target = target * (1.0 - s) + 0.5 * s

        loss = F.binary_cross_entropy_with_logits(
            logits,
            target,
            reduction="none",
        )

        ramp = self._build_ramp_weights(
            batch_sizes=y_pred.batch_sizes,
            ramp_min=self.ramp_min,
            dtype=loss.dtype,
            device=loss.device
        )

        # Broadcast for multilabel case
        #
        # [N] -> [N, 1]
        while ramp.ndim < loss.ndim:
            ramp = ramp.unsqueeze(-1)

        weighted_ramp = ramp.expand_as(loss)

        weighted_loss = loss * weighted_ramp

        return weighted_loss.sum() / (
            weighted_ramp.sum() + self.eps
        )
