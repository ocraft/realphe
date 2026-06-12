import torch
from torchmetrics import Metric


class MeanAveragePrecisionAtK(Metric):

    full_state_update = False

    def __init__(
        self,
        k: int,
        dist_sync_on_step: bool = False
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.k = k

        self.add_state("ap_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    @torch.no_grad()
    def update(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor
    ):

        B, C = y_pred.shape
        k = min(self.k, C)

        # top-k predictions
        topk_idx = torch.topk(y_pred, k, dim=1, sorted=True).indices

        # relevance at top-k
        rel = torch.gather(y_true, dim=1, index=topk_idx).float()

        # precision@i
        precision_at_i = (
            rel.cumsum(dim=1)
            / torch.arange(
                1,
                k + 1,
                device=y_pred.device,
                dtype=torch.float32
            )
        )

        # sum precision only at relevant hits
        ap = (precision_at_i * rel).sum(dim=1)

        # IMPORTANT:
        # normalize by ALL relevant labels
        denom = y_true.sum(dim=1).float()

        # avoid divide-by-zero
        valid = denom > 0

        ap = torch.where(
            valid,
            ap / denom.clamp(min=1),
            torch.zeros_like(ap)
        )

        self.ap_sum += ap.sum()
        self.total += valid.sum()

    def compute(self):
        return self.ap_sum / self.total.clamp(min=1)
