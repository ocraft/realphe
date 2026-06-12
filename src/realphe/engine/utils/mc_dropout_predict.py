import os
import random
from typing import Tuple

import numpy as np
import torch
from torch import nn


@torch.no_grad()
def mc_dropout_predict(
    model: nn.Module,
    X: np.ndarray | torch.Tensor,
    n_iter: int = 50,
    seed: int = 42,
    device: str | torch.device = "cpu",
) -> Tuple[np.ndarray, np.ndarray]:

    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X.astype(np.float32))

    X = X.to(device)

    model = model.to(device)
    preds = []
    model.train()

    for i in range(n_iter):

        # strict determinism
        current_seed = seed + i

        os.environ["PYTHONHASHSEED"] = str(current_seed)
        random.seed(current_seed)
        np.random.seed(current_seed)
        torch.manual_seed(current_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(current_seed)
            torch.cuda.manual_seed_all(current_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        y_pred = model(X)

        preds.append(
            y_pred.detach().cpu().numpy()
        )

    # shape:
    # (n_iter, n_samples, n_classes)
    preds = np.stack(preds, axis=0)

    mean_pred = preds.mean(axis=0)
    std_pred = preds.std(axis=0)

    return mean_pred, std_pred
