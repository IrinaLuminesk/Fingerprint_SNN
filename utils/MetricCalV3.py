import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc
import numpy as np
class MetricCalV3():
    def __init__(self, device) -> None:
        self.device = device
        self.reset()
    def reset(self):
        self.loss = torch.zeros(1, device=self.device)


        self.total = torch.zeros(1, device=self.device)

        self.embeds = []
        self.labels = []
    
    @torch.no_grad()
    def _update(self, loss, outputs, targets):

        batch_size = targets.size(0)

        # accumulate loss
        self.loss += loss.detach() * batch_size
        self.total += batch_size

        # store embeddings + labels
        self.embeds.append(outputs.detach().cpu())
        self.labels.append(targets.detach().cpu())

    @torch.no_grad()
    def update_test(self, loss, outputs, targets):
        self._update(loss, outputs, targets)

    def update_train(self, loss, outputs, targets):
        self._update(loss, outputs, targets)


    @property
    def avg_loss(self):
        """Average loss over all accumulated batches."""
        return (self.loss / self.total).item() if self.total > 0 else 0.0
    
    def compute_fpr_tpr_thresholds(self):
        embeds = torch.cat(self.embeds, dim=0)
        labels = torch.cat(self.labels, dim=0)

        # -------- pairwise distances --------
        dists = torch.cdist(embeds, embeds)

        # -------- same-class matrix --------
        labels = labels.unsqueeze(0)
        targets = labels == labels.t()

        # -------- keep upper triangle only --------
        mask = torch.ones_like(dists).triu(1).bool()

        dists = dists[mask]
        targets = targets[mask]

        # ROC expects higher score = positive
        scores = (-dists).numpy()
        targets = targets.numpy().astype(int)

        # -------- ROC --------
        fpr, tpr, thresholds = roc_curve(targets, scores)
        self.roc_auc = auc(fpr, tpr)

        fnr = 1 - tpr
        eer_idx = np.nanargmin(np.abs(fnr - fpr))
        self.eer = fpr[eer_idx]

        idx = np.argmin(np.abs(fpr - 1e-3))
        self.tpr = tpr[idx]
    
    @property
    def ROC_AUC(self):
        """
        AUC ≈ 0.5 → useless
        AUC > 0.95 → strong
        AUC > 0.99 → excellent
        """
        return self.roc_auc
    
    @property
    def EER(self):
        """
        EER < 10% → weak
        EER < 5% → decent
        EER < 1% → strong
        EER < 0.1% → excellent
        """
        return self.eer
    @property
    def tar_at_far(self):
        self.tpr
    