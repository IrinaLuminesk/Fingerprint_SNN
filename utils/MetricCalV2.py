import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc
import numpy as np
class MetricCalV2():
    def __init__(self, device) -> None:
        self.device = device
        self.reset()
    def reset(self):
        self.total_cosemb_loss = torch.zeros(1, device=self.device)
        
        self.correct = torch.zeros(1, device=self.device)
        self.total = torch.zeros(1, device=self.device)

        # self.scores = []
        # self.labels = []

        self.all_scores = []
        self.all_labels = []

        # self.fpr = 0
        # self.tpr = 0
        # self.thresholds = 0
        # self.roc_auc = 0

    @torch.no_grad()
    def update_test(self, loss, outputs1, outputs2, targets):
        #Dùng để tính classification loss
        batch_size = targets.size(0)
        self.total_cosemb_loss += loss.detach() * batch_size
        self.total += batch_size
        
        scores = F.cosine_similarity(outputs1, outputs2)
        self.all_scores.append(scores.detach().cpu())
        self.all_labels.append(targets.detach().cpu())

    @torch.no_grad()
    def update_train(self, loss, outputs1, outputs2, targets):
        batch_size = targets.size(0)


        self.total_cosemb_loss += loss.detach() * batch_size
        self.total += batch_size
        
        scores = F.cosine_similarity(outputs1, outputs2)
        self.all_scores.append(scores.detach().cpu())
        self.all_labels.append(targets.detach().cpu())

    def compute_fpr_tpr_thresholds(self):
        scores = torch.cat(self.all_scores)
        labels = torch.cat(self.all_labels)

        labels01 = (labels == 1).int() #Chuyển đổi labels từ [-1, +1] sang [0, 1]

        scores_np = scores.numpy()
        labels_np = labels01.numpy()
        
        self.fpr, self.tpr, self.thresholds = roc_curve(labels_np, scores_np)
        self.roc_auc = auc(self.fpr, self.tpr)

    @property
    def avg_cosemb_loss(self):
        """Average loss over all accumulated batches."""
        return (self.total_cosemb_loss / self.total).item() if self.total > 0 else 0.0


    @property
    def avg_accuracy(self):
        """Accuracy (%) over all accumulated batches."""
        return (self.correct / self.total).item() if self.total > 0 else 0.0
    
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
        fnr = 1 - self.tpr
        eer_idx = np.nanargmin(np.abs(fnr - self.fpr))
        eer = self.fpr[eer_idx]
        return eer
    
    def tar_at_far(self, far_target):
        idx = np.where(self.fpr <= far_target)[0]
        return self.tpr[idx[-1]] if len(idx) > 0 else 0.0
    