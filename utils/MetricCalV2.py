import torch
class MetricCalV2():
    def __init__(self, device) -> None:
        self.device = device
        self.reset()
    def reset(self):
        self.total_cosemb_loss = torch.zeros(1, device=self.device)
        
        self.correct = torch.zeros(1, device=self.device)
        self.total = torch.zeros(1, device=self.device)

    @torch.no_grad()
    def update_test(self, loss, outputs1, outputs2, targets):
        #Dùng để tính classification loss
        batch_size = targets.size(0)
        self.total_cosemb_loss += loss.detach() * batch_size
        self.total += batch_size
        

    @torch.no_grad()
    def update_train(self, loss, outputs1, outputs2, targets):
        batch_size = targets.size(0)


        self.total_cosemb_loss += loss.detach() * batch_size
        self.total += batch_size
        


    @property
    def avg_cosemb_loss(self):
        """Average loss over all accumulated batches."""
        return (self.total_cosemb_loss / self.total).item() if self.total > 0 else 0.0


    @property
    def avg_accuracy(self):
        """Accuracy (%) over all accumulated batches."""
        return (self.correct / self.total).item() if self.total > 0 else 0.0