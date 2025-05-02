from torchmetrics import Metric
import torch

# [TODO] Implement this!
class MyF1Score(Metric):
    def __init__(self, num_classes=200):
        super().__init__()
        self.num_classes = num_classes
        self.add_state('true_positives', default=torch.zeros(num_classes), dist_reduce_fx='sum')
        self.add_state('false_positives', default=torch.zeros(num_classes), dist_reduce_fx='sum')
        self.add_state('false_negatives', default=torch.zeros(num_classes), dist_reduce_fx='sum')

    def update(self, preds, target):
        pred_labels = torch.argmax(preds, dim=1)

        assert pred_labels.shape == target.shape, \
            f"Predicted labels shape {pred_labels.shape} does not match target shape {target.shape}"

        for c in range(self.num_classes):
            tp = torch.sum((pred_labels == c) & (target == c))
            fp = torch.sum((pred_labels == c) & (target != c))
            fn = torch.sum((pred_labels != c) & (target == c))

            self.true_positives[c] += tp
            self.false_positives[c] += fp
            self.false_negatives[c] += fn

    def compute(self):
        precision = torch.zeros(self.num_classes)
        recall = torch.zeros(self.num_classes)
        for c in range(self.num_classes):
            tp = self.true_positives[c]
            fp = self.false_positives[c]
            fn = self.false_negatives[c]

            precision[c] = tp / (tp + fp + 1e-10)
            recall[c] = tp / (tp + fn + 1e-10)

        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)

        return torch.mean(f1_scores)
    
class MyAccuracy(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('correct', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds, target):
        # [TODO] The preds (B x C tensor), so take argmax to get index with highest confidence
        pred_labels = torch.argmax(preds, dim=1)  # [B, C] -> [B]

        # [TODO] check if preds and target have equal shape
        assert pred_labels.shape == target.shape, \
            f"Predicted labels shape {pred_labels.shape} does not match target shape {target.shape}"

        # [TODO] Cound the number of correct prediction
        correct = torch.sum(pred_labels == target) 

        self.correct += correct

        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total.float()
