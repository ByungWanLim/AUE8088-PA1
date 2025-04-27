from torchmetrics import Metric
import torch

# [TODO] Implement this!
class MyF1Score(Metric):
    def __init__(self, num_classes=200):
        super().__init__()
        self.num_classes = num_classes
        # TP, FP, FN을 클래스별로 누적
        self.add_state('true_positives', default=torch.zeros(num_classes), dist_reduce_fx='sum')
        self.add_state('false_positives', default=torch.zeros(num_classes), dist_reduce_fx='sum')
        self.add_state('false_negatives', default=torch.zeros(num_classes), dist_reduce_fx='sum')

    def update(self, preds, target):
        # preds: [B, C] -> argmax로 예측 클래스 인덱스 [B]
        pred_labels = torch.argmax(preds, dim=1)

        # shape 확인
        assert pred_labels.shape == target.shape, \
            f"Predicted labels shape {pred_labels.shape} does not match target shape {target.shape}"

        # 클래스별 TP, FP, FN 계산
        for c in range(self.num_classes):
            # TP: 예측과 타겟이 모두 클래스 c인 경우
            tp = torch.sum((pred_labels == c) & (target == c))
            # FP: 예측은 c인데 타겟은 c가 아닌 경우
            fp = torch.sum((pred_labels == c) & (target != c))
            # FN: 타겟은 c인데 예측은 c가 아닌 경우
            fn = torch.sum((pred_labels != c) & (target == c))

            self.true_positives[c] += tp
            self.false_positives[c] += fp
            self.false_negatives[c] += fn

    def compute(self):
        # 클래스별 Precision, Recall 계산
        precision = torch.zeros(self.num_classes)
        recall = torch.zeros(self.num_classes)
        for c in range(self.num_classes):
            tp = self.true_positives[c]
            fp = self.false_positives[c]
            fn = self.false_negatives[c]

            # Precision = TP / (TP + FP), 0으로 나누기 방지
            precision[c] = tp / (tp + fp + 1e-10)
            # Recall = TP / (TP + FN)
            recall[c] = tp / (tp + fn + 1e-10)

        # 클래스별 F1 스코어: 2 * (Precision * Recall) / (Precision + Recall)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)

        # Macro F1: 클래스별 F1 평균
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
        correct = torch.sum(pred_labels == target)  # True인 경우 합산

        # Accumulate to self.correct
        self.correct += correct

        # Count the number of elements in target
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total.float()
