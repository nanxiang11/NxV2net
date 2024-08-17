import numpy as np

class NXScore:
    def __init__(self, smooth=1e-5):
        self.smooth = smooth
    # iou
    def iou(self, y_pred, y):
        intersection = np.sum(y_pred.flatten() * y.flatten())
        unionset = np.sum(y_pred.flatten() + y.flatten()) - intersection
        single_iou = float(intersection) / float(unionset + self.smooth)
        return single_iou

    # dice系数
    def dice(self, y_pred, y):
        intersection = np.sum(y_pred.flatten() * y.flatten())
        unionset = np.sum(y_pred.flatten()) + np.sum(y.flatten())
        single_dice = 2 * float(intersection) / float(unionset + self.smooth)
        return single_dice


    # Precision
    def precision(self, y_pred, y):
        y_pred_flat = y_pred.flatten()
        y_flat = y.flatten()

        tp = np.sum(y_flat * y_pred_flat)
        fp = np.sum((1 - y_flat) * y_pred_flat)

        precision = float(tp) / float(tp + fp + self.smooth)
        return precision

    # Recall
    def recall(self, y_pred, y):
        y_pred_flat = y_pred.flatten()
        y_flat = y.flatten()

        tp = np.sum(y_flat * y_pred_flat)
        fn = np.sum(y_flat * (1 - y_pred_flat))

        recall = tp / (tp + fn + self.smooth)
        return recall

    # Accuracy
    def accuracy(self, y_pred, y):
        y_pred_flat = y_pred.flatten()
        y_flat = y.flatten()

        tp = np.sum(y_flat * y_pred_flat)
        tn = np.sum((1 - y_flat) * (1 - y_pred_flat))

        accuracy = (tp + tn) / (len(y_flat) + self.smooth)
        return accuracy

    def calculate_metrics(self, y_pred, y, metrics=('iou', 'dice', 'precision', 'recall', 'accuracy')):
        y_pred_np = y_pred.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()

        results = {}
        for metric_name in metrics:
            if metric_name == 'iou':
                value = self.iou(y_pred_np, y_np)
            elif metric_name == 'dice':
                value = self.dice(y_pred_np, y_np)
            elif metric_name == 'precision':
                value = self.precision(y_pred_np, y_np)
            elif metric_name == 'recall':
                value = self.recall(y_pred_np, y_np)
            elif metric_name == 'accuracy':
                value = self.accuracy(y_pred_np, y_np)
            else:
                raise ValueError(f"Unsupported metric: {metric_name}")
            results[metric_name] = value
        return results
