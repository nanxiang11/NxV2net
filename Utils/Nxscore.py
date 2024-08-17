import numpy as np
from scipy.spatial.distance import directed_hausdorff

class NXScore:
    def __init__(self, smooth=1e-5):
        self.smooth = smooth

    # 准确率
    def acc(self, y_pred, y):
        tp = np.sum(y_pred.flatten() == y.flatten(), dtype=y_pred.dtype)
        total = len(y_pred.flatten())
        single_acc = float(tp) / float(total)
        return single_acc

    # iou
    def iou(self, y_pred, y):
        intersection = np.sum(y_pred.flatten() * y.flatten())
        unionset = np.sum(y_pred.flatten() + y.flatten()) - intersection
        single_iou = float(intersection) / float(unionset + self.smooth)
        return single_iou

    # dice系数/f1
    def dice(self, y_pred, y):
        intersection = np.sum(y_pred.flatten() * y.flatten())
        unionset = np.sum(y_pred.flatten()) + np.sum(y.flatten())
        single_dice = 2 * float(intersection) / float(unionset + self.smooth)
        return single_dice

    # 灵敏度
    def sens(self, y_pred, y):
        tp = np.sum(y_pred.flatten() * y.flatten())
        actual_positives = np.sum(y.flatten())
        single_sens = float(tp) / float(actual_positives + self.smooth)
        return single_sens

    # 特异性
    def spec(self, y_pred, y):
        true_neg = np.sum((1 - y.flatten()) * (1 - y_pred.flatten()))
        total_neg = np.sum((1 - y.flatten()))
        single_spec = float(true_neg) / float(total_neg + self.smooth)
        return single_spec

    def auc(self, y_pred, y):
        y_pred_flat = y_pred.flatten()
        y_flat = y.flatten()
        sorted_indices = np.argsort(y_pred_flat)
        y_pred_sorted = y_pred_flat[sorted_indices]
        y_sorted = y_flat[sorted_indices]

        auc_score = 0
        n_positive = np.sum(y_sorted)
        n_negative = len(y_sorted) - n_positive

        tp = 0
        fp = 0
        prev_fp = 0
        prev_tp = 0
        prev_threshold = -np.inf

        for i in range(len(y_sorted)):
            if y_pred_sorted[i] != prev_threshold:
                auc_score += (fp + prev_fp) / 2 * (tp + prev_tp) / 2 / (n_negative * n_positive)
                prev_fp = fp
                prev_tp = tp
                prev_threshold = y_pred_sorted[i]

            if y_sorted[i] == 1:
                tp += 1
            else:
                fp += 1

        auc_score += (fp + prev_fp) / 2 * (tp + prev_tp) / 2 / (n_negative * n_positive)
        return auc_score

    # 衡量的是在预测边界和真实边界之间，95% 的距离误差
    def hd95(self, y_pred, y):
        y_pred = y_pred.astype(np.bool_)
        y = y.astype(np.bool_)

        # Compute hd95 for each pair of y_pred and y in the batch
        batch_size = y_pred.shape[0]
        hd95_values = []
        for i in range(batch_size):
            # Reshape to (height, width)
            y_pred_2d = y_pred[i].reshape(-1, y_pred.shape[-1])
            y_2d = y[i].reshape(-1, y.shape[-1])
            hd1 = directed_hausdorff(y_pred_2d, y_2d)[0]
            hd2 = directed_hausdorff(y_2d, y_pred_2d)[0]
            hd95_values.append(max(hd1, hd2))

        return np.mean(hd95_values)  # or np.min(hd95_values) depending on your metric definition

    # 是 HD95 的平均值，同时考虑了从预测到真实和从真实到预测两个方向的距离。
    def ahd95(self, y_pred, y):
        y_pred = y_pred.astype(np.bool_)
        y = y.astype(np.bool_)

        # Compute ahd95 for each pair of y_pred and y in the batch
        batch_size = y_pred.shape[0]
        ahd95_values = []
        for i in range(batch_size):
            # Reshape to (height, width)
            y_pred_2d = y_pred[i].reshape(-1, y_pred.shape[-1])
            y_2d = y[i].reshape(-1, y.shape[-1])
            hd1 = directed_hausdorff(y_pred_2d, y_2d)[0]
            hd2 = directed_hausdorff(y_2d, y_pred_2d)[0]
            ahd95_values.append((hd1 + hd2) / 2.0)

        return np.mean(ahd95_values)

    def calculate_metrics(self, y_pred, y,
                          metrics=('acc', 'iou', 'dice', 'sens', 'spec', 'auc', 'hd95', 'ahd95')):
        y_pred_np = y_pred.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()

        results = {}
        for metric_name in metrics:
            if metric_name == 'acc':
                value = self.acc(y_pred_np, y_np)
            elif metric_name == 'iou':
                value = self.iou(y_pred_np, y_np)
            elif metric_name == 'dice':
                value = self.dice(y_pred_np, y_np)
            elif metric_name == 'sens':
                value = self.sens(y_pred_np, y_np)
            elif metric_name == 'spec':
                value = self.spec(y_pred_np, y_np)
            elif metric_name == 'auc':
                value = self.auc(y_pred_np, y_np)
            elif metric_name == 'hd95':
                value = self.hd95(y_pred_np, y_np)
            elif metric_name == 'ahd95':
                value = self.ahd95(y_pred_np, y_np)
            else:
                raise ValueError(f"Unsupported metric: {metric_name}")
            results[metric_name] = value
        return results