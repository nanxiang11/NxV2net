import psutil
import torch
from d2l import torch as d2l
import numpy as np
from torch import nn
from tqdm import tqdm
import time
from TrainingLogger import TrainingLogger
from NxLoss.Loss import SoftJaccardDiceLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from lion_pytorch import Lion
from torch.optim.lr_scheduler import CyclicLR

class NXScore:
    def __init__(self, smooth=1e-5):
        self.smooth = smooth

    # 准确率
    def accuracy(self, y_pred, y):
        y_pred_flat = y_pred.flatten()
        y_flat = y.flatten()

        tp = np.sum(y_flat * y_pred_flat)
        tn = np.sum((1 - y_flat) * (1 - y_pred_flat))

        accuracy = (tp + tn) / (len(y_flat) + self.smooth)
        return accuracy

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

    # IoU
    def iou(self, y_pred, y):


        intersection = np.sum(y_pred.flatten() * y.flatten())
        unionset = np.sum(y_pred.flatten() + y.flatten()) - intersection
        single_iou = float(intersection) / float(unionset + self.smooth)
        return single_iou

    # Dice 系数
    def dice(self, y_pred, y):
        intersection = np.sum(y_pred.flatten() * y.flatten())
        unionset = np.sum(y_pred.flatten()) + np.sum(y.flatten())
        single_dice = 2 * float(intersection) / float(unionset + self.smooth)
        return single_dice

    def inference_time_and_fps(self, net, val_loader, device):
        net.eval()
        total_frames = 0
        start_time = None

        # 预热GPU，避免首次推理时间过长
        with torch.no_grad():
            for image, _ in val_loader:
                image = image.to(device, dtype=torch.float)
                _ = net(image)  # 预热
                break  # 只预热一次，退出循环

        # 开始正式计时
        start_time = time.perf_counter()

        with torch.no_grad():
            for image, _ in val_loader:
                image = image.to(device, dtype=torch.float)
                _ = net(image)
                total_frames += image.size(0)

        total_time = time.perf_counter() - start_time  # 总推理时间（秒）
        fps = total_frames / total_time  # 计算 FPS

        # 计算平均每张图片的推理时间（毫秒）
        avg_inference_time = (total_time / total_frames) * 1000 if total_frames > 0 else 0

        return total_time, fps, avg_inference_time  # 返回总时间、FPS 和平均推理时间

    # 计算内存消耗 (Memory Consumption)
    def memory_consumption(self, device):
        torch.cuda.synchronize(device)  # 确保所有操作完成
        return torch.cuda.memory_allocated(device) / (1024 ** 2)  # 返回 MB 单位的显存占用

    # 计算多个指标
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


class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0):
        """
        :param patience: 在验证指标未改善的情况下，允许的epoch数量
        :param verbose: 是否打印早停信息
        :param delta: 指标改善的最小阈值，只有超过该值才算改善
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:  # 检查是否有改善
            self.counter += 1
            if self.verbose:
                print(f'Validation score did not improve. Counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            if self.verbose:
                print(f'Validation score improved to {score}. Resetting counter.')




def train(net, train_loader, val_loader, lr, num_epoch, name, device=d2l.try_gpu(), isSave=True):
    torch.cuda.empty_cache()  # 清空未使用的显存
    torch.cuda.synchronize()  # 同步 GPU 操作，确保释放的显存被记录
    # 优化 GPU 配置
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    # 计时

    fieldnames = ['epoch', 'iou', 'dice', 'accuracy', 'recall', 'precision']
    logger = TrainingLogger(name, fieldnames)

    iou = []
    dice = []
    accuracy = []
    recall = []
    precision = []
    d1 = {'iou': 0, 'dice': 0, 'accuracy': 0, 'recall': 0, 'precision': 0}

    # 初始化权重
    def init_weights(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)
    nxscore = NXScore()

    print('训练开始', device)
    net.to(device)

    criterion = SoftJaccardDiceLoss(bce_weight=0.2)
    # optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    # 定义 Lion 优化器
    optimizer = Lion(net.parameters(), lr=lr, weight_decay=1e-2)
    # scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.75, patience=8, verbose=True)
    # 定义 CyclicLR 学习率调度器
    scheduler = CyclicLR(
        optimizer,
        base_lr=1e-6,  # 最小学习率
        max_lr=1e-3,  # 最大学习率
        step_size_up=2000,  # 学习率从 base_lr 增加到 max_lr 所需的 iteration 数
        mode='triangular2',  # 学习率模式，可选 'triangular', 'triangular2', 'exp_range'
        cycle_momentum=False  # 对于 Lion，不需要动量，因此设置为 False
    )
    early_stopping = EarlyStopping(patience=45, verbose=False)

    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epoch], figsize=(10, 6),
                            legend=['iou', 'dice'])

    epoch_tqdm = tqdm(range(num_epoch), desc="Epochs", postfix=d1, leave=True, position=0)
    for epoch in epoch_tqdm:

        iou.clear()
        dice.clear()
        accuracy.clear()
        recall.clear()
        precision.clear()

        net.train()
        for i, (image, mask) in enumerate(train_loader):
            image = image.to(device, dtype=torch.float)
            mask = mask.to(device, dtype=torch.float)
            optimizer.zero_grad()
            output = net(image)
            loss = criterion(output, mask)
            loss.backward()
            optimizer.step()
            # 更新学习率
            scheduler.step()

        net.eval()
        with torch.no_grad():
            for image, mask in val_loader:
                image = image.to(device, dtype=torch.float)
                mask = mask.to(device, dtype=torch.float)
                output = net(image)
                scoreList = nxscore.calculate_metrics(output, mask)
                iou.append(scoreList['iou'])
                dice.append(scoreList['dice'])
                accuracy.append(scoreList['accuracy'])
                recall.append(scoreList['recall'])
                precision.append(scoreList['precision'])


        miou = np.mean(iou)
        mdice = np.mean(dice)
        maccuracy = np.mean(accuracy)
        mrecall = np.mean(recall)
        mprecision = np.mean(precision)

        epoch_tqdm.set_postfix(iou=miou, dice=mdice,
                               accuracy=maccuracy, recall=mrecall, precision=mprecision)
        epoch_tqdm.update(1)
        logger.log(epoch=epoch + 1, iou=miou, dice=mdice, accuracy=maccuracy,
                   recall=mrecall, precision=mprecision)
        logger.save()
        animator.add((epoch + 1), (miou, mdice))

        early_stopping(np.mean(iou))
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    d2l.plt.show()

    # 计算推理时间和帧率
    inference_time, fps, avg_inference_time = nxscore.inference_time_and_fps(net, val_loader, device)
    memory_usage = nxscore.memory_consumption(device)

    # 将推理时间转换为毫秒
    inference_time_ms = inference_time * 1000

    # 打印推理时间（毫秒），帧率（取整），内存消耗（精确到小数点后一位）
    print(f"计算总共推理时间: {inference_time_ms:.3f} 毫秒")
    print(f"计算每张平均推理时间: {avg_inference_time:.3f} 毫秒")
    print(f"帧率: {int(fps)} FPS")
    print(f"内存消耗: {memory_usage:.3f} MB")

    if isSave:
        torch.save(net.state_dict(), name)



def save_model_parameters(model, file_path):
    """
    保存模型参数到指定路径。

    参数:
    - model: 需要保存的模型（继承自HybridBlock）
    - file_path: 保存参数的文件路径，文件扩展名通常为 .params
    """
    try:
        torch.save(model.state_dict(), file_path)
        print(f"模型参数已保存到 {file_path}")
    except Exception as e:
        print(f"保存模型参数时出错: {e}")
