import pandas as pd
import os


class TrainingLogger:
    def __init__(self, filename, fieldnames):
        """
        初始化记录器。

        :param filename: 要保存的CSV文件名
        :param fieldnames: 字段名列表
        """
        self.filename = filename
        self.fieldnames = fieldnames
        self.records = []

        # 如果文件存在，读取现有数据
        if os.path.exists(self.filename):
            self.df = pd.read_csv(self.filename)
        else:
            self.df = pd.DataFrame(columns=self.fieldnames)

    def log(self, **kwargs):
        """
        记录训练过程中的数据。

        :param kwargs: 关键字参数，对应字段名及其值
        """
        record = {field: kwargs.get(field, None) for field in self.fieldnames}
        self.records.append(record)

    def save(self):
        """
        将记录的数据保存到CSV文件中。如果文件已存在，则追加数据。
        """
        new_df = pd.DataFrame(self.records)

        if not self.df.empty:
            self.df = self.df[self.df['epoch'].isin(new_df['epoch']) == False]

        self.df = pd.concat([self.df, new_df], ignore_index=True)
        self.df = self.df.sort_values(by='epoch')

        self.df.to_csv('./画图/' + self.filename + '.csv', index=False)

        self.records = []


# 示例用法
if __name__ == "__main__":
    fieldnames = ['epoch', 'loss', 'accuracy']
    logger = TrainingLogger('training_log.csv', fieldnames)

    # 模拟记录一些训练数据
    for epoch in range(1, 6):
        loss = 0.5 / epoch
        accuracy = 0.8 + 0.04 * epoch
        logger.log(epoch=epoch, loss=loss, accuracy=accuracy)

    # 保存数据到CSV文件
    logger.save()
