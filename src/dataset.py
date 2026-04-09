import json
import torch
import config
import random

from torch.utils.data import Dataset, DataLoader


# 定义dataset
class ReviewAnalyzeDataset(Dataset):
    def __init__(self, file_path):
        self.data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            # jsonl文件固定方法
            for line in f:
                sample = json.loads(line)
                self.data.append(sample)
        # 是否需要缩小数据集
        # self.data = random.sample(self.data, 10000)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # 取每行数据
        sample = self.data[index]
        input_seq = sample['review']
        target_seq = sample['label']
        # 转张量，需要注意类型要与损失函数要求一致
        input_tensor = torch.tensor(input_seq, dtype=torch.long)
        target_tensor = torch.tensor(target_seq, dtype=torch.float) # BCEWithLogitsLoss要求float
        return input_tensor, target_tensor

# 提供获取dataloader的方法
def get_dataloader(train=True):
    shuffle = train
    batch_size = config.BATCH_SIZE
    file_path = config.PROCESSED_DATA_DIR / 'train.jsonl' if train else config.PROCESSED_DATA_DIR / 'test.jsonl'

    dataset = ReviewAnalyzeDataset(file_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=config.NUM_WORKERS)
    return dataloader

if __name__ == '__main__':
    train_dataloader = get_dataloader()
    test_dataloader = get_dataloader(train=False)
    print(len(train_dataloader))
    print(len(test_dataloader))

    for input_tensor, target_tensor in train_dataloader:
        print(input_tensor.shape) # [batch_size, seq_len]
        print(target_tensor.shape) # [batch_size]
        break