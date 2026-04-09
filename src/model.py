from torch import nn
import torch
import config

class ReviewAnalyzeModel(nn.Module):
    def __init__(self, vocab_size, padding_index):
        super().__init__()
        # padding_idx用于在随机初始化时将<pad>设为全0
        self.embedding = nn.Embedding(vocab_size, config.EMBEDDING_DIM, padding_idx = padding_index)
        self.lstm = nn.LSTM(input_size=config.EMBEDDING_DIM,
                            hidden_size=config.HIDDEN_DIM,
                            batch_first=True)
        # out_features用于指定输出维度，此处为二分类
        self.linear = nn.Linear(config.HIDDEN_DIM, 1)

    def forward(self, x):
        # x.shape:[batch_size, seq_len]
        embed = self.embedding(x)
        # embed.shape:[batch_size, seq_len, embedding_dim]
        lstm_out, (_, _) =  self.lstm(embed) # lstm同时返回output, (h_n, c_n)
        # output.shape:[batch_size, seq_len, hidden_dim]

        # 获取每个样本有价值的最后一个token
        batch_indexes = torch.arange(0, lstm_out.shape[0])
        length = (x!=self.embedding.padding_idx).sum(dim=1) # 统计每行实际长度(向量化操作)
        last_hidden = lstm_out[batch_indexes, length-1] # 操作张量时会默认保留剩余维度
        # last_hidden.shape:[seq_len, hidden_dim]

        output = self.linear(last_hidden)
        return output

