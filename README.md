# 评论分析LSTM模型

一个使用LSTM实现的中文评论情感分析系统。

## 项目结构

```
.
├── data/
│   ├── raw/              # 原始数据
│   │   └── online_shopping_10_cats.csv
│   └── processed/        # 处理后的数据
│       ├── train.jsonl
│       └── test.jsonl
├── models/               # 保存的模型和词表
│   ├── best.pth
│   └── vocab.txt
├── logs/                 # TensorBoard日志
├── src/                  # 源代码
│   ├── config.py         # 配置文件
│   ├── dataset.py        # 数据集处理
│   ├── model.py          # LSTM模型定义
│   ├── train.py          # 训练脚本
│   ├── evaluate.py       # 评估脚本
│   ├── predict.py        # 预测脚本
│   ├── process.py        # 数据处理脚本
│   └── tokenizer.py      # 分词器
└── README.md             # 项目说明
```

## 环境设置

### 1. 安装依赖

```bash
pip install torch pandas scikit-learn jieba tqdm tensorboard
```

### 2. 数据准备

- 原始数据放在 `data/raw/` 目录下
- 运行 `python src/process.py` 处理数据到 `data/processed/` 目录

## 使用方式

### 处理数据

```bash
cd src
python process.py
```

### 训练模型

```bash
cd src
python train.py
```

### 评估模型

```bash
cd src
python evaluate.py
```

### 预测

```bash
cd src
python predict.py
```

## 配置说明

编辑 `src/config.py` 调整以下参数：

- `SEN_LEN`: 句子长度 (默认: 128)
- `BATCH_SIZE`: 批次大小 (默认: 64)
- `EMBEDDING_DIM`: 词嵌入维度 (默认: 128)
- `HIDDEN_DIM`: 隐层维度 (默认: 256)
- `LEARNING_RATE`: 学习率 (默认: 1e-3)
- `EPOCHS`: 训练轮次 (默认: 10)

## 模型架构

```
Input → Embedding → LSTM → Linear → Output
```

- **Embedding**: 将词索引转换为密集向量
- **LSTM**: 处理序列信息，捕捉长期依赖
- **Linear**: 输出层进行二分类预测

## 环境要求

- Python 3.7+
- PyTorch 1.9+
- pandas
- scikit-learn
- jieba
- tqdm
- tensorboard

## 注意事项

- 大型模型文件 (*.pth) 不被版本控制跟踪
- TensorBoard日志保存在 `logs/` 目录
- 使用 CUDA 加速（如果可用）

## 日志查看

```bash
tensorboard --logdir=logs
```

然后访问 `http://localhost:6006`