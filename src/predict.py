import jieba
import torch
import config
from model import ReviewAnalyzeModel
from src.tokenizer import JiebaTokenizer


def predict_batch(model, inputs):
    """
    批量预测
    :param model: 模型
    :param inputs: 输入,shape:[batch_size, sql_len]
    :return: 预测结果,shape:[batch_size]
    """
    model.eval()
    with torch.no_grad():
        output = model(inputs)
        # output.shape: [batch_size, 1]
        output = output.squeeze(-1)
    batch_result = torch.sigmoid(output)

    return batch_result.tolist()


def predict(text, model, tokenizer, device):
    # 1. 处理输入
    indexes = tokenizer.encode(text, config.SEN_LEN)
    input_tensor = torch.tensor([indexes], dtype=torch.long)
    input_tensor = input_tensor.to(device)
    # input_tensor.shape: [batch_size, seq_len]

    # 2.预测逻辑
    batch_result = predict_batch(model, input_tensor)
    return batch_result[0]


def run_predict():
    # 准备资源
    # 1. 确定设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2.词表
    tokenizer = JiebaTokenizer.from_vocab(config.MODELS_DIR / 'vocab.txt')
    print("词表加载成功")

    # 3. 模型
    model = ReviewAnalyzeModel(tokenizer.vocab_size, tokenizer.pad_token_index).to(device)
    model.load_state_dict(torch.load(config.MODELS_DIR / 'best.pth'))
    print("模型加载成功")

    print("欢迎使用情感分析模型")

    while True:
        user_input = input("> ")
        if user_input in ['q', 'quit']:
            print("欢迎下次再来")
            break
        if user_input.strip() == '':
            print("请输入内容")
            continue

        result = predict(user_input, model, tokenizer, device)
        if result > 0.5:
            print(f"没急。置信度：{result:3f}")
        else:
            print(f"急了。置信度：{1-result:3f}")


if __name__ == '__main__':
    run_predict()
