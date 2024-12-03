import re

# 读取日志文件
log_file_path = '/mnt/f/1.中国农大/数据集/vcnu_pretrain_weight/vcnu_expansibility/vcnu-swin/finetune/diffusion_finetune/convnext/adapter/adapter-convnext-b-22kto1k-diffusion-finetune-step_stage3/log_rank0.txt'
output_file_path = './extracted_accuracy.txt'

# 正则表达式模式，用于匹配 INFO * Acc@1 和 Acc@5 的值
pattern = re.compile(r'INFO  \* Acc@1 (\d+\.\d+) Acc@5 (\d+\.\d+)')

# 用于存储提取的准确率
accuracies = []

# 读取日志文件并提取准确率
with open(log_file_path, 'r') as file:
    for line in file:
        match = pattern.search(line)
        if match:
            acc1, acc5 = match.groups()
            accuracies.append(f'Acc@1: {acc1}, Acc@5: {acc5}\n')

# 将提取的准确率写入新的文本文件
with open(output_file_path, 'w') as output_file:
    output_file.writelines(accuracies)

print(f'Accuracy values have been extracted and saved to {output_file_path}')