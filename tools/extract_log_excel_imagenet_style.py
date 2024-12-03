import os
import pandas as pd
import re
# 定义根路径
root_path = 'vcnu_expansibility/tools/diffusion_tuning'

# 遍历的子路径列表
sub_paths = ['convnext', 'swin', 'smt']

# 子子路径列表
sub_sub_paths = ['adapter', 'fft']

# 正则表达式模式，用于匹配 INFO * Acc@1 和 Acc@5 的值
pattern = re.compile(r'INFO  \* Acc@1 (\d+\.\d+) Acc@5 (\d+\.\d+)')


# 遍历每个子路径
for sub_path in sub_paths:
    # 遍历每个子子路径
    for sub_sub_path in sub_sub_paths:
        # 构建完整的路径
        full_path = os.path.join(root_path, sub_path, sub_sub_path)
        
        # 检查路径是否存在
        if not os.path.exists(full_path):
            print(f"路径 {full_path} 不存在")
            continue
        
        # 初始化一个字典来存储数据
        data = {}
        
        # 记录最长列的长度
        max_length = 0

        # 遍历当前子子路径下的所有txt文件
        for filename in os.listdir(full_path):
            if filename.endswith('.txt'):
                file_path = os.path.join(full_path, filename)
                # 用于存储提取的准确率
                accuracies = []
                with open(file_path, 'r', encoding='utf-8') as file:
                    i = 0
                    for line in file:
                        match = pattern.search(line)
                        if match:
                            acc1, acc5 = match.groups()
                            accuracies.append(acc1)
                            i += 1
                    
                    data[filename] = accuracies

                    lines_ = file.readlines()
                    # 更新最长列的长度
                    max_length = max(max_length, i)
        
        # 填充较短的列
        for key in data:
            data[key].extend([''] * (max_length - len(data[key])))
        
        # 将字典转换为DataFrame
        df = pd.DataFrame(data)
        
        # 保存为Excel文件
        excel_path = os.path.join(full_path, f'{sub_sub_path}.xlsx')
        df.to_excel(excel_path, index=False)
        print(f"已生成文件: {excel_path}")