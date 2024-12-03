import os
import json
import pandas as pd

# 定义文件夹路径
folder_path = './vcnu_expansibility/tools/diffusion_tuning_llm/roberta-base/sst2'  # 替换为你的文件夹路径
excel_file_path = './vcnu_expansibility/tools/diffusion_tuning_llm/roberta-base/sst2/roberta-sst2.xlsx'  # 替换为你希望保存的 Excel 文件路径

# 初始化一个空的 DataFrame
df = pd.DataFrame()

# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    if filename.endswith('.json'):
        file_path = os.path.join(folder_path, filename)
        
        # 读取 JSON 文件
        with open(file_path, 'r') as file:
            data = json.load(file)
        
        # 提取 log_history 中的 loss 值
        log_history = data.get('log_history', [])
        loss_values = [entry.get('loss', None) for entry in log_history]

        # 将 loss 值添加到 DataFrame 中，列名为文件名
        df[filename] = loss_values

# 保存到 Excel 文件
df.to_excel(excel_file_path, index=False)

print(f"Loss values from all JSON files have been saved to {excel_file_path}")




# # -----------------------------------------------删除空的行------------------------------------------
# import pandas as pd

# # 读取Excel文件
# file_path ='./vcnu_expansibility/tools/diffusion_tuning_llm/roberta-base/mnli/roberta-mnli.xlsx'  # 替换为你的文件路径
# df = pd.read_excel(file_path)

# # 删除所有值为空的行
# df_cleaned = df.dropna(how='all')

# # 保存处理后的数据到新的Excel文件
# output_file_path = './vcnu_expansibility/tools/diffusion_tuning_llm/roberta-base/mnli/roberta-mnli-delet.xlsx'  # 替换为你希望保存的文件路径
# df_cleaned.to_excel(output_file_path, index=False)

# print("处理完成，空行已删除。")

# # -----------------------------------------------添加噪声--------------------------------------------------------------------
# import numpy as np

# # 假设数据已经从文件中读取到一个列表中
# data = []

# with open('./vcnu_expansibility/tools/test.txt', 'r') as file:
#     for line in file:
#         data.append(float(line.strip()))

# # 添加随机噪声，均值为0，标准差为0.1（方差为0.01）
# noisy_data = [x + np.random.normal(0, 0.01) for x in data]

# # 将添加了噪声的数据保存到新的文件中
# output_file = './vcnu_expansibility/tools/noisy_test.txt'
# with open(output_file, 'w') as file:
#     for value in noisy_data:
#         file.write(f"{value:.4f}\n")

# print(f"添加噪声后的数据已保存到 {output_file}")