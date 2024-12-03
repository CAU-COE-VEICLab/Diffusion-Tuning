# 定义计算公式
def calculate_value(a, b):
    return a * 60 + b

# 读取文件
input_file_path = './vcnu_expansibility/tools/diffusion_training_time/test.txt'
output_file_path = './vcnu_expansibility/tools/diffusion_training_time/output.txt'

with open(input_file_path, 'r') as file:
    lines = file.readlines()

# 应用公式并存储结果
results = []
for line in lines:
    try:
        # 去除行尾的换行符，并替换可能的逗号为冒号
        line = line.strip().replace(',', ':')
        # 拆分字符串
        parts = line.split(':')
        if len(parts) == 2:
            a = float(parts[0])
            b = float(parts[1])
            result = calculate_value(a, b)
            results.append(result)
        else:
            print(f"Warning: Skipping invalid line '{line}'")
    except ValueError:
        print(f"Warning: Skipping invalid line '{line}'")

# 将结果写入输出文件
with open(output_file_path, 'w') as file:
    for result in results:
        file.write(f"{result}\n")

print("计算完成，结果已保存到 output.txt")