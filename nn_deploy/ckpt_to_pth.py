import torch
import sys
from collections import OrderedDict
 
# 加载ckpt文件并转换为字典形式
checkpoint = torch.load(sys.argv[1])
state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
 
# 创建空的OrderedDict对象
new_state_dict = OrderedDict()
 
# 遍历原始状态字典中的每个key-value对
for key, value in state_dict.items():
    new_key = '.'.join(key) # 如果有多层结构，则通过点号连接各层名称
    new_state_dict[new_key] = value
 
# 打印新的状态字典
print(new_state_dict)