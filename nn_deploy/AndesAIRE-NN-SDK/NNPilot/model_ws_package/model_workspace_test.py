import torch
from mobilenet_v2_ssd_lite import model_fp32
from mobilenet_v2_ssd_lite import dataset
from mobilenet_v2_ssd_lite import evaluation


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

"""
Prepare Floating Point FP32 model
"""

model_fp32=model_fp32.return_fp32_model()

"""
Prepare following data pairs 
training data pairs(tra_dataloader) 
validation data pairs(tra_dataloader)
calibration data pairs(tra_dataloader)
cosine metrics  data pairs(cos_dataloader)
"""

tra_dataloader,val_dataloader,cal_dataloader,cos_dataloader=dataset.return_dataset()

"""
Prepare golden data pairs
"""

golden_dataloader=dataset.prepare_golden(save_sample=False)

"""
model Floating point evaluation
"""
evaluation.inference_FP32(model_fp32, val_dataloader, device)
