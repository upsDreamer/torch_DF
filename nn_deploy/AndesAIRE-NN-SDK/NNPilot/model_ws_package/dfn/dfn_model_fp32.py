import torch
import torchvision.models as models
import onnx
import sys
import torch.fx as fx
import os
from common.onnx2torch.converter import convert
from common.fx_utils import andes_preprocessing, add_idd
now_dir=os.path.dirname(__file__)
mapping_insert_idd=['<built-in function add>','<built-in function mul>', '<built-in function sub>']
mapping_insert_idd_general=['built-in method exp of type object','built-in method log of type object']

torchDF_path = "/Work/audio_algo/zy/Pro/Pro1/DeepFilterNet/torchDF"
sys.path.append(torchDF_path)

# g_model_base_dir = torchDF_path + "/test_dir/base_dir_1_1M_d3/"
g_model_base_dir = "/Work/audio_algo/zy/Pro/DeepFilterNet/DeepFilterNet/base_dir_8_4M_d6"

from torch_df_streaming import TorchDFPipeline

def return_fp32_model():


    with torch.no_grad():
        print("dfn")
        # onnx_model_path=now_dir+'/base_1_1M_d3.onnx'
        # model_test = onnx.load(onnx_model_path)
        # model_test = convert(model_test)
        # model_test=andes_preprocessing(model_test)
        # model_test=add_idd(model_test)
        streaming_pipeline = TorchDFPipeline(always_apply_all_stages=True, device='cpu', model_base_dir = g_model_base_dir)
        torch_df = streaming_pipeline.torch_streaming_model
        states = streaming_pipeline.states
        atten_lim_db = streaming_pipeline.atten_lim_db        
        torch_df = andes_preprocessing(torch_df)
        torch_df = add_idd(torch_df)
    print(torch_df)
    torch_df.eval()
    return torch_df

if __name__ == "__main__":
    model = return_fp32_model()