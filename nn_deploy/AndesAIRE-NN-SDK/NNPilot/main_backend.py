import sys
#from andesConvert.convert_pytorch_model import ConvertModel
from andesQuant.quant_func import quantize_model_loader
#from andesQuant.cls_quant_func import collect_partial_state

import numpy as np
import argparse
import yaml
import torch
from tqdm import tqdm, trange
import os

from backend_converter.backend_converter.pytorch_reader import PytorchGraph
from backend_converter.tflite_parser import tflite_reader, tflite_generator

# from ethosu.vela.vela import pytorch_process
from tensorflow.lite.python import interpreter as interpreter_wrapper

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
import importlib
import pathlib


parser = argparse.ArgumentParser(description="Convert fake-quant model to tflite QNN model")
parser.add_argument("--name", type=str, default='model.tflite')
parser.add_argument('--quant_format', "-f", type=str, default='None')
parser.add_argument("--model_ws_pkg", type=str, default="none", help="specify the model_workspace_package path")
parser.add_argument("--model_pkg", type=str, default="none", help="specify the model_package path device")
parser.add_argument("--device", type=str, default="cpu", help="specify the gpu device")
parser.add_argument("--is_tflite", action='store_true', help="is it convert from tflite")
parser.add_argument("--remove_list", action='store_true',help="enable parse bacekend_remove_list in model_ws_package")
args = parser.parse_args()

device = args.device
print("gpu_device={}".format(device))

model_fq, DAG_object, quant_config = quantize_model_loader(args.model_pkg, export=True)
print(model_fq)

if args.model_ws_pkg=='none':
    pass
else: 
    ws_path=args.model_ws_pkg[:-1] if args.model_ws_pkg[-1]=="/" else args.model_ws_pkg
    ws_path=str(pathlib.Path(ws_path))
    ws_path=ws_path.replace("/",".")
    data_ws=importlib.import_module(ws_path+'.dataset')
    tra_dataloader,val_dataloader,cal_dataloader,cos_dataloader = data_ws.return_dataset()
    data_config = data_ws.dataset_cfg()
    eval_ws=importlib.import_module(ws_path+'.evaluation')
    # auc=eval_ws.inference_FQ(model_fq,val_dataloader,data_config,device,symm=quant_config['a_symmetry'])


"""
prepare tflite_quant_config for tflite converter
"""

tflite_quant_config = {
    'float': False,
    'bits_weight': quant_config['bits_weight'],
    'bits_activation':quant_config['bits_activation'],
    'bits_bias': quant_config['bits_bias'],
    'per_channel': quant_config['per_channel'],
    'w_symmetry': quant_config['w_symmetry'],
    'a_symmetry': quant_config['a_symmetry']
}

for key in tflite_quant_config:
    if key in quant_config:
        tflite_quant_config[key] = quant_config[key]

if args.quant_format == 'float':
    tflite_quant_config['float'] = True

print(tflite_quant_config)

fp32_input_range_array = quant_config.get('fp32_range_array', None)

"""
Remove layer cannot quant, such as softmax define in {$model_ws_pkg}/backend_remove_list.yaml
"""
if os.path.isfile(args.model_ws_pkg+"/backend_remove_list.yaml"):
    with open(args.model_ws_pkg+"/backend_remove_list.yaml", 'r') as remove_list_file:
        remove_list = yaml.load(remove_list_file, Loader=yaml.FullLoader)['remove_list']
    for target in remove_list:
        PytorchGraph.CHECK_REMOVE_LIST.append(target)

"""
Start generate model.tflite
"""
model_graph = PytorchGraph(
    model_fq, DAG_object, tflite_quant_config,
    quant_config['fp32_min'], quant_config['fp32_max'],
    args.is_tflite, fp32_input_range_array
)
nng = tflite_reader.read_pytorch(args.name, model_graph)
'''if os.path.abspath(folder_name) != os.path.abspath(os.getcwd()):
    args.name = folder_name + '/model_' + backend_get_file_name(folder_name) + '.tflite'''
args.name = os.path.join(args.model_pkg, args.name)
tflite_generator.write_tflite(nng, args.name)

"""
write to pre_shift.yaml if pre_shift.yaml exist
"""

if (
    quant_config.get("preshift_path", "None") != "None"
    and quant_config.get("preshift_path", "None\n") != "None\n"
):
    with open(os.path.join(args.model_pkg, "pre_shift.yaml"), 'r') as preshift_file:
        preshift = yaml.load(preshift_file.read(), Loader=yaml.FullLoader)

    tflite_preshift = {}
    for op in nng.report_all_ops():
        if "est_preshift" in op.attrs:
            tflite_preshift[op.name] = op.attrs["est_preshift"]

    preshift['tflite_preshift_info'] = tflite_preshift
    with open(os.path.join(args.model_pkg, "pre_shift.yaml"), 'w') as preshift_file:
        yaml.dump(preshift, preshift_file)

model_path = args.name
# model_path = args.model_pkg + "/model.tflite"

"""
Prepare to evaluation tflite model
"""
if args.model_ws_pkg=='none':
    print("no evaluation logic provide")
    print('only generate model.tflite')
    exit()
interpreter = interpreter_wrapper.Interpreter(model_path=model_path)
auc = eval_ws.inference_Backend(
    interpreter, val_dataloader, data_config, device, symm=quant_config['a_symmetry']
)
print(auc)

with open(os.path.join(args.model_pkg, "quantize_config.yaml"), 'r') as f:
    quant_config = yaml.load(f, Loader=yaml.FullLoader)

quant_config['acc_backend'] = float(auc)

with open(os.path.join(args.model_pkg, "quantize_config.yaml"), 'w') as f:
    yaml.dump(quant_config, f, default_flow_style=False, sort_keys=False, width=100)
