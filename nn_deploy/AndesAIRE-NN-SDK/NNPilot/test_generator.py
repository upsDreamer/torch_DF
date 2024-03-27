import sys
import numpy as np
import argparse
import yaml
import torch
device = torch.device( "cpu")
 
from tqdm import tqdm, trange
import os
 
from backend_converter.backend_converter.pytorch_reader import PytorchGraph
from backend_converter.tflite_parser import tflite_reader, tflite_generator
 
from tensorflow.lite.python import interpreter as interpreter_wrapper
 
import importlib
import pathlib
 
 
 
 
parser = argparse.ArgumentParser(description="Convert Input data to test_bin")
parser.add_argument("--model_ws_pkg", type=str, default="none", help="specify the model_workspace_package path")
parser.add_argument("--model_pkg", type=str, default="none", help="specify the model_package path device")
parser.add_argument("--save_path", type=str, default="./output/", help="specify the model_package path device")
args = parser.parse_args()
 
 
# model_path="tyolov2/model.tflite"
 
if args.model_ws_pkg=='none':
    print("give me ws")
    exit()
else:
    ws_path=args.model_ws_pkg[:-1] if args.model_ws_pkg[-1]=="/" else args.model_ws_pkg
    ws_path=str(pathlib.Path(ws_path))
    ws_path=ws_path.replace("/",".")
    data_ws=importlib.import_module(ws_path+'.dataset')
    _,val_dataloader,_,_ = data_ws.return_dataset()
   
    data_config = data_ws.dataset_cfg()
    # eval_ws=importlib.import_module(ws_path+'.evaluation')
 
if args.model_pkg=='none':
    print("give me tflite path")
    exit()
 
model_path=args.model_pkg+"/model.tflite"
save_path=args.save_path
interpreter = interpreter_wrapper.Interpreter(model_path=model_path)
 
if os.path.isdir(save_path):
    print("no need create, overwrite directory")
else:
    print("create new directory")
    os.makedirs(save_path)
data_ws.prepare_testbin(interpreter, val_dataloader, save_path)
