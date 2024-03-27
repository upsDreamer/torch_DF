#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 09 13:00:10 2022

@author: evansong
"""
import os
import torch
import numpy as np
import importlib
import pathlib
from torch.utils.data import DataLoader, Subset
import argparse
import yaml
from common import andesPickle
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
from andesQuant import quant_func
from andesQuant.sim_utils.utils import FQ2SIM,insert_DequantStub
from torch.quantization.fake_quantize import disable_observer,disable_fake_quant,enable_observer,enable_fake_quant


device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser(description="Quant model loader Interface.")
parser.add_argument("--model_ws_pkg", type=str, default="none")
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--model_pkg", type=str, default="none")
parser.add_argument("--preshift", action='store_true')
args = parser.parse_args()



    
if __name__ == "__main__":
    print("Load Quantization model package...")
    device = torch.device(args.device)

    """
    Load and check model workspace package is valid
    """
    ws_path=args.model_ws_pkg[:-1] if args.model_ws_pkg[-1]=="/" else args.model_ws_pkg #deal with additional "/" when user give directory path
    ws_path=str(pathlib.Path(ws_path))
    print(f"The workspace work on %s" % ws_path)
    ws_path=ws_path.replace("/",".") #ready for create workspace path

    """
    Load Quantization model workspace package is valid
    """
    #read quantize_config.yaml first
    assert args.model_pkg != "none","please input --model_pkg={$(Your model package path to load quantizaiton model)}"

    model_fq, transformer,quant_config = quant_func.quantize_model_loader(args.model_pkg)
    """
    Dequant insertion
    """
    targ={}
    #targ['layer28']=0.55841

    """
    """
    if ws_path != "none":
        data_ws=importlib.import_module(ws_path+'.dataset')
        tra_dataloader,val_dataloader,cal_dataloader,cos_dataloader = data_ws.return_dataset()
        data_config = data_ws.dataset_cfg()
        eval_ws=importlib.import_module(ws_path+'.evaluation')
        eval_ws.inference_FQ(model_fq,val_dataloader,data_config,device,symm=quant_config['a_symmetry'])
    



