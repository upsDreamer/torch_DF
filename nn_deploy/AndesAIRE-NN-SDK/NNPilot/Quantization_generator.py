#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 09 13:00:10 2022

@author: evansongs
"""

"""
Add metric calculation
"""
import os
import torch
import yaml
import numpy as np
import importlib
import pathlib
import copy
from torch.utils.data import DataLoader, Subset
import argparse
import traceback
import sys
from common import andesPickle, fx_utils
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
from torch.quantization.fake_quantize import disable_observer,disable_fake_quant,enable_observer,enable_fake_quant

from andesQuant import quant_func,qsensitivity_analysis
from andesQuant.qat_utils.utils import FakeQuant_2_QAT_loader,QAT_2_FQ_loader

parser = argparse.ArgumentParser(description="FQ Task on General Interface.")
parser.add_argument("--model_ws_pkg", type=str, default="none")
parser.add_argument("--model_pkg", type=str, default="none")
parser.add_argument("--save_path", type=str, default="./default_quant_model_pkg")
parser.add_argument("--enable_metrics", action='store_true')
parser.add_argument("--w_symmetry", action='store_true')
parser.add_argument("--a_symmetry", action='store_true')
parser.add_argument("--sft_based", action='store_true')
parser.add_argument("--per_channel", action='store_true')
parser.add_argument("--qat", action='store_true')
parser.add_argument("--qat_pth", type=str, default="none")
parser.add_argument("--bits_weight", type=int, default=8)
parser.add_argument("--bits_activation", type=int, default=8)
parser.add_argument("--bits_bias", type=int, default=32)
parser.add_argument("--quantile",type=float, default=6.0)
parser.add_argument("--epoch",type=int, default=1)
parser.add_argument("--observer_epoch",type=int, default=20)
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--env_check", action='store_true')
parser.add_argument("--specific_combine",type=int, default=-1)
args = parser.parse_args()



    
if __name__ == "__main__":
    print("Start generate Quantization model package...")
    device = torch.device(args.device)

    """
    Load and check model workspace package is valid
    """
    assert args.model_ws_pkg != "none", "Please load exist model workspace package or define yourself accroding to example in example/model_ws_tmp/"
    ws_path=args.model_ws_pkg[:-1] if args.model_ws_pkg[-1]=="/" else args.model_ws_pkg #deal with additional "/" when user give directory path
    ws_path=str(pathlib.Path(ws_path))
    print(f"The workspace work on %s" % ws_path)
    ws_path=ws_path.replace("/",".") #ready for create workspace path

    """
    load fp32 model from workspace or exist pickle
    """
    module_ws=importlib.import_module(ws_path+'.model_fp32')
    if args.model_pkg == "none":
        #load from defined {$(MODEL_WS_PAKAGE)}
        model_fp32 = module_ws.return_fp32_model()
    else:
        #load from exist model package
        model_fp32 = andesPickle.load(args.model_pkg+"/model_fp32.pickle")
    print("Load FP32 model Success")
    if args.env_check:
        model_fq, DAG_object,quant_config = quant_func.quantize_model_loader("example/resnet8_onnx/rn8_symm/")
        print("env prepare success")
        exit()


    """
    load dataloader or dataset and dataset_configration from {$(MODEL_WS_PAKAGE)}/dataset.py
    """
    data_ws=importlib.import_module(ws_path+'.dataset')
    tra_dataloader,val_dataloader,cal_dataloader,cos_dataloader = data_ws.return_dataset() #return dataset pairs for (tra_dataloader,val_dataloader,cal_dataloader,cos_dataloader)
    data_config = data_ws.dataset_cfg() #Read channel,height,weight,fp32_max,fp32_min from {$(MODEL_WS_PAKAGE)}/model_cfg.yaml
    if data_config['width']==1:
        test_data = torch.zeros((1, data_config['channel'], data_config['height']))
    else:
        test_data = torch.zeros((1, data_config['channel'], data_config['height'], data_config['width']))
    if "dummy_input" in data_config:
        test_data = []
        print("create dummy input")
        for data_shape in data_config['dummy_input']:
            data=torch.zeros(data_shape)
            test_data.append(data)
        test_data = tuple(test_data)

    """
    load evaluation function from {$(MODEL_WS_PAKAGE)}/evaluation.py
    """
    eval_ws=importlib.import_module(ws_path+'.evaluation')
    print()
    #eval_ws.inference_FP32(model_fp32,val_dataloader,device)
    #eval_ws.inference_FQ(model_fp32,val_dataloader,data_config,device)

    """
    Prepare saving directory
    """
    if os.path.exists(args.save_path):
        print("no need create, overwrite directory")
    else:
        print("create new directory")
        os.mkdir(args.save_path)

    """
    Convert to Quantization use model
    """
    model_fp32=fx_utils.andes_preprocessing(model_fp32)
    model_fp32=fx_utils.add_idd(model_fp32)
    """
    Saving FP32 model in save_path
    """
    andesPickle.save(model_fp32,args.save_path+"/model_fp32.pickle")

    model_fp32 = andesPickle.load(args.save_path+"/model_fp32.pickle")
    #eval_ws.inference_FP32(model_fp32,val_dataloader,device)
    #eval_ws.inference_FQ(model_fp32,val_dataloader,data_config,device)
    print("Ready to search the Best Fake_Quant Model...")
    generated_from_pkg=True
    print("ssssssssssssssssssssssssssssssssssssssssssssssss")

    quant_config = None
    if not args.qat:
        if os.path.exists(args.model_pkg+"/quantize_config.yaml"):
            with open(args.model_pkg+'/quantize_config.yaml', 'r') as f:
                quant_config = yaml.load(f, Loader=yaml.FullLoader)
    #________________________________________________________________________________________________________________
        if args.enable_metrics:
            results = qsensitivity_analysis.cosine_main_func(
                model_path=args.save_path+"/model_fp32.pickle",
                forward_one = eval_ws.forward_one,
                forward_one_Q = eval_ws.forward_one_Q,
                forward_all_Q = eval_ws.inference_FQ,
                calibrate_dataloader = cal_dataloader,
                cos_dataloader = cos_dataloader,
                data = test_data,
                data_config = data_config,
                bits_weight = args.bits_weight,
                bits_activation = args.bits_activation,
                bits_bias = args.bits_bias,
                quantile = args.quantile,
                per_channel = args.per_channel, 
                w_symmetry = args.w_symmetry, 
                a_symmetry = args.a_symmetry, 
                sft_based = args.sft_based,
                log_dir = args.save_path, 
                device = device 
                )

        if not quant_config:
            print("no quantize_config.yaml, start generate ")
        try:
            model_fq = quant_func.quant_main_func(
                                        model_path = args.save_path+"/model_fp32.pickle",  
                                        quant_config = quant_config,
                                        forward_all = eval_ws.inference_FQ, #forward_all_Q,
                                        inference_all_f = eval_ws.inference_FP32,#inference_all_f,
                                        inference_all = eval_ws.inference_FQ,#inference_all_Q, 
                                        cal_dataloaders = cal_dataloader, 
                                        dataloaders = val_dataloader,
                                        data = test_data, 
                                        data_config = data_config,
                                        bits_weight = args.bits_weight,
                                        bits_activation = args.bits_activation,
                                        bits_bias = args.bits_bias,
                                        quantile = args.quantile,
                                        per_channel = args.per_channel, 
                                        w_symmetry = args.w_symmetry, 
                                        a_symmetry = args.a_symmetry, 
                                        sft_based = args.sft_based,
                                        qat = args.qat,
                                        log_dir = args.save_path,
                                        device = device,
                                        specific_combine=args.specific_combine
                                        )
        except Exception as e:
            cl, exc, tb = sys.exc_info()
            lastCallStack = traceback.extract_tb(tb)[-1]
            print(e)
            print(lastCallStack)
            print("Quantization fail, Please check error message")
            exit()

        generated_from_pkg=False
    '''else:
        model = quant_func.quant_main_func(args = args, 
                                           model = model_fp32, 
                                           forward_all = forward_all_Q, 
                                           inference_all_f = inference_all_f,
                                           inference_all = inference_all_Q, 
                                           cal_dataloaders = calibrate_dataloader, 
                                           dataloaders = val_dataloaders, 
                                           channel = 3, 
                                           width = 224, 
                                           height = 224, 
                                           log_dir = 'loggers',
                                           device = device)'''
    if args.qat:
        if generated_from_pkg:
            target_path=args.model_pkg
            #model_fq,transformer,quant_config=quant_func.quantize_model_loader(args.model_pkg)#from model_pkg exist quantize_cfg.yaml
        else:
            target_path=args.save_path
            #model_fq,transformer,quant_config=quant_func.quantize_model_loader(args.save_path)#from this script
        print("do qat")
        #disable qat flag-----------------
        with open(target_path+'/quantize_config.yaml', 'r') as f:
            quant_config = yaml.load(f, Loader=yaml.FullLoader)
        quant_config['qat']=False
        with open(target_path +'/quantize_config.yaml', 'w') as f:
            yaml.dump(quant_config, f, default_flow_style=False, sort_keys=False, width=100)
        #----------------------------------
        model_fq,transformer,quant_config=quant_func.quantize_model_loader(target_path)
        if args.qat_pth != "none":
            print("qat from check point")
        else:
            print("qat from source")
            model_qat,metadict = FakeQuant_2_QAT_loader(model_fq, quant_config)
        """
        Training Start
        """
        train_ws=importlib.import_module(ws_path+'.train')
        #model_qat=module_ws.return_fp32_model()
        criterion,optimizer,scheduler = train_ws.training_set(model_qat,device)
        best_acc=0.0
        best_model_wts = copy.deepcopy(model_qat.state_dict())
        for epoch in range(args.epoch):
            model_qat.train()
            for name,module in model_qat.named_modules():
                if epoch < args.observer_epoch: #Close observer after args.observer_epoch time
                    enable_observer(module)
            model_qat=train_ws.train_one_epoch(model_qat,tra_dataloader,data_config,criterion,optimizer,scheduler,device,quant_config['a_symmetry'],args.bits_activation)
            for name,module in model_qat.named_modules():
                disable_observer(module)
            model_qat.eval()
            acc=eval_ws.inference_FQ(model_qat,val_dataloader,data_config,device,quant_config['a_symmetry'],args.bits_activation)
            if acc > best_acc:
                best_acc=acc
                torch.save(model_qat.state_dict(),target_path+"/quant_model_qat.pth")
                best_model_wts = copy.deepcopy(model_qat.state_dict())

        """
        Convert back to FQ
        """
        model_qat.load_state_dict(best_model_wts)
        model_fq,transformer=QAT_2_FQ_loader(model_qat,metadict,quant_config,target_path)
        torch.save(model_fq.state_dict(),target_path+'/quant_model_qat_fq.pth')
        with open(target_path+'/quantize_config.yaml', 'r') as f:
            quant_config = yaml.load(f, Loader=yaml.FullLoader)
        quant_config['qat']=True
        print("convert to FQ test")
        fq_acc=eval_ws.inference_FQ(model_fq,val_dataloader,data_config,device,quant_config['a_symmetry'],args.bits_activation)
        print(model_fq)
        quant_config['acc_quant']=float(fq_acc)
        with open(target_path +'/quantize_config.yaml', 'w') as f:
            yaml.dump(quant_config, f, default_flow_style=False, sort_keys=False, width=100)




    print("Out_Files...")
    print()
    #torch.save(model.state_dict(), 'loggers'+'/quant_model.pth')
    
    #if not args.per_tensor:
        #if args.shift_based:
            #Tofile_utils_SftBased.record_sfile(model, 'loggers/PTCV_MBNet_v1_para.S')
            #Tofile_utils_SftBased.record_w_header2bin(model, args.bits_weight, args.bits_bias, symmetry=args.w_symmetry, pre_pross = 1/127, logger = 'loggers/bins/')
            #Tofile_utils_SftBased.record_header2bin(model,'loggers/PTCV_MBNet_v1.h', 224, args.bits_weight, args.bits_activation, args.w_symmetry, 16, pre_pross = 1/127)
            #Tofile_utils_SftBased.record_cfile(model, 'loggers/PTCV_MBNet_v1.c', 'loggers/PTCV_MBNet_v1.h')
        #elif args.a_symmetry:
        #else:
    #Tofile_utils_Symm.record_sfile(model, 'loggers/PTCV_MBNet_v1_para.S')
    #Tofile_utils_Symm.record_w_header2bin(model, args.bits_weight, args.bits_bias, symmetry=args.w_symmetry, pre_pross = 0.02078740157480315, logger = 'loggers/bins/')
    #Tofile_utils_Symm.record_header2bin(model,'loggers/PTCV_MBNet_v1.h', 224, args.bits_weight, args.bits_activation, args.w_symmetry, 16, pre_pross = 0.02078740157480315)
    #Tofile_utils_Symm.record_cfile(model, 'loggers/PTCV_MBNet_v1.c', 'loggers/PTCV_MBNet_v1.h')
    #else:
        #if asymmetric:
        #else:
    
    # Outfile 
    #________________________________________________________________________________________________________________
    # outfile parameters bins
    # outfile .h
    # outfile .c
    # outfile .S
    

