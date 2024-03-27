#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 09:52:07 2021

@author: aitester
"""

from quantize_utils.relation import create_relation
from quantize_utils.quantizer import QuantNConv2d, QuantNLinear, QuantNReLU, QuantMeasure, QuantNReLU6, QuantNLeakyReLUCal, QuantNLeakyReLU, Constant_module
from quantize_utils.quantizer import CalQuantMeasure, QuantNConv2dCal, QuantNLinearCal, QuantNReLUCal, QuantNReLU6Cal, set_layer_bits_cal, QuantNPReLUCal, QuantNPReLU
from quantize_utils.sft_quantizer import SftQuantNConv2d, SftQuantNLinear, SftQuantNReLU, SftQuantMeasure, SftQuantNReLU6
from quantize_utils.sft_quantizer import CalSftQuantMeasure, SftQuantNConv2dCal, SftQuantNLinearCal, SftQuantNReLUCal, SftQuantNReLU6Cal, set_layer_bits_sft
from quantize_utils.calibrations import record_extreme, record_hist, normal_eval, do_calibration, set_symmetry, set_quant_minmax, fixed_quantizer, passthrough_quantizer
from quantize_utils.prepare import switch_layers, replace_op, restore_op, merge_batchnorm
from quantize_utils.prepare import activation_process, QuantMeasure_Manager
from quantize_utils.prepare import quantize_targ_layer, quantize_targ_layer_PC 
from compensation import weight_forging, qbias_compensation, qweight_compensation
from quantizer_mapping import FIXED_DICT
from andesDAG.DAG_builder import DAGraph
from quantize_utils.deploy_utils import register_prev_requant, list_model, setting_modules, record_ACC, CustomTracer, dummy_input_creator
from quantize_utils.deploy_utils import estimate_preshift, RecorderACC, module_dict_generator, Pytorch2FQ, insert_constant, insert_Quantstub


import torch
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.functional as F

import numpy as np

from common import andesPickle

def comb(calibration, forging, Bcompensation, Wcompensation, signed, relu):
    st = 'The Metrics of '
    if relu:
        st = st + 'ReLU6 '
    else:
        st = st + 'ReLU '
    if signed:
        st = st + 'Signed_'
    if forging:
        st = st + 'forging '
    if Bcompensation:
        st  = st + '+ Bcompensation '
    if Wcompensation:
        st  = st + '+ Wcompensation '
    if calibration:
        st  = st + '+ Calibration: '
    else:
        st  = st + '+ DFQ: '
    return st

def cos_sim(dimension, tensor1, tensor2):
    cos = torch.nn.CosineSimilarity(dim = dimension,eps = 1e-6)
    return cos(tensor1, tensor2)

def SQNR(x, y):
    Ps = torch.norm(x)
    Pn = torch.norm(x-y)
    return 20*torch.log10(Ps/Pn)

def kl_div(tensor1, tensor2):
    #tensor1 is target from fp32
    #tensor2 is input from quantization
    src=F.log_softmax(tensor2,dim=-1)
    target=F.softmax(tensor1,dim=-1)
    result = F.kl_div(src, target, None, None, 'sum')
    if result<0:
        result=torch.tensor(0.0)
    return result

def compensation_FP32(model, cos_dataloaders, forward_all, data, bits_weight,
                      per_channel, w_symmetry, a_symmetry, forge, Bcompensation, signed_e, Wcompensation, isReLU6, device):
    
    test_model = model
    test_model=insert_constant(test_model,a_symmetry)
    test_model.to('cpu')
    data = data
    DAG_object = DAGraph()
    module_dict = {}
    if isReLU6:
        module_dict[0] = [(torch.nn.ReLU, torch.nn.ReLU6)]
    else:
        module_dict[0] = [(torch.nn.ReLU6, torch.nn.ReLU6)] # with Relu, we can do the equalization.
    ig_layer = []
    targ_layer = [torch.nn.Conv2d,torch.nn.Linear]
    
    test_model, DAG_object = switch_layers(test_model, DAG_object, data, module_dict, ignore_layer=ig_layer, quant_op=False, sym = w_symmetry)        
    vertices = DAG_object.proxy.getVertices()
    edges = DAG_object.proxy.getEdges()
    
    test_model = merge_batchnorm(test_model, vertices, edges, targ_layer)

    if forge:
        res = create_relation(vertices, edges, targ_layer, delete_single=False)
        if signed_e:            
            weight_forging(vertices, res, targ_layer, visualize_state=False, converge_thres=2e-3, converge_count = 20, signed=w_symmetry, s_range=[1e-8, 1e8]) #original program : 2e-7
        else:
            weight_forging(vertices, res, targ_layer, visualize_state=False, converge_thres=2e-3, converge_count = 20, s_range=[1e-8, 1e8]) #original program : 2e-7
        if Bcompensation:
            qbias_compensation(vertices, res, edges, 3)
    if Wcompensation:
        if per_channel:
            qweight_compensation(vertices, edges, targ_layer, bits_weight, signed=w_symmetry, per_channel = True, logger = False)
        else:
            qweight_compensation(vertices, edges, targ_layer, bits_weight, signed=w_symmetry, logger = False)
    
    test_model = test_model.to(device)
    #print(test_model)
    outfmap = []
    def H_hook_fn_forward(module, ifmp, ofmp):
        outfmap.append(ofmp)
    with torch.no_grad():
        for name, module in test_model.named_modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.ReLU) or isinstance(module, torch.nn.ReLU6):
                #print(name)
                module.register_forward_hook(H_hook_fn_forward)
    forward_all(test_model, cos_dataloaders, device)
    
    test_model = None
    del test_model
    
    return outfmap

def compensation_quantized(model, cal_dataloaders, cos_dataloaders, data_config, forward_all, forward_all_cal, 
                           data, 
                           bits_weight, bits_activation, bits_bias, quantile,
                           per_channel, w_symmetry, a_symmetry, sft_based,
                           forge, Bcompensation, signed_e, Wcompensation, isReLU6, calibration, device):
    data_1 = data
    test_model = model
    test_model=insert_constant(test_model,a_symmetry)
    test_model.to('cpu')
    
    # Model numerical compensation here.
    # default fusion, then combination of the equalization, Bcompensation, and Wcompensation 
    DAG_object = DAGraph()
    module_dict, activation_targ, targ_layer, ig_layer, cal_quant = Pytorch2FQ(calibration, isReLU6, sft_based)
    test_model, DAG_object = switch_layers(test_model, DAG_object, data_1, module_dict, ignore_layer=ig_layer, quant_op=True, sym = w_symmetry)        
    vertices = DAG_object.proxy.getVertices()
    edges = DAG_object.proxy.getEdges()
    print(test_model)
    if sft_based and a_symmetry and w_symmetry:
        set_layer_bits_sft(vertices, bits_weight, bits_activation, bits_bias, targ_layer, cal = cal_quant)
    else:
        set_layer_bits_cal(vertices, bits_weight, bits_activation, bits_bias, targ_layer, cal = cal_quant)
    test_model = merge_batchnorm(test_model, vertices, edges, targ_layer)

    if forge:
        res = create_relation(vertices, edges, targ_layer, delete_single=False)
        if signed_e:            
            weight_forging(vertices, res, targ_layer, visualize_state=False, converge_thres=2e-3, converge_count = 20, signed=signed_e, s_range=[1e-8, 1e8]) #original program : 2e-7
        else:
            weight_forging(vertices, res, targ_layer, visualize_state=False, converge_thres=2e-3, converge_count = 20, s_range=[1e-8, 1e8]) #original program : 2e-7
        if Bcompensation:
            qbias_compensation(vertices, res, edges, 3) #some case that the quantile 3 may fails.
    if Wcompensation:
        if per_channel:
            qweight_compensation(vertices, edges, targ_layer, bits_weight, signed=w_symmetry, per_channel = True, logger = False)
        else:
            if sft_based and a_symmetry and w_symmetry:
                qweight_compensation(vertices, edges, targ_layer, bits_weight, signed=w_symmetry, sft_based = sft_based, logger = False, log_dir = None)
            else:
                qweight_compensation(vertices, edges, targ_layer, bits_weight, signed=w_symmetry, logger = False, log_dir = None)

    activation_process(test_model, DAG_object, activation_targ, bits_activation)    
    DAG_object._build_graph(test_model, data_1, [torch.nn.Identity,ig_layer[0]])
    vertices = DAG_object.proxy.getVertices()
    edges = DAG_object.proxy.getEdges()
    if sft_based and a_symmetry and w_symmetry:
        QuantMeasure_Manager(vertices, edges, quant_bits = bits_activation, calibration = calibration, sft_based = sft_based)
    else:
        QuantMeasure_Manager(vertices, edges, quant_bits = bits_activation, calibration = calibration)
        if a_symmetry:
             set_symmetry(test_model, ig_layer)
    if calibration:
        targ_act_layer = [QuantNPReLUCal]
    else:
        targ_act_layer = [QuantNPReLU]

    if calibration:
        if per_channel:
            vertices = quantize_targ_layer_PC(vertices, bits_weight, bits_bias, targ_layer, sym=w_symmetry)
            vertices = quantize_targ_layer(vertices, bits_weight, bits_bias, targ_act_layer, sym=a_symmetry)
        else:
            if sft_based and a_symmetry and w_symmetry:
                quantize_targ_layer(vertices, bits_weight, bits_bias, targ_layer, sym=w_symmetry, sft_base = sft_based)
            else:
                vertices = quantize_targ_layer(vertices, bits_weight, bits_bias, targ_layer, sym=w_symmetry)
                vertices = quantize_targ_layer(vertices, bits_weight, bits_bias, targ_act_layer, sym=a_symmetry)
        test_model = test_model.to(device)
        test_model.eval()
        record_extreme(test_model, ig_layer)#[CalQuantMeasure])
        print("Collecting Extreme Value")
        forward_all_cal(test_model, cal_dataloaders, data_config, device, a_symmetry, bits_activation,calibration=True)
        record_hist(test_model, ig_layer)#[CalQuantMeasure])
        print("Collecting Histogram")
        forward_all_cal(test_model, cal_dataloaders, data_config, device, a_symmetry, bits_activation,calibration=True)
        print("Calibration")
        do_calibration(test_model, ig_layer)#[CalQuantMeasure])
        fixed_quantizer(test_model, FIXED_DICT)
        normal_eval(test_model, ig_layer)#[CalQuantMeasure])
    
    else:
        if per_channel:
            vertices = quantize_targ_layer_PC(vertices, bits_weight, bits_bias, targ_layer, sym=w_symmetry)
        else:
            if sft_based and a_symmetry and w_symmetry:
                quantize_targ_layer(vertices, bits_weight, bits_bias, targ_layer, sym=w_symmetry, sft_base = sft_based)
            else:
                vertices = quantize_targ_layer(vertices, bits_weight, bits_bias, targ_layer, sym=w_symmetry)
        try:
            set_quant_minmax(vertices, edges, N=quantile)
        except:
            print("DFQ not support for no batchnormal")
            return 0.0

    for name,module in test_model.named_modules():
        if hasattr(module,"update_stat"):
            module.update_stat=False
            module.training=False

    if a_symmetry:
        DAG_object = DAGraph()
        DAG_object._build_graph(test_model.to("cpu"), data_1, [torch.nn.Identity,Constant_module,ig_layer[0]])
        vertices = DAG_object.proxy.getVertices()
        edges = DAG_object.proxy.getEdges()
        targ_type=[QuantNLeakyReLUCal,QuantNLeakyReLU]
        passthrough_quantizer(test_model, targ_type, vertices, edges)
    #torch.cuda.empty_cache()
    test_model = test_model.to(device)
    replace_op()
    restore_op()

    if not(a_symmetry):
        for name, module in test_model.named_modules():
            if isinstance(module,torch.nn.ReLU) or isinstance(module,torch.nn.ReLU6):
                #print('>0')
                if hasattr(module,'quant'):
                    if hasattr(module.quant,'running_min'):
                        qmin = 0.
                        qmax = 2. ** bits_activation - 1.
                        scale = (module.quant.running_max - module.quant.running_min) / (qmax - qmin)            
                        scale = max(scale, 1e-20) # original : 1e-8
                        zero_point = module.quant.running_min.div(scale).round()
                        #print(zero_point)
                        if zero_point.item() > 0.0:
                            module.quant.running_min = torch.zeros(1).to(device).data[0]
    
    outfmap = []
    namelist=[]
    def H_hook_fn_forward(module, ifmp, ofmp):
        outfmap.append([module.name,ofmp])
        namelist.append(module.name)
    with torch.no_grad():
        for name, module in test_model.named_modules():
            if hasattr(module,'quant'):
                module.quant.name=name
                module.quant.register_forward_hook(H_hook_fn_forward)
    forward_all(test_model, cos_dataloaders, data_config, device, a_symmetry, bits_activation)
    
    test_model = None
    del test_model
    
    return outfmap
#s

def compare_combination(model_path, cal_dataloaders, cos_dataloaders, data, data_config, forward_all_f, forward_all, forward_all_cal,
    bits_weight, bits_activation, bits_bias, quantile, per_channel, w_symmetry, a_symmetry, sft_based,
    forge, Bcompensation, signed_e, Wcompensation, isReLU6, calibration, device):
    print('__________________________________________________')
    model_test = andesPickle.load(model_path)

    FP32_ofmp_list = compensation_FP32(model = model_test, 
                                        cos_dataloaders = cos_dataloaders, 
                                        forward_all = forward_all_f, 
                                        data = data,
                                        bits_weight = bits_weight,
                                        per_channel = per_channel,
                                        w_symmetry = w_symmetry,
                                        a_symmetry = a_symmetry,
                                        forge = forge, 
                                        Bcompensation = Bcompensation, 
                                        signed_e = signed_e, 
                                        Wcompensation = Wcompensation, 
                                        isReLU6 = isReLU6, 
                                        device = device)
    model_test.to('cpu')
    model_test = None
    del model_test

    model_test = andesPickle.load(model_path)
    Q_ofmp_list = compensation_quantized(model = model_test,   
                                   cal_dataloaders = cal_dataloaders, 
                                   cos_dataloaders = cos_dataloaders,
                                   data_config = data_config, 
                                   forward_all = forward_all,
                                   forward_all_cal = forward_all_cal, 
                                   data = data,
                                   bits_weight = bits_weight, 
                                   bits_activation = bits_activation, 
                                   bits_bias = bits_bias, 
                                   quantile = quantile,
                                   per_channel = per_channel,
                                   w_symmetry = w_symmetry, 
                                   a_symmetry = a_symmetry,
                                   sft_based = sft_based,
                                   forge = forge, 
                                   Bcompensation = Bcompensation, 
                                   signed_e = signed_e, 
                                   Wcompensation = Wcompensation, 
                                   isReLU6 = isReLU6, 
                                   calibration = calibration,
                                   device = device)
    model_test.to('cpu')
    model_test = None
    del model_test

    return FP32_ofmp_list, Q_ofmp_list

def cosine_main_func(model_path, forward_one, forward_one_Q, forward_all_Q, calibrate_dataloader, cos_dataloader, data, data_config, bits_weight, bits_activation, bits_bias, quantile, per_channel, w_symmetry, a_symmetry, sft_based, log_dir, device):
    
    """
    """
    model_test = andesPickle.load(model_path)
    dummy_input_list=[]
    if isinstance(data,tuple):
        for element in data:
            dummy_input_list.append(list(element.size()))
    else:
        dummy_input_list.append(list(data.size()))
    test_model=insert_Quantstub(model_test,a_symmetry)
    forward_one(test_model, cos_dataloader,device)
    fp32_range_list=[]
    input_shape_list=[]
    for name,module in test_model.named_modules():
        if hasattr(module,'finder_min') and module.finder_min<module.finder_max:
            fp32_range_list.append([float(module.finder_min.cpu().detach()),float(module.finder_max.cpu().detach())])
            input_shape_list.append([1]+list(module.shape)[1:])
    print(input_shape_list)
    """
    """
    log_dir = log_dir + '/metrics_results.txt'
    config = 'The Q_Config is '
    if per_channel:
        config = config + 'PerChannel_'
    else:
        config = config + 'PerTensor_'
        
    if w_symmetry:
        config = config + 'Symm_wQ: ' + str(bits_weight) + '_bits, '
    else:
        config = config + 'Asymm_wQ: ' + str(bits_weight) + '_bits, '
    
    if a_symmetry:
        config = config + 'PerTensor_Symm_aQ: ' + str(bits_activation) + '_bits, ' 
    else:
        config = config + 'PerTensor_Asymm_aQ: ' + str(bits_activation) + '_bits, ' 
    
    config = config + 'bias: ' + str(bits_bias) + '_bits.'
    
    
    print("___________________________________________________________________")
    f = open(log_dir,'wb')
    f.close()
    with open(log_dir,'a') as f:
        f.write(config)
        f.write('\r\n')
        f.write('\r\n')
    f.close()
    print(config)
    print("___________________________________________________________________")
    
    calibrations = [False,False,True,True,False,False,False,False,False,False,False,False,False,False,True,True,True,True,True,True,True,True,True,True]
    forging = [False,False,False,False,False,False,True,True,True,True,True,True,True,True,False,False,True,True,True,True,True,True,True,True]
    Bcompensations = [False,False,False,False,False,False,False,False,True,True,False,False,True,True,False,False,False,False,True,True,False,False,True,True]
    Wcompensations = [False,True,False,True,False,True,False,True,False,True,False,True,False,True,False,True,False,True,False,True,False,True,False,True,]
    signeds = [False,False,False,False,False,False,False,False,False,False,True,True,True,True,False,False,False,False,False,False,True,True,True,True]
    relus = [True,True,True,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False]

    for i in range(len(calibrations)):
        combine = comb(calibrations[i], forging[i], Bcompensations[i], Wcompensations[i], signeds[i], relus[i])
        has_quant = True
        FP32_ofmp_list = []
        FP32_ofmp_list, Q_ofmp_list = compare_combination(
                                            model_path,
                                            cal_dataloaders = calibrate_dataloader,
                                            cos_dataloaders = cos_dataloader,
                                            data = data, 
                                            data_config = data_config,
                                            forward_all_f = forward_one,
                                            forward_all = forward_one_Q,
                                            forward_all_cal = forward_all_Q,
                                            bits_weight = bits_weight,
                                            bits_activation = bits_activation,
                                            bits_bias = bits_bias,
                                            quantile = quantile,
                                            per_channel = per_channel,
                                            w_symmetry = w_symmetry,
                                            a_symmetry = a_symmetry,
                                            sft_based = sft_based ,
                                            forge = forging[i],
                                            Bcompensation = Bcompensations[i],
                                            signed_e = signeds[i],
                                            Wcompensation = Wcompensations[i],
                                            isReLU6 = relus[i],
                                            calibration = calibrations[i],
                                            device = device)
        if Q_ofmp_list==0.0:
            FP32_ofmp_list=[]
        with open(log_dir,'a') as f:
            f.write('\r\n')
            f.write(combine)
            f.write('\r\n')
            with torch.no_grad():
                for i in range(len(FP32_ofmp_list)):
                    if has_quant:
                        FP32_ofmp_list[i] = FP32_ofmp_list[i].view(-1)
                        Q_ofmp_list[i][1] = Q_ofmp_list[i][1].view(-1)
                        diff = FP32_ofmp_list[i].to('cpu') - Q_ofmp_list[i][1].to('cpu')
                        MAE = diff.abs().mean()
                        cosine = cos_sim(dimension = 0, tensor1 = FP32_ofmp_list[i].to('cpu'), tensor2 = Q_ofmp_list[i][1].to('cpu'))
                        kldiv=kl_div(tensor1 = FP32_ofmp_list[i].to('cpu'), tensor2 = Q_ofmp_list[i][1].to('cpu'))
                        sqnr=SQNR(FP32_ofmp_list[i].to('cpu'), Q_ofmp_list[i][1].to('cpu'))
                        tmp=np.append(MAE.to('cpu').numpy(),cosine.to('cpu').numpy())
                        metrics = np.array2string(np.append(tmp,sqnr.to('cpu').numpy()), precision=5, separator=',', suppress_small=True)
                        f.write(metrics+Q_ofmp_list[i][0])
                        f.write('\r\n')
        f.close()
        
    print()
    #return results
