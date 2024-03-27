import sys 
import importlib
import common
sys.path.append('andesQuant')
sys.path.append('andesQuant/quantize_utils')

from quantize_utils.relation import create_relation
from quantize_utils.quantizer import QuantNConv2d, QuantNLinear, QuantNReLU, QuantMeasure, QuantNReLU6, QuantNLeakyReLU
from quantize_utils.quantizer import CalQuantMeasure, QuantNConv2dCal, QuantNLinearCal, QuantNReLUCal, QuantNReLU6Cal, set_layer_bits_cal, Constant_module,Chunk_idd,QuantNSigmoidCal,QuantNTanhCal,QuantNReLUCal, QuantNLeakyReLUCal
from quantize_utils.sft_quantizer import SftQuantNConv2d, SftQuantNLinear, SftQuantNReLU, SftQuantMeasure, SftQuantNReLU6
from quantize_utils.sft_quantizer import CalSftQuantMeasure, SftQuantNConv2dCal, SftQuantNLinearCal, SftQuantNReLUCal, SftQuantNReLU6Cal, set_layer_bits_sft
from quantize_utils.calibrations import record_extreme, record_hist, normal_eval, do_calibration, set_symmetry, set_quant_minmax, fixed_quantizer, passthrough_quantizer
from quantize_utils.prepare import switch_layers, replace_op, restore_op, merge_batchnorm
from quantize_utils.prepare import activation_process, QuantMeasure_Manager
from quantize_utils.prepare import quantize_targ_layer, quantize_targ_layer_PC 
from compensation import weight_forging, qbias_compensation, qweight_compensation
from compensation import write_layer_relation, write_layer_bottom
from andesDAG.DAG_builder import DAGraph
from fx_utils import add_idd
from quantizer_mapping import FIXED_DICT

from quantize_utils.deploy_utils import register_prev_requant, list_model, setting_modules, record_ACC, CustomTracer, dummy_input_creator
from quantize_utils.deploy_utils import estimate_preshift, RecorderACC, module_dict_generator, Pytorch2FQ, insert_constant, insert_Quantstub
from qat_utils.utils import FakeQuant_2_QAT_loader,QAT_2_FQ_loader
from copy import deepcopy
import yaml

import torch
import torch.fx as fx
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
import torch.onnx
sys.path.append('../')
#from andesConvert.convert_pytorch_model import ConvertModel
import common
from common import andesPickle
import re
import numpy as np
import pickle
import os.path
import copy
import time
import sys
import traceback
from common.fx_utils import  AndesSlice
sys.path.append('../')
#from andesPrune.utils.fusion import fuse_bn
#import pdb;pdb.set_trace()

rm_layer=(QuantNSigmoidCal,)+(CalQuantMeasure,)+(QuantNConv2dCal,)+(QuantNLinearCal,)+(Constant_module,)+(QuantNReLUCal,)+(QuantNTanhCal,)+(QuantNReLU6Cal,)
# To ignore fx trace module


def quant_get_file_name(whole_path):
    strpro = "{}".format(str(whole_path))
    strpro = strpro.rsplit("/", 1)
    size = len(strpro)
    if (size > 0):
        filename = strpro[size - 1]
    else:
        filename = "None"
    return filename

def load_preshift(model,preshift_path):
    """
    only symmetry activation model should load preshift parameters.
    """
    if preshift_path != "None" and preshift_path != "None\n" and preshift_path != None:
        file_name_preshift = "/"+quant_get_file_name(preshift_path)
        with open(work_path + file_name_preshift, 'r') as preshift_file:
            preshift = yaml.load(preshift_file.read(), Loader=yaml.FullLoader)
        for name, module in model.named_modules():
            if name in preshift.keys():
                setattr(module, 'est_preshift', preshift[name])
    return model
    
    
def quantize_combination(model, cal_dataloaders, dataloaders, data_config,
                         forward_all, inference_all,
                         data,
                         bits_weight = 8, bits_activation = 8, bits_bias = 8, quantile = 6.0,
                         per_channel = False, w_symmetry = True, a_symmetry = False, sft_based = False,
                         forge = False, Bcompensation = False, signed_e = False, Wcompensation = False, 
                         isReLU6 = False, calibration = False, deployment = False, log_dir = 'loggers', device = torch.device('cpu')):
    # Data, Model process here.
    # Users should define their own Data and Model here 
    #___________________________________________________________________________________________________________________________
    data_1 = data
    test_model = model
    test_model=insert_constant(test_model,a_symmetry)
    test_model.to(device)
    test_model.to('cpu')
    #___________________________________________________________________________________________________________________________
    
    
    # Model numerical compensation here.
    # default fusion, then combination of the equalization, Bcompensation, and Wcompensation 
    #___________________________________________________________________________________________________________________________
    DAG_object = DAGraph()
    import pdb; pdb.set_trace()
    module_dict, activation_targ, targ_layer, ig_layer, cal_quant = Pytorch2FQ(calibration, isReLU6, sft_based)
    ig_layer.append(Constant_module)
    ig_layer.append(Constant_module)
    test_model, DAG_object = switch_layers(test_model, DAG_object, data_1, module_dict, ignore_layer=ig_layer, quant_op=True, sym = w_symmetry)        
    vertices = DAG_object.proxy.getVertices()
    edges = DAG_object.proxy.getEdges()
    outshapes = DAG_object.proxy.getOutShapes()
    if deployment:
        write_layer_bottom(vertices, edges, log_dir + "/bottom_logger.txt")
    if sft_based and a_symmetry and w_symmetry:
        set_layer_bits_sft(vertices, bits_weight, bits_activation, bits_bias, targ_layer, cal = cal_quant)
    else:
        set_layer_bits_cal(vertices, bits_weight, bits_activation, bits_bias, targ_layer, cal = cal_quant)
    test_model = merge_batchnorm(test_model, vertices, edges, targ_layer)
    

    if forge:
        res = create_relation(vertices, edges, targ_layer, delete_single=False)
        if deployment:
            write_layer_relation(vertices, res, log_dir + "/relation_logger.txt")
        if signed_e:            
            weight_forging(vertices, res, targ_layer, visualize_state=False, converge_thres=2e-3, converge_count = 20, signed=signed_e, s_range=[1e-8, 1e8], logger = deployment, log_dir = log_dir) #original program : 2e-7
        else:
            weight_forging(vertices, res, targ_layer, visualize_state=False, converge_thres=2e-3, converge_count = 20, s_range=[1e-8, 1e8], logger = deployment, log_dir = log_dir) #original program : 2e-7
        if Bcompensation:
            qbias_compensation(vertices, res, edges, 3, logger = deployment, log_dir = log_dir) #some case that the quantile 3 may fails.
    if Wcompensation:
        if per_channel:
            qweight_compensation(vertices, edges, targ_layer, signed=w_symmetry, per_channel = True, logger = deployment, log_dir = log_dir)
        else:
            if sft_based and a_symmetry and w_symmetry:
                qweight_compensation(vertices, edges, targ_layer, bits_weight, signed=w_symmetry, sft_based = sft_based, logger = deployment, log_dir = log_dir)
            else:
                qweight_compensation(vertices, edges, targ_layer, bits_weight, signed=w_symmetry, logger = deployment, log_dir = log_dir)
    #___________________________________________________________________________________________________________________________
    
    
    # QuantMeasure process here.
    # the position of the QuantMeasure should be checked 
    #___________________________________________________________________________________________________________________________
    activation_process(test_model, DAG_object, activation_targ, bits_activation)
    DAG_object._build_graph(test_model, data_1, [torch.nn.Identity,Constant_module,ig_layer[0]])
    vertices = DAG_object.proxy.getVertices()
    edges = DAG_object.proxy.getEdges()
    if sft_based and a_symmetry and w_symmetry:
        QuantMeasure_Manager(vertices, edges, quant_bits = bits_activation, calibration = calibration, sft_based = sft_based)
    else:
        QuantMeasure_Manager(vertices, edges, quant_bits = bits_activation, calibration = calibration)
        if a_symmetry:
             set_symmetry(test_model, [ig_layer[0]])
    from quantize_utils import quantizer
    if calibration:
        targ_act_layer = [quantizer.QuantNPReLUCal]
    else:
        targ_act_layer = [quantizer.QuantNPReLU]
    if calibration:
        if per_channel:
            vertices = quantize_targ_layer_PC(vertices, bits_weight, bits_bias, targ_layer, sym=w_symmetry)
            vertices = quantize_targ_layer(vertices, bits_weight, bits_bias, targ_act_layer, sym=a_symmetry)
        else:
            if sft_based and a_symmetry and w_symmetry:
                quantize_targ_layer(vertices, bits_weight, bits_bias, targ_layer, sym=w_symmetry, sft_base = sft_based)
            else:
                #targ_layer = targ_layer + target_act_layer
                vertices = quantize_targ_layer(vertices, bits_weight, bits_bias, targ_layer, sym=w_symmetry)
                vertices = quantize_targ_layer(vertices, bits_weight, bits_bias, targ_act_layer, sym=a_symmetry)
        test_model = test_model.to(device)
        test_model.eval()
        record_extreme(test_model, [ig_layer[0]])#[CalQuantMeasure])
        print("Collecting Extreme Value")
        forward_all(test_model, cal_dataloaders, data_config, device, a_symmetry,bits_activation,calibration=True)
        record_hist(test_model, [ig_layer[0]])#[CalQuantMeasure])
        print("Collecting Histogram")
        forward_all(test_model, cal_dataloaders, data_config, device, a_symmetry,bits_activation,calibration=True)
        print("Calibration")
        do_calibration(test_model, [ig_layer[0]])#[CalQuantMeasure])
        fixed_quantizer(test_model, FIXED_DICT)
        normal_eval(test_model, [ig_layer[0]])#[CalQuantMeasure])
    
    else:
        if per_channel:
            vertices = quantize_targ_layer_PC(vertices, bits_weight, bits_bias, targ_layer, sym=w_symmetry)
            vertices = quantize_targ_layer(vertices, bits_weight, bits_bias, targ_act_layer, sym=a_symmetry)
        else:
            if sft_based and a_symmetry and w_symmetry:
                quantize_targ_layer(vertices, bits_weight, bits_bias, targ_layer, sym=w_symmetry, sft_base = sft_based)
            else:
                vertices = quantize_targ_layer(vertices, bits_weight, bits_bias, targ_layer, sym=w_symmetry)
                vertices = quantize_targ_layer(vertices, bits_weight, bits_bias, targ_act_layer, sym=a_symmetry)
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

    test_model.to(device)
    replace_op()
    restore_op()
    #___________________________________________________________________________________________________________________________
    if not(a_symmetry):
        for name, module in test_model.named_modules():
            if hasattr(module,'running_min'):
                qmin = 0.
                qmax = 2. ** bits_activation - 1.
                scale = (module.running_max - module.running_min) / (qmax - qmin)            
                scale = max(scale, 1e-20) # original : 1e-8
                zero_point = module.running_min.div(scale).round()
                #print(zero_point)
                if zero_point.item() > 0.0:
                    module.running_min = torch.zeros(1).to(device).data[0]

    #acc = inference_all(test_model, dataloaders, data_config, device, a_symmetry, bits_activation)
    if deployment:
        result_model=test_model
        test_model=insert_Quantstub(test_model,a_symmetry)
    acc = inference_all(test_model, dataloaders, data_config, device, a_symmetry, bits_activation)
    if deployment:
        fp32_range_list=[]
        for name,module in test_model.named_modules():
            if hasattr(module,'finder_min') and module.finder_min<=module.finder_max:
                fp32_range_list.append([float(module.finder_min.cpu().detach()),float(module.finder_max.cpu().detach())])
        return result_model,acc,fp32_range_list
    else:
        result = 'The Accuracy of '
        if isReLU6:
            result = result + 'ReLU6,'
        if forge:
            result = result + 'WF,'
        if signed_e:
            result = result + 'S,'
        if Bcompensation:
            result = result + 'QBc,'
        if Wcompensation:
            result = result + 'QWc,'
        if calibration:
            result = result + 'KL_div: '
        else:
            result = result + 'DFQ: '
        
        
        test_model.to('cpu')
        test_model = None
        dataloaders = None
        vertices = None
        del test_model, dataloaders, vertices
        print(result, acc, '%')
        print()
        return acc

def find_preshifts(model, cal_dataloaders, data_config,  
                    forward_all, data,
                    min_i = -2.11790393, max_i = 2.64, fp32_range_list=[], quantile = 6.0,
                    forge = False, Bcompensation = False, signed_e = False, Wcompensation = False, 
                    isReLU6 = False, calibration = False, log_dir = 'loggers', device = torch.device('cpu')):
    print('Finding Pre_shift of symmetric model')
    
    data_1 = data
    test_model = model
    test_model=insert_constant(test_model,True)
    test_model.eval()
    test_model.to('cpu')
    
    DAG_object = DAGraph()
    module_dict, activation_targ, targ_layer, ig_layer, cal_quant = Pytorch2FQ(calibration, isReLU6)
    test_model, DAG_object = switch_layers(test_model, DAG_object, data_1, module_dict, ignore_layer=ig_layer, quant_op=True, sym = True)        
    vertices = DAG_object.proxy.getVertices()
    edges = DAG_object.proxy.getEdges()
    set_layer_bits_cal(vertices, 8, 8, 32, targ_layer, cal = cal_quant)
    test_model = merge_batchnorm(test_model, vertices, edges, targ_layer)
    

    if forge:
        res = create_relation(vertices, edges, targ_layer, delete_single=False)
        if signed_e:            
            weight_forging(vertices, res, targ_layer, visualize_state=False, converge_thres=2e-3, converge_count = 20, signed=signed_e, s_range=[1e-8, 1e8], logger = False, log_dir = log_dir) #original program : 2e-7
        else:
            weight_forging(vertices, res, targ_layer, visualize_state=False, converge_thres=2e-3, converge_count = 20, s_range=[1e-8, 1e8], logger = False, log_dir = log_dir) #original program : 2e-7
        if Bcompensation:
            qbias_compensation(vertices, res, edges, 3, logger = False, log_dir = log_dir) #some case that the quantile 3 may fails.
    if Wcompensation:
        qweight_compensation(vertices, edges, targ_layer, 8, signed=True, logger = False, log_dir = log_dir)
    
    activation_process(test_model, DAG_object, activation_targ, 8)
    DAG_object._build_graph(test_model, data_1, [torch.nn.Identity,Constant_module,ig_layer[0]])
    vertices = DAG_object.proxy.getVertices()
    edges = DAG_object.proxy.getEdges()
    QuantMeasure_Manager(vertices, edges, quant_bits = 8, calibration = calibration)
    set_symmetry(test_model, [ig_layer[0]])
    
    if calibration:
        vertices = quantize_targ_layer(vertices, 8, 32, targ_layer, sym=True)
        test_model = test_model.to(device)
        test_model.eval()
        record_extreme(test_model, [ig_layer[0]])
        print("Collecting Extreme Value")
        forward_all(test_model, cal_dataloaders, data_config, device, calibration=True)
        record_hist(test_model, [ig_layer[0]])
        print("Collecting Histogram")
        forward_all(test_model, cal_dataloaders, data_config, device, calibration=True)
        print("Calibration")
        do_calibration(test_model, [ig_layer[0]])
        fixed_quantizer(test_model, FIXED_DICT)
        normal_eval(test_model, [ig_layer[0]])
    
    else:
        vertices = quantize_targ_layer(vertices, 8, 32, targ_layer, sym=True)
        set_quant_minmax(vertices, edges, N=quantile)
    test_model.to(device)
    replace_op()
    restore_op()

    DAG_object = None
    vertices = None
    edges = None
    module_dict = None
    ig_layer = None
    del DAG_object, vertices, edges, module_dict, ig_layer
    
    test_model = test_model.to('cpu')
    DAG_object1 = DAGraph()
    module_dict, ig_layer = module_dict_generator(calibration, isReLU6)
    DAG_object1 = DAGraph()
    DAG_object1._build_graph(test_model, data_1, [torch.nn.Identity,Constant_module,ig_layer[0]])
    vertices = DAG_object1.proxy.getVertices()
    edges = DAG_object1.proxy.getEdges()
    register_prev_requant(vertices, edges, pdata_min = torch.tensor(min_i), pdata_max = torch.tensor(max_i), fp32_range_list=fp32_range_list)
    model_list = list_model(test_model)
    
    DAG_object = None
    del DAG_object

    DAG_object2 = DAGraph()
    test_model, DAG_object2 = switch_layers(test_model, DAG_object2, data_1, module_dict, ignore_layer=ig_layer, quant_op=True, sym = True)
    DAG_object2._build_graph(test_model, data_1, [torch.nn.Identity,Constant_module,ig_layer[0]])
    vertices = DAG_object2.proxy.getVertices()
    for layer_idx in vertices:
        if hasattr(vertices[layer_idx],'quant'):
            delattr(vertices[layer_idx],'quant')
    setting_modules(test_model, model_list)
    
    test_model = test_model.to(device)
    test_model.eval()
    print('Recording_InterMediate value')
    record_ACC(test_model, [RecorderACC])
    forward_all(test_model, cal_dataloaders, data_config, device,calibration=True)
    print('Compute_pre_shift')
    estimate_preshift(test_model, [RecorderACC])

    test_model = test_model.to('cpu')
    output_dict = {}
    for name, module in test_model.named_modules():
        if hasattr(module,'pre_shift'): 
            name_i = deepcopy(name)
            name_i = name_i[:-2]
            output_dict[name_i] = deepcopy(int(module.pre_shift.item()))
    print('Output_Preshift_YAML')
    with open(log_dir + '/pre_shift.yaml', 'w') as f:
            yaml.dump(output_dict, f, default_flow_style=False, sort_keys=False, width=100)



def reuse_vertices(model,dummy_input):
    reuse_list=[]
    DAG_object = DAGraph()
    DAG_object._build_graph(model, dummy_input, [torch.nn.Identity,Constant_module,CalQuantMeasure,QuantMeasure])
    vertices = DAG_object.proxy.getVertices()
    edges = DAG_object.proxy.getEdges()
    shapes=DAG_object.proxy.getOutShapes()
    for layer_idx in vertices:
        bot = edges[layer_idx]
        if 'mul' in str(bot) or 'add' in str(bot) or 'sub' in str(bot) or 'div' in str(bot) or 'matmul' in str(bot):
            collector=[]
            #print("this roundssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss")
            source_layer=bot.pop(0)
            g=edges[source_layer]
            for layer_idx2 in edges:
                if edges[layer_idx2]==g and layer_idx2!=layer_idx:
                    collector.append(layer_idx2)
            if len(collector)>0:
                #print(collector)
                collector.remove(source_layer)
                #print(collector)
                reuse_list=reuse_list+collector
    return reuse_list

"""
quantize_model_loader filter not necessary keys before input load_quantize_model 
"""

def multi_output_idd(test_model):
    trace = CustomTracer()
    fx_trace = trace.trace(test_model)
    fx_model = fx.graph_module.GraphModule(trace.root, fx_trace)
    target_node_list = [] #list for recording output node
    for node in fx_model.graph.nodes:
        if 'output' == node.op:
            for target in node.args:
                if isinstance(target,tuple):
                    for target_node in target:
                        if isinstance(target_node,torch.fx.Node):
                            target_node_list.append(target_node)
                elif isinstance(target,torch.fx.Node):
                    target_node_list.append(target)
    env={}
    model_dict = dict(fx_model.named_modules())
    new_graph = fx.Graph()
    for node in fx_model.graph.nodes:
        if node in target_node_list:
            new_node = new_graph.node_copy(node, lambda x: env[x.name])
            env[node.name] = new_node
            setattr(fx_model, node.name + "_outidd", torch.nn.Identity())
            new_node = new_graph.call_module(node.name + "_outidd", args=(env[node.name],))
            env[new_node.name] = new_node
        elif 'mul' in node.name and 'built-in method' in str(node.target):
            new_node = new_graph.node_copy(node, lambda x: env[x.name])
            env[node.name] = new_node
        else:
            new_node = new_graph.node_copy(node, lambda x: env[x.name])
            env[node.name] = new_node
    #serach idd in new graph
    success_node = {}

    for node in new_graph.nodes:
        if '_outidd' in node.name:
            data={node.name:node}
            success_node.update(data)
    #concasde output to outidd
    for node in new_graph.nodes:
        if 'output' == node.op:
            if isinstance(node.args[0],tuple):
                new_args=()
                new_args0=()
                for target_node in node.args[0]:
                    new_args0=new_args0+(success_node[target_node.name+"_outidd"],)
                for idx,target_node in enumerate(node.args):
                    if idx==0:
                        new_args = new_args + (new_args0,)
                    else:
                        new_args = new_args + (target_node,)
                node.args = new_args
            else:
                new_args=()
                for target_node in node.args:
                    if isinstance(target_node,fx.node.Node):
                        new_args = new_args + (success_node[target_node.name+"_outidd"],)
                    else:
                        new_args = new_args + (target_node,)

    new_model = fx.GraphModule(fx_model, new_graph)
    return new_model

def insert_chunk_idd(test_model):
    trace2 = CustomTracer()
    fx_trace = trace2.trace(test_model)
    fx_model = fx.graph_module.GraphModule(trace2.root, fx_trace)
    target_node_list = []
    for node in fx_model.graph.nodes:
        if node.op == 'call_function' and isinstance(node.args[0],fx.node.Node):
            if 'chunk' in node.args[0].name:
                target_node_list.append(node)
    env={}
    model_dict = dict(fx_model.named_modules())
    new_graph = fx.Graph()
    for node in fx_model.graph.nodes:
        if node in target_node_list:
            new_node = new_graph.node_copy(node, lambda x: env[x.name])
            env[node.name] = new_node
            setattr(fx_model, node.name + "_chunkidd", Chunk_idd(node.args[1]))
            new_node = new_graph.call_module(node.name + "_chunkidd", args=(env[node.name],))
            env[new_node.name] = new_node
        else:
            new_node = new_graph.node_copy(node, lambda x: env[x.name])
            env[node.name] = new_node
    success_node = {}
    for node in new_graph.nodes:
        if '_chunkidd' in node.name:
            data={node.name:node}
            success_node.update(data)
    for node in new_graph.nodes:
        if not 'output' == node.name:
            if not (node.name in success_node) and len(node.args)>0:
                if isinstance(node.args[0],tuple):
                    new_args=()
                    new_args0=()
                    for args_ in node.args[0]:
                        if isinstance(args_,fx.node.Node) and (args_.name)+"_chunkidd" in success_node:
                            new_args0 = new_args0 + (success_node[(args_.name)+"_chunkidd"],)
                        else:
                            new_args0 = new_args0 + (args_, )
                    for idx,args_ in enumerate(node.args):
                        if idx==0:
                            new_args = new_args + (new_args0,)
                        else:
                            if isinstance(args_,fx.node.Node) and (args_.name)+"_chunkidd" in success_node:
                                new_args = new_args + (success_node[(args_.name)+"_chunkidd"],)
                            else:
                                new_args = new_args + (args_, )

                else:
                    # general gemm or method args
                    new_args=()
                    for args_ in node.args:
                        if isinstance(args_,fx.node.Node) and (args_.name)+"_chunkidd" in success_node:
                            new_args = new_args + (success_node[(args_.name)+"_chunkidd"],)
                        else:
                            new_args = new_args + (args_, )
                node.args=new_args
    new_model = fx.GraphModule(fx_model, new_graph)
    return new_model

def quantize_model_loader(work_path,export=False):
    need_list=['model_fp32','dummy_input','model_fp32_pth','model_quant_path','size0','size1','size2','fp32_min','fp32_max',
    'bits_weight','bits_activation', 'bits_bias','per_channel', 'w_symmetry','a_symmetry','sft_based',
    'calibration','sft_based','forge','Bcompensation','signed_e','Wcompensation','isReLU6','qat','preshift_path']
    with open(work_path+'/quantize_config.yaml', 'r') as f:
        quant_config_in = yaml.load(f, Loader=yaml.FullLoader)
    quant_config={}
    for target in need_list:
        if target in quant_config_in:
            quant_config[target]=quant_config_in[target]
        else:
            quant_config[target]=[]
    model_fq, DAG_object=load_quantize_model(work_path,**quant_config)
    if quant_config['qat'] and os.path.exists(work_path+"/quant_model_qat.pth"):
        print("do qat convert in load quant model")
        model_qat,metadict = FakeQuant_2_QAT_loader(model_fq, quant_config)
        model_qat.to("cuda:1")
        model_qat.load_state_dict(torch.load(work_path+"/quant_model_qat.pth"))
        model_fq,DAG_object=QAT_2_FQ_loader(model_qat,metadict,quant_config,work_path)
    if export:
        from common.fx_utils import operator2bimethod
        model_fq=operator2bimethod(model_fq)
        model_fq=insert_chunk_idd(model_fq)
        model_fq = multi_output_idd(model_fq)
        DAG_object = DAGraph()
        dummy_input = dummy_input_creator(quant_config_in['dummy_input'],quant_config_in['size0'],quant_config_in['size1'],quant_config_in['size2'])
        reuse_list=reuse_vertices(model_fq,dummy_input)
        DAG_object._build_graph(model_fq, dummy_input, [torch.nn.Identity,Constant_module,CalQuantMeasure,QuantMeasure])
        vertices = DAG_object.proxy.getVertices()
        edges = DAG_object.proxy.getEdges()
        for layer_id in reuse_list:
            vertices.pop(layer_id)
            edges.pop(layer_id)


    return model_fq,DAG_object,quant_config_in

def load_quantize_model(work_path,model_fp32, model_fp32_pth, model_quant_path, 
                         fp32_min,
                         fp32_max,dummy_input,
                         size0 = 1, size1 = 1, size2 = 1,
                         bits_weight = 8, bits_activation = 8, bits_bias = 8, 
                         per_channel = False, w_symmetry = True, a_symmetry = False, sft_based = False, 
                         forge = False, Bcompensation = False, signed_e = False, Wcompensation = False, 
                         isReLU6 = False, calibration = False, qat = False, preshift_path = None, device = torch.device('cpu')):
    print('Load Quantized Model...')
    # Data, Model process here.
    # Users should define their own Data and Model here 
    #___________________________________________________________________________________________________________________________
    data=dummy_input
    if len(data)<1:
        dummy_input = torch.zeros((1, size0, size1, size2))
        if size2==1:
            dummy_input = torch.zeros((1, size0, size1))
    elif len(data)==1:
        dummy_input = torch.zeros(data[0])
    else:
        dummy_input=()
        for data_size in data:
            dummy_input_tmp = torch.zeros(data_size)
            dummy_input=dummy_input+(dummy_input_tmp,)

    #data_1 = torch.zeros((1, 1, 640))

    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print("Pytorch model from andesConvert")
    #work_path="./mbv1ssd/"
    test_model=andesPickle.load(work_path+model_fp32)
    test_model=insert_constant(test_model,a_symmetry)
    test_model.eval()
    #test_model = add_idd(test_model)
    test_model.to('cpu')

    #for k in state_dict_all_quant.keys():
    #    if 'running_max' in k:
    #        dd_quant[k] = state_dict_all_quant[k]
    #    if 'running_min' in k:
    #        dd_quant[k] = state_dict_all_quant[k]

    #___________________________________________________________________________________________________________________________
    
    
    # Model numerical compensation here.
    # default fusion, then combination of the equalization, Bcompensation, and Wcompensation 
    #___________________________________________________________________________________________________________________________
    DAG_object = DAGraph()
    module_dict, activation_targ, targ_layer, ig_layer, cal_quant = Pytorch2FQ(calibration, isReLU6)
    test_model, DAG_object = switch_layers(test_model, DAG_object, dummy_input, module_dict, ignore_layer=ig_layer, quant_op=True, sym = w_symmetry)        
    vertices = DAG_object.proxy.getVertices()
    edges = DAG_object.proxy.getEdges()
    if sft_based and a_symmetry and w_symmetry:
        set_layer_bits_sft(vertices, bits_weight, bits_activation, bits_bias, targ_layer, cal = cal_quant)
    else:
        set_layer_bits_cal(vertices, bits_weight, bits_activation, bits_bias, targ_layer, cal = cal_quant)
    test_model = merge_batchnorm(test_model, vertices, edges, targ_layer)
    

    if forge:
        res = create_relation(vertices, edges, targ_layer, delete_single=False)
        if signed_e:            
            weight_forging(vertices, res, targ_layer, visualize_state=False, converge_thres=2e-3, converge_count = 20, signed=signed_e, s_range=[1e-8, 1e8], logger = False, log_dir = None) #original program : 2e-7
        else:
            weight_forging(vertices, res, targ_layer, visualize_state=False, converge_thres=2e-3, converge_count = 20, s_range=[1e-8, 1e8], logger = False, log_dir = None) #original program : 2e-7
        if Bcompensation:
            qbias_compensation(vertices, res, edges, 3, logger = False, log_dir = None) #some case that the quantile 3 may fails.
    if Wcompensation:
        if per_channel:
            qweight_compensation(vertices, edges, targ_layer, signed=w_symmetry, per_channel = True, logger = False, log_dir = None)
        else:
            if sft_based and a_symmetry and w_symmetry:
                qweight_compensation(vertices, edges, targ_layer, bits_weight, signed=w_symmetry, sft_based = sft_based, logger = False, log_dir = None)
            else:
                qweight_compensation(vertices, edges, targ_layer, bits_weight, signed=w_symmetry, logger = False, log_dir = None)
    #___________________________________________________________________________________________________________________________
    
    # QuantMeasure process here.
    # the position of the QuantMeasure should be checked 
    #___________________________________________________________________________________________________________________________
    activation_process(test_model, DAG_object, activation_targ, bits_activation)
    DAG2_object=DAGraph()
    DAG2_object._build_graph(test_model, dummy_input, [torch.nn.Identity,Constant_module,ig_layer[0]])
    vertices = DAG2_object.proxy.getVertices()
    edges = DAG2_object.proxy.getEdges()
    if sft_based and a_symmetry and w_symmetry:
        QuantMeasure_Manager(vertices, edges, quant_bits = bits_activation, calibration = calibration, sft_based = sft_based)
    else:
        QuantMeasure_Manager(vertices, edges, quant_bits = bits_activation, calibration = calibration)
        if a_symmetry:
             set_symmetry(test_model, [ig_layer[0]])
    
    if calibration:
        if per_channel:
            vertices = quantize_targ_layer_PC(vertices, bits_weight, bits_bias, targ_layer, sym=w_symmetry)
        else:
            if sft_based and a_symmetry and w_symmetry:
                quantize_targ_layer(vertices, bits_weight, bits_bias, targ_layer, sym=w_symmetry, sft_base = sft_based)
            else:
                vertices = quantize_targ_layer(vertices, bits_weight, bits_bias, targ_layer, sym=w_symmetry)
        test_model = test_model.to(device)
        test_model.eval()
        normal_eval(test_model, [ig_layer[0]])#[CalQuantMeasure])
    
    else:
        if per_channel:
            vertices = quantize_targ_layer_PC(vertices, bits_weight, bits_bias, targ_layer, sym=w_symmetry)
        else:
            if sft_based and a_symmetry and w_symmetry:
                quantize_targ_layer(vertices, bits_weight, bits_bias, targ_layer, sym=w_symmetry, sft_base = sft_based)
            else:
                vertices = quantize_targ_layer(vertices, bits_weight, bits_bias, targ_layer, sym=w_symmetry)

    for name, module in test_model.named_modules():
        if hasattr(module, 'running_min'):
            module.running_min = torch.tensor(0.)
            module.running_max = torch.tensor(0.)

    state_dict_all_quant = torch.load(work_path+model_quant_path, map_location=device)

    if hasattr(test_model, 'custom_tensor_op'):
        delattr(test_model, 'custom_tensor_op')
    for k in state_dict_all_quant.copy().keys():
        if k.startswith('custom_tensor_op.'):
            state_dict_all_quant.pop(k)
    dd_quant = state_dict_all_quant

    test_model.to(device)
    replace_op()
    restore_op()
    if not qat:
        test_model.load_state_dict(dd_quant)
        for name,module in test_model.named_modules():
            if hasattr(module,"update_stat"):
                module.training=False
                module.update_stat=False

    preshift = None
    if preshift_path != "None" and preshift_path != "None\n" and preshift_path != None and not qat:
        #print(preshift_path)
        file_name_preshift = "/"+quant_get_file_name(preshift_path)
        with open(work_path + file_name_preshift, 'r') as preshift_file:
            preshift = yaml.load(preshift_file.read(), Loader=yaml.FullLoader)

        for name, module in test_model.named_modules():
            if name in preshift.keys():
                setattr(module, 'est_preshift', preshift[name])
    """
    Add identity to output node for tflite
    """
    if False:
        """
        main_backend.py use
        """
        test_model=multi_output_idd(test_model)
    """
    Insert identity for chunk to storeage idx
    notice that this function must later than output_idd
    """
    #test_model=insert_chunk_idd(test_model)
    DAG3_object=DAGraph()
    DAG3_object._build_graph(test_model, dummy_input, [torch.nn.Identity,Chunk_idd,Constant_module,ig_layer[0]])
    
    #___________________________________________________________________________________________________________________________
    return test_model, DAG3_object 

def find_champion_index(quant_config):
    calibration = [True,False,True,True,False,False,False,False,False,False,False,False,False,False,False,True,True,True,True,True,True,True,True,True]
    forge= [False,False,False,False,False,False,True,True,True,True,True,True,True,True,False,False,True,True,True,True,True,True,True,True]
    Bcompensation = [False,False,False,False,False,False,False,False,True,True,False,False,True,True,False,False,False,False,True,True,False,False,True,True]
    Wcompensation = [False,True,False,True,False,True,False,True,False,True,False,True,False,True,False,True,False,True,False,True,False,True,False,True,]
    signed_e = [False,False,False,False,False,False,False,False,False,False,True,True,True,True,False,False,False,False,False,False,True,True,True,True]
    isReLU6 = [False,True,True,True,False,False,False,False,False,False,False,False,False,False,True,False,False,False,False,False,False,False,False,False]
    for i in range(len(calibration)):
        if (calibration[i]==quant_config["calibration"] and forge[i]==quant_config["forge"] and Bcompensation[i]==quant_config["Bcompensation"] and Wcompensation[i]==quant_config["Wcompensation"] and signed_e[i]==quant_config["signed_e"] and isReLU6[i]==quant_config["isReLU6"]):
            champion = i
            return champion

def quant_main_func(model_path, quant_config, forward_all, inference_all_f, inference_all, cal_dataloaders, dataloaders, data, data_config, bits_weight, bits_activation, bits_bias, quantile, per_channel, w_symmetry, a_symmetry, sft_based, log_dir, device, qat = False, specific_combine = -1):

    channel = data_config['channel']
    width = data_config['width']
    height = data_config['height']
    fp32_min = data_config['fp32_min']
    fp32_max = data_config['fp32_max']

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
    
    
    print()
    print(config)
    print()
    KL_ONLY=False
    if not w_symmetry:
        print()
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('  Asymmetric Weight Quant Noticed!! ')
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    results = [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]
    calibration = [True,False,True,True,False,False,False,False,False,False,False,False,False,False,False,True,True,True,True,True,True,True,True,True]
    forge= [False,False,False,False,False,False,True,True,True,True,True,True,True,True,False,False,True,True,True,True,True,True,True,True]
    Bcompensation = [False,False,False,False,False,False,False,False,True,True,False,False,True,True,False,False,False,False,True,True,False,False,True,True]
    Wcompensation = [False,True,False,True,False,True,False,True,False,True,False,True,False,True,False,True,False,True,False,True,False,True,False,True,]
    signed_e = [False,False,False,False,False,False,False,False,False,False,True,True,True,True,False,False,False,False,False,False,True,True,True,True]
    isReLU6 = [False,True,True,True,False,False,False,False,False,False,False,False,False,False,True,False,False,False,False,False,False,False,False,False]
    if not specific_combine == -1:
        results = [results[specific_combine]]
        calibration = [calibration[specific_combine]]
        forge= [forge[specific_combine]]
        Bcompensation = [Bcompensation[specific_combine]]
        Wcompensation = [Wcompensation[specific_combine]]
        signed_e = [signed_e[specific_combine]]
        isReLU6 = [isReLU6[specific_combine]]
    model_test = andesPickle.load(model_path)
    dummy_input_list=[]
    if isinstance(data,tuple):
        for element in data:
            dummy_input_list.append(list(element.size()))
    else:
        dummy_input_list.append(list(data.size()))
    test_model=insert_Quantstub(model_test,a_symmetry)
    #inference_all(test_model, dataloaders, data_config, device, a_symmetry, bits_activation,calibration=False)
    fp32_range_list=[]
    input_shape_list=[]
    for name,module in test_model.named_modules():
        if hasattr(module,'finder_min') and module.finder_min<=module.finder_max:
            fp32_range_list.append([float(module.finder_min.cpu().detach()),float(module.finder_max.cpu().detach())])
            input_shape_list.append([1]+list(module.shape)[1:])
    print(fp32_range_list)
    print(dummy_input_list)
    print(input_shape_list)
    if len(input_shape_list)>0:
        dummy_input_list=input_shape_list

    '''try:
        acc = inference_all(test_model, cal_dataloaders, data_config, device, a_symmetry, bits_activation,calibration=True)
    except Exception as e:
        cl, exc, tb = sys.exc_info()
        lastCallStack = traceback.extract_tb(tb)[-1]
        print(e)
        print(lastCallStack)
        print("calibration fail, Please check yout inference_FQ is also suitable for calibration dataset")
        exit()'''
   

    if quant_config:
        champion = find_champion_index(quant_config)
        model_test = andesPickle.load(model_path)
        results[champion] = quantize_combination(model = model_test, 
                                        cal_dataloaders = cal_dataloaders, 
                                        dataloaders = dataloaders,
                                        data_config = data_config,
                                        forward_all = forward_all,
                                        inference_all = inference_all,
                                        data = data, 
                                        bits_weight = bits_weight, 
                                        bits_activation = bits_activation, 
                                        bits_bias = bits_bias,
                                        quantile = quantile,
                                        per_channel = per_channel, 
                                        w_symmetry = w_symmetry, 
                                        a_symmetry = a_symmetry,
                                        sft_based = sft_based,
                                        forge = forge[champion], 
                                        Bcompensation = Bcompensation[champion], 
                                        signed_e = signed_e[champion], 
                                        Wcompensation = Wcompensation[champion], 
                                        isReLU6 = isReLU6[champion], 
                                        calibration = calibration[champion], 
                                        deployment = False,
                                        log_dir = log_dir,
                                        device = device)

        model_test.to("cpu")
        model_test = None
        del model_test
    else:
        for i in range(len(results)):
            print('__________________________________________________')
            txt_champion = "Dealing with"
            if isReLU6[i]:
                txt_champion = txt_champion + " isReLU6,"
            if forge[i]:
                txt_champion = txt_champion + " forge,"
            if signed_e[i]:
                txt_champion = txt_champion + " Signed,"
            if Bcompensation[i]:
                txt_champion = txt_champion + " Absorption,"
            if Wcompensation[i]:
                txt_champion = txt_champion + " Correction,"
            if calibration[i]:
                txt_champion = txt_champion + " Calibration."
            else:
                txt_champion = txt_champion + " DFQ."
            print(txt_champion)
            model_test = andesPickle.load(model_path)
            results[i] = quantize_combination(model = model_test, 
                                                    cal_dataloaders = cal_dataloaders, 
                                                    dataloaders = dataloaders,
                                                    data_config = data_config,
                                                    forward_all = forward_all,
                                                    inference_all = inference_all,
                                                    data = data, 
                                                    bits_weight = bits_weight, 
                                                    bits_activation = bits_activation, 
                                                    bits_bias = bits_bias,
                                                    quantile = quantile,
                                                    per_channel = per_channel, 
                                                    w_symmetry = w_symmetry, 
                                                    a_symmetry = a_symmetry,
                                                    sft_based = sft_based,
                                                    forge = forge[i], 
                                                    Bcompensation = Bcompensation[i], 
                                                    signed_e = signed_e[i], 
                                                    Wcompensation = Wcompensation[i], 
                                                    isReLU6 = isReLU6[i], 
                                                    calibration = calibration[i], 
                                                    deployment = False,
                                                    log_dir = log_dir,
                                                    device = device)
            #except AttributeError:
                #print("The model contains no Batch_Normal layer.")
            #except TypeError:
                #print("Set_Quant_minMax fails")
            #except AssertionError:
                #print("Unkown Structure.")
            model_test.to("cpu")
            model_test = None
            del model_test
    
    if all(item == 0. for item in results):
        print()
        raise AttributeError('!!!!!~~~~~ All Combinations Fails, please check model ~~~~~!!!!!')

    print()
    champion = results.index(max(results))
    acc_quant = float(max(results))
    txt_champion = "The champion is"
    if isReLU6[champion]:
        txt_champion = txt_champion + " isReLU6,"
    if forge[champion]:
        txt_champion = txt_champion + " forge,"
    if signed_e[champion]:
        txt_champion = txt_champion + " Signed,"
    if Bcompensation[champion]:
        txt_champion = txt_champion + " Absorption,"
    if Wcompensation[champion]:
        txt_champion = txt_champion + " Correction," 
    if calibration[champion]:
        txt_champion = txt_champion + " Calibration."
    else:
        txt_champion = txt_champion + " DFQ."    
    print(txt_champion)
    if specific_combine==-1:
        print("Combination number is "+str(champion))
    else:
        print("Combination number is "+str(specific_combine))
    print("Accuracy: ", results[champion], '%')
    print()
    
    '''if (w_symmetry) and (a_symmetry) and (not per_channel) and (not sft_based):
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        model_test = andesPickle.load(model_path)
        find_preshifts(model = model_test, 
                       cal_dataloaders = cal_dataloaders,  
                       data_config = data_config,
                       forward_all = forward_all, 
                       data = data,
                       min_i = fp32_min, 
                       max_i = fp32_max,
                       fp32_range_list = fp32_range_list,
                       quantile = quantile,
                       forge = forge[champion], 
                       Bcompensation = Bcompensation[champion], 
                       signed_e = signed_e[champion], 
                       Wcompensation = Wcompensation[champion], 
                       isReLU6 = isReLU6[champion], 
                       calibration = calibration[champion], 
                       log_dir = log_dir, 
                       device = device)
        print('')
        model_test.to('cpu')
        model_test = None
        del model_test'''

    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('output Fake_quant model')
    model_test = andesPickle.load(model_path)
    output_model,acc_quant,fp32_range_list = quantize_combination(model = model_test, 
                                        cal_dataloaders = cal_dataloaders, 
                                        dataloaders = dataloaders,
                                        data_config = data_config,
                                        forward_all = forward_all,
                                        inference_all = inference_all,
                                        data = data, 
                                        bits_weight = bits_weight, 
                                        bits_activation = bits_activation, 
                                        bits_bias = bits_bias,
                                        quantile = quantile, 
                                        per_channel = per_channel, 
                                        w_symmetry = w_symmetry, 
                                        a_symmetry = a_symmetry,
                                        sft_based = sft_based,
                                        forge = forge[champion], 
                                        Bcompensation = Bcompensation[champion], 
                                        signed_e = signed_e[champion], 
                                        Wcompensation = Wcompensation[champion], 
                                        isReLU6 = isReLU6[champion], 
                                        calibration = calibration[champion], 
                                        deployment = True,
                                        log_dir = log_dir,
                                        device = device)
    print("save model result pth first")
    torch.save(output_model.state_dict(), log_dir+'/quant_model.pth')

    print("delete saved model and find preshift for symmetry model ")
    output_model.to('cpu')
    output_model=None
    del output_model
    if (w_symmetry) and (a_symmetry) and (not per_channel) and (not sft_based):
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        model_test = andesPickle.load(model_path)
        find_preshifts(model = model_test,
                       cal_dataloaders = cal_dataloaders,
                       data_config = data_config,
                       forward_all = forward_all,
                       data = data,
                       min_i = fp32_min,
                       max_i = fp32_max,
                       fp32_range_list = fp32_range_list,
                       quantile = quantile,
                       forge = forge[champion],
                       Bcompensation = Bcompensation[champion],
                       signed_e = signed_e[champion],
                       Wcompensation = Wcompensation[champion],
                       isReLU6 = isReLU6[champion],
                       calibration = calibration[champion],
                       log_dir = log_dir,
                       device = device)
        print('')
        model_test.to('cpu')
        model_test = None
        del model_test

    pre_shift_path = 'None'
    if (w_symmetry) and (a_symmetry) and (not per_channel) and (not sft_based):
        pre_shift_path = '/pre_shift.yaml'

    champion_parameter = {'model_fp32': '/model_fp32.pickle',
                          'model_fp32_pth': '/model_fp32.pth',     
                          'size0': channel, 
                          'size1': height, 
                          'size2': width,
                          'dummy_input': dummy_input_list,
                          'bits_weight': bits_weight, 
                          'bits_activation': bits_activation, 
                          'bits_bias': bits_bias,
                          'per_channel': per_channel,
                          'w_symmetry': w_symmetry, 
                          'a_symmetry': a_symmetry,
                          'sft_based': sft_based, 
                          'forge': forge[champion], 
                          'Bcompensation': Bcompensation[champion], 
                          'signed_e': signed_e[champion], 
                          'Wcompensation': Wcompensation[champion], 
                          'isReLU6': isReLU6[champion], 
                          'calibration': calibration[champion], 
                          'qat': qat,
                          'model_quant_path': '/quant_model.pth',
                          'preshift_path': pre_shift_path,
                          'fp32_range_array': fp32_range_list,
                          'fp32_min': fp32_min,
                          'fp32_max': fp32_max,
                          'acc_quant': float(acc_quant)}
    import yaml
    with open(log_dir +'/quantize_config.yaml', 'w') as f:
        yaml.dump(champion_parameter, f, default_flow_style=False, sort_keys=False, width=100)
    print("check load process")
    output_model, transformer,quant_config = quantize_model_loader(log_dir)
    print(output_model)
    
    return output_model
