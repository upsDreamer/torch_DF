#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 13:42:17 2021

@author: aitester
"""

import sys
sys.path.append("/Work/audio_algo/zy/Pro/DeepFilterNet/DeepFilterNet/nn_deploy/AndesAIRE-NN-SDK/NNPilot")

import torch
import torch.fx as fx
import copy
import operator
import numpy as np
from common.onnx2torch import node_converters
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

idd_list=['<built-in method cat','<built-in method add','<built-in method sub','<built-in method mul','<built-in method div','<built-in method matmul','<built-in method sum']

def _get_slices(
    starts: Union[torch.Tensor, np.ndarray],
    ends: Union[torch.Tensor, np.ndarray],
    axes: Optional[Union[torch.Tensor, np.ndarray]],
    steps: Optional[Union[torch.Tensor, np.ndarray]],
) -> Tuple[List, List, List]:
    if axes is None:
        axes = list(range(len(starts)))
    else:
        axes = axes.detach().cpu().tolist() #.numpy()

    if steps is None:
        steps = [1] * len(starts)
    else:
        steps = steps.detach().cpu().tolist() #.numpy()

    slices = {}
    flip_dims = []
    for start, end, axis, step in zip(starts, ends, axes, steps):
        if step < 0:
            flip_dims.append(axis)
            start, end, step = -start - 1, -end - 1, -step

        slices[axis] = slice(start, end, step)

    pos_axes_slices = list(slices.get(a, slice(None, None)) for a in range(max(axes) + 1))
    neg_axes_slices = list(slices.get(a, slice(None, None)) for a in range(min(axes), 0))

    if neg_axes_slices:
        neg_axes_slices = [Ellipsis] + neg_axes_slices

    return flip_dims, pos_axes_slices, neg_axes_slices


def get_value_from_module(module,target):
    attr_list=target.split(".")
    for idx,attr in enumerate(attr_list):
        if idx==0:
            re=getattr(module,attr)
        else:
            re=getattr(re,attr)
    value=re
    return value

class CustomTracer(fx.Tracer):
    def is_leaf_module(self, m,module_qualified_name: str):
        return (m.__module__.startswith('torch.nn') and
                    not isinstance(m, torch.nn.Sequential)) or \
                    hasattr(m, "leaf_m_register") or hasattr(m, "running_min") or hasattr(m, "quant")

class BN_decompose(torch.nn.Module):
    def __init__(self,bn_module):
        super().__init__()
        bn_weight = bn_module.weight.detach()
        bn_mean = bn_module.running_mean.detach()
        bn_bias = bn_module.bias.detach()
        bn_var = bn_module.running_var.detach()
        bn_eps = bn_module.eps
        var_sqrt = torch.sqrt(bn_var+bn_eps)
        self.scalar = (bn_weight / var_sqrt)
        scale=(torch.max(self.scalar)-torch.min(self.scalar)).div(255)
        zp=torch.min(self.scalar).div(scale).round()
        if zp>0:
            zp=0
        self.scalar = self.scalar.div(scale).round().sub(zp).clamp(0,255).add(zp).mul(scale)
        self.shift = (bn_bias-(bn_mean/var_sqrt*bn_weight))
        scale=(torch.max(self.shift)-torch.min(self.shift)).div(255)
        zp=torch.min(self.shift).div(scale).round()
        if zp>0:
            zp=0
        self.shift = self.shift.div(scale).round().sub(zp).clamp(0,255).add(zp).mul(scale)
    def forward(self,x):
        return torch.add(torch.mul(x,self.scalar),self.shift)

class GRU_decompose(torch.nn.Module):
    def __init__(self, gru: torch.nn.GRUCell) -> None:
        super().__init__()
        is_bias = hasattr(gru, "bias")
        self.linear_ih = torch.nn.Linear(gru.weight_ih.shape[1], gru.weight_ih.shape[0], bias=is_bias)
        self.linear_hh = torch.nn.Linear(gru.weight_hh.shape[1], gru.weight_hh.shape[0], bias=is_bias)
        self.linear_ih.weight.data.copy_(gru.weight_ih)
        self.linear_hh.weight.data.copy_(gru.weight_hh)
        if is_bias:
            self.linear_ih.bias.data.copy_(gru.bias_ih)
            self.linear_hh.bias.data.copy_(gru.bias_hh)

    def forward(self, input: torch.Tensor, hx: torch.Tensor) -> torch.Tensor:
        ih = self.linear_ih(input).chunk(3, dim=-1)
        hh = self.linear_hh(hx).chunk(3, dim=-1)
        r = torch.sigmoid(ih[0] + hh[0])
        z = torch.sigmoid(ih[1] + hh[1])
        n = torch.tanh(ih[2] + r * hh[2])
        return (1 - z) * n + z * hx

class LSTM_decompose(torch.nn.Module):
    def __init__(self, gru: torch.nn.LSTMCell) -> None:
        super().__init__()
        is_bias = hasattr(gru, "bias")
        self.linear_ih = torch.nn.Linear(gru.weight_ih.shape[1], gru.weight_ih.shape[0], bias=is_bias)
        self.linear_hh = torch.nn.Linear(gru.weight_hh.shape[1], gru.weight_hh.shape[0], bias=is_bias)
        self.linear_ih.weight.data.copy_(gru.weight_ih)
        self.linear_hh.weight.data.copy_(gru.weight_hh)
        if is_bias:
            self.linear_ih.bias.data.copy_(gru.bias_ih)
            self.linear_hh.bias.data.copy_(gru.bias_hh)

    def forward(self, input: torch.Tensor, hx: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        ih = self.linear_ih(input).chunk(4, dim=-1)
        hh = self.linear_hh(hx).chunk(4, dim=-1)
        i = torch.sigmoid(ih[0] + hh[0])
        f = torch.sigmoid(ih[1] + hh[1])
        g = torch.tanh(ih[2] + hh[2])
        o = torch.sigmoid(ih[3] + hh[3])
        c = f * c + i * g
        return o * torch.tanh(c), c

class HardSigmoid_decompose(torch.nn.Module):
    """docstring for QuantMeasure."""

    def __init__(self):
        super(HardSigmoid_decompose, self).__init__()
        self.relu6_hardsigmoid=torch.nn.ReLU6()

    def forward(self, input):
        #a = self.tanh(torch.log(torch.exp(input) + 1))
        a = self.relu6_hardsigmoid(input)
        output = (1/6)*a
        return output
def get_layer(model, name):
    layer = model
    for attr in name.split("."):
        layer = getattr(layer, attr)
    return layer
def set_layer(model, name, layer):
    try:
        attrs, name = name.rsplit(".", 1)
        model = get_layer(model, attrs)
    except ValueError:
        pass
    setattr(model, name, layer)

def hardsigmoid_replace(model: torch.nn.Module) -> torch.nn.Module:
    trace = CustomTracer()
    fx_trace = trace.trace(model)
    fx_model = fx.graph_module.GraphModule(trace.root, fx_trace)
    model_dict = dict(fx_model.named_modules())
    replace_dict={}
    for node in fx_model.graph.nodes:
        if node.op=="call_module":
            if isinstance(model_dict[node.target],torch.nn.Hardsigmoid):
                replace_module=model_dict[node.args[0].target]
                replace_module.bias.data=replace_module.bias.data+torch.tensor(3.0)
                model_dict[node.args[0].target]=replace_module
                replace_dict.update({node.args[0].target:replace_module})
    for name,module in model.named_modules():
        if name in replace_dict:
            set_layer(model,name,replace_dict[name])
    model_result=fx.GraphModule(model, fx_model.graph)
    for name,module in model_result.named_modules():
        if isinstance(module,torch.nn.Hardsigmoid):
            set_layer(model_result,name,HardSigmoid_decompose())
    return model_result

def bn_alone_replace(model: torch.nn.Module) -> torch.nn.Module:
    trace = CustomTracer()
    fx_trace = trace.trace(model)
    fx_model = fx.graph_module.GraphModule(trace.root, fx_trace)
    replace_dict={}
    for node in fx_model.graph.nodes:
        if node.op=="call_module":
            if isinstance(get_value_from_module(fx_model,node.target),torch.nn.BatchNorm1d):
                if node.args[0].op=="call_module" and (isinstance(get_value_from_module(fx_model,node.args[0].target),torch.nn.Linear) or isinstance(get_value_from_module(fx_model,node.args[0].target),torch.nn.Conv1d)):
                    pass #exist linear to fuse
                else:
                    """
                    deal with bn
                    """
                    module = get_value_from_module(fx_model,node.target)
                    ln1=torch.nn.Linear(module.num_features,module.num_features,True)
                    ln1.weight.data=torch.diag(torch.ones(module.num_features))
                    ln1.bias.data=torch.zeros(module.num_features)
                    a = torch.nn.Sequential(
                            ln1,
                            module
                            )
                    fx_model.add_module(node.name+'_newmodule', a)
                    node.target=node.name+'_newmodule'
            if isinstance(get_value_from_module(fx_model,node.target),torch.nn.BatchNorm2d):
                if node.args[0].op=="call_module" and isinstance(get_value_from_module(fx_model,node.args[0].target),torch.nn.Conv2d):
                    pass
                else:
                    module = get_value_from_module(fx_model,node.target)
                    bn_conv=torch.nn.Conv2d(module.num_features,module.num_features,1,groups=module.num_features)
                    bn_conv.weight.data=torch.ones(bn_conv.weight.data.size())
                    bn_conv.bias.data=torch.zeros(bn_conv.bias.data.size())
                    a = torch.nn.Sequential(
                            bn_conv,
                            module
                            )
                    fx_model.add_module(node.name+'_newmodule', a)
                    node.target=node.name+'_newmodule'

    model_result=fx.GraphModule(fx_model, fx_model.graph)
    return model_result

def get_layer(model, name):
    layer = model
    for attr in name.split("."):
        layer = getattr(layer, attr)
    return layer
def set_layer(model, name, layer):
    try:
        attrs, name = name.rsplit(".", 1)
        model = get_layer(model, attrs)
    except ValueError:
        pass
    setattr(model, name, layer)

class SiLU_decompose(torch.nn.Module):
    """docstring for QuantMeasure."""

    def __init__(self):
        super(SiLU_decompose, self).__init__()
        self.first=torch.nn.Sigmoid()

    def forward(self, input):
        output=input*self.first(input)
        return output

class Identity_decompose(torch.nn.Module):
    """docstring for QuantMeasure."""

    def __init__(self):
        super(Identity_decompose, self).__init__()

    def forward(self, input):
        return input

class Constant_(torch.nn.Module):
    """docstring for QuantMeasure."""

    def __init__(self,weight):
        super(Constant_, self).__init__()
        self.weight=weight
        self.cons_m=True
        self.leaf_m_register=True

    def forward(self,input):
        return self.weight
class Matmul_module(torch.nn.Module):
    """docstring for QuantMeasure."""

    def __init__(self):
        super(Matmul_module, self).__init__()
        self.leaf_m_register=True

    def forward(self,input1,input2):
        return torch.matmul(input1,input2)

class Transpose_linear(torch.nn.Module):
    """docstring for QuantMeasure."""

    def __init__(self,weight):
        super(Transpose_linear, self).__init__()
        self.constant=Constant_(weight)
        #self.matmul_m=Matmul_module()

    def forward(self, input):
        a=self.constant(input)
        output=torch.matmul(a,input)
        return output

class Mish_decompose(torch.nn.Module):
    """docstring for QuantMeasure."""

    def __init__(self):
        super(Mish_decompose, self).__init__()
        self.tanh=torch.nn.Tanh()
        self.softplus=torch.nn.Softplus()

    def forward(self, input):
        #a = self.tanh(torch.log(torch.exp(input) + 1))
        a = self.tanh(self.softplus(input))
        output = input*a
        return output

class AndesSlice(torch.nn.Module):
    """docstring for QuantMeasure."""

    def __init__(self,start,end,axes,step,result):
        super(AndesSlice, self).__init__()
        self.start=start
        self.end=end
        self.axes=axes
        self.step=step
        self.result=result
        self.leaf_m_register=True

    def forward(self, input):
        output = input[self.result]
        return output
    def extra_repr(self) -> str:
        return f'start={self.start}, end={self.end}, axes={self.axes}, step={self.step}'

def method2bimethod(model: torch.nn.Module) -> torch.nn.Module:
    OP2METHOD = {
        "sub": torch.sub,
    }
    trace = CustomTracer()
    fx_trace = trace.trace(model)
    fx_model = fx.graph_module.GraphModule(trace.root, fx_trace)
    model_dict = dict(fx_model.named_modules())
    for node in fx_model.graph.nodes:
        if node.op=="call_method" and str(node.target) in OP2METHOD:
            node.op="call_function"
            node.target=OP2METHOD[node.target]
            new_args=()
            for pre_node in node.args:
                if pre_node.op=="get_attr":
                    *parent, sub_attr = str(pre_node.target).rsplit(".", 1)
                    if parent:
                        parent=parent[0]
                    else:
                        parent="noparent"
                    if parent in model_dict:
                        constant_tensor=getattr(model_dict[parent],sub_attr).item()
                        new_args=new_args+(constant_tensor,)
                    else:
                        new_args=new_args+(pre_node,)
                else:
                    new_args=new_args+(pre_node,)
            node.args=new_args
    fx_model.graph.lint()
    model = torch.fx.GraphModule(fx_model, fx_model.graph)
    model.recompile()
    return model

def input_replace(model: torch.nn.Module) -> torch.nn.Module:
    """
    convert call-function operator to build in method
    """
    input_replace=[]
    trace = CustomTracer()
    fx_trace = trace.trace(model)
    fx_model = fx.graph_module.GraphModule(trace.root, fx_trace)
    for node in fx_model.graph.nodes:
        if node.op=="call_function":
            for tn in idd_list:
                if tn in str(node.target):
                    for prev_node in node.args:
                        if isinstance(prev_node,(tuple,list)):
                            for ele in prev_node:
                                if isinstance(ele,fx.Node) and ele.op=="get_attr":
                                    value=get_value_from_module(fx_model,ele.target)
                                    if torch.count_nonzero(value)==0:
                                        input_replace.append(ele.name)
                        elif isinstance(prev_node,fx.Node) and prev_node.op=="get_attr":
                            value=get_value_from_module(fx_model,prev_node.target)
                            if torch.count_nonzero(value)==0:
                                input_replace.append(prev_node.name)
    """
    start to change to input
    """
    for node in fx_model.graph.nodes:
        if node.name in input_replace:
            node.op="placeholder"
            node.target=node.name
    model = torch.fx.GraphModule(fx_model, fx_model.graph)
    return model

def reshape_replace(model: torch.nn.Module) -> torch.nn.Module:
    trace = CustomTracer()
    fx_trace = trace.trace(model)
    fx_model = fx.graph_module.GraphModule(trace.root, fx_trace)
    for node in fx_model.graph.nodes:
        if node.op=="call_function" or node.op=="call_method":
            if 'reshape' in str(node.target):
                if len(node.args)>1 and isinstance(node.args[1],(tuple,list)):
                    dy_shape=False
                    new_shape=[]
                    for ele in node.args[1]:
                        if ele==-1:
                            dy_shape=True
                    if dy_shape:
                        continue
                    else:
                        for ele in node.args[1]:
                            new_shape.append(ele)
                        new_shape[0]=-1
                        new_args=tuple(new_shape)
                        change_node=True
                if change_node:
                    node.args=(node.args[0],new_args)
    model = torch.fx.GraphModule(fx_model, fx_model.graph)
    return model

def operator2bimethod(model: torch.nn.Module) -> torch.nn.Module:
    """
    convert call-function operator to build in method
    """
    trace = CustomTracer()
    fx_trace = trace.trace(model)
    fx_model = fx.graph_module.GraphModule(trace.root, fx_trace)
    for node in fx_model.graph.nodes:
        """
        add
        """
        if node.target == 'add' and node.op == 'call_method':
            node.op='call_function'
            node.target=torch.add
        if str(node.target) == '<built-in function add>' and node.op == 'call_function':
            node.target=torch.add
        """
        sub
        """
        if node.target == 'sub' and node.op == 'call_method':
            node.op='call_function'
            node.target=torch.sub
        if str(node.target) == '<built-in function sub>' and node.op == 'call_function':
            node.target=torch.sub
        """
        mul
        """
        if node.target == 'mul' and node.op == 'call_method':
            node.op='call_function'
            node.target=torch.mul
        if str(node.target) == '<built-in function mul>' and node.op == 'call_function':
            node.target=torch.mul
        """
        div
        """
        if node.target == 'div' and node.op == 'call_method':
            node.op='call_function'
            node.target=torch.div
        if str(node.target) == '<built-in function div>' and node.op == 'call_function':
            node.target=torch.div
        """
        matmul
        """
        if node.target == 'matmul' and node.op == 'call_method':
            node.op='call_function'
            node.target=torch.matmul
        if str(node.target) == '<built-in function matmul>' and node.op == 'call_function':
            node.target=torch.matmul
        """
        squeeze
        """
        if node.target == 'squeeze' and node.op == 'call_method':
            node.op='call_function'
            node.target=torch.squeeze
        if "<built-in method squeeze" in str(node.target) and node.op == 'call_function':
            new_args=()
            new_args=new_args+node.args
            for key in node.kwargs.keys():
                if isinstance(node.kwargs[key],fx.Node):
                    value=get_value_from_module(fx_model,node.kwargs[key].target)
                    new_args=new_args+(int(value),)
                    #new_node = fx_model.graph.call_function(torch.squeeze, args=(node.args[0],int(value)))
                else:
                    new_args=new_args+(node.kwargs[key],)
                    #new_node = fx_model.graph.call_function(torch.squeeze, args=(node.args[0],node.kwargs['dim']))
            new_node = fx_model.graph.call_function(torch.squeeze, args=new_args)
            node.args=new_node.args
            node.kwargs=new_node.kwargs
            fx_model.graph.erase_node(new_node)
        """
        unsqueeze
        """
        if node.target == 'unsqueeze' and node.op == 'call_method':
            node.op='call_function'
            node.target=torch.unsqueeze
        if "<built-in method unsqueeze" in str(node.target) and node.op == 'call_function':
            new_args=()
            new_args=new_args+node.args
            for key in node.kwargs.keys():
                if isinstance(node.kwargs[key],fx.Node):
                    value=get_value_from_module(fx_model,node.kwargs[key].target)
                    # print(value)
                    new_args=new_args+(int(value),)
                    #new_node = fx_model.graph.call_function(torch.squeeze, args=(node.args[0],int(value)))
                else:
                    new_args=new_args+(node.kwargs[key],)
                    #new_node = fx_model.graph.call_function(torch.squeeze, args=(node.args[0],node.kwargs['dim']))
            new_node = fx_model.graph.call_function(torch.unsqueeze, args=new_args)
            node.args=new_node.args
            node.kwargs=new_node.kwargs
            fx_model.graph.erase_node(new_node)
        """
        sum
        """
        if node.target == 'sum' and node.op == 'call_method':
            node.op='call_function'
            node.target=torch.sum
        if "<built-in method sum" in str(node.target) and node.op == 'call_function':
            new_args=()
            new_args=new_args+node.args
            for key in node.kwargs.keys():
                if isinstance(node.kwargs[key],fx.Node):
                    value=get_value_from_module(fx_model,node.kwargs[key].target)
                    new_args=new_args+(int(value),)
                    #new_node = fx_model.graph.call_function(torch.squeeze, args=(node.args[0],int(value)))
                else:
                    new_args=new_args+(node.kwargs[key],)
                    #new_node = fx_model.graph.call_function(torch.squeeze, args=(node.args[0],node.kwargs['dim']))
            new_node = fx_model.graph.call_function(torch.sum, args=new_args)
            node.args=new_node.args
            node.kwargs=new_node.kwargs
            fx_model.graph.erase_node(new_node)
        """
        chunk
        """
        if node.target == 'chunk' and node.op == 'call_method':
            node.op='call_function'
            node.target=torch.chunk
        if "<built-in method chunk" in str(node.target) and node.op == 'call_function':
            new_args=()
            new_args=new_args+node.args
            for key in node.kwargs.keys():
                if isinstance(node.kwargs[key],fx.Node):
                    value=get_value_from_module(fx_model,node.kwargs[key].target)
                    new_args=new_args+(int(value),)
                    #new_node = fx_model.graph.call_function(torch.squeeze, args=(node.args[0],int(value)))
                else:
                    new_args=new_args+(node.kwargs[key],)
                    #new_node = fx_model.graph.call_function(torch.squeeze, args=(node.args[0],node.kwargs['dim']))
            new_node = fx_model.graph.call_function(torch.chunk, args=new_args)
            node.args=new_node.args
            node.kwargs=new_node.kwargs
            fx_model.graph.erase_node(new_node)

    fx_model.graph.lint()
    model = torch.fx.GraphModule(model, fx_model.graph)
    model.recompile()
    return model

def function_replace(model):
    trace = CustomTracer()
    fx_trace = trace.trace(model)
    fx_model = fx.graph_module.GraphModule(trace.root, fx_trace)
    for node in fx_model.graph.nodes:
        if "<built-in method tanh of type object" in str(node.target):
            setattr(fx_model, node.name + '_replace_module', torch.nn.Tanh())
            node.op="call_module"
            node.target = node.name + '_replace_module'
    model = torch.fx.GraphModule(fx_model, fx_model.graph)
    model.recompile()
    return model

def transpose_simplify(model):
    model.eval()
    trace = CustomTracer()
    fx_trace = trace.trace(model)
    fx_model = fx.graph_module.GraphModule(trace.root, fx_trace)
    remove_list=[]
    for node in fx_model.graph.nodes:
        if 'permute' in str(node.target) and node.args[0].op=='call_module':
            """
            transpose --> linear --> transpose
            """
            if isinstance(get_value_from_module(fx_model,node.args[0].target),torch.nn.Linear):
                if 'permute' in str(node.args[0].args[0].target):
                    remove_list.append(node.args[0].args[0].name)
                    new_weight=get_value_from_module(fx_model,node.args[0].target).weight
                    fx_model.add_module(node.args[0].name+'module', Transpose_linear(new_weight))
                    node.args[0].target=node.args[0].name+'module'
                    remove_list.append(node.name)
            """
            transpose --> matmul --> transpose
            """
        if 'permute' in str(node.target) and node.args[0].op=='call_function':
            if 'matmul' in str(node.args[0].target):
                prev_node=node.args[0]
                permute_exist=False
                for idx,args_node in enumerate(prev_node.args):
                    if 'permute' in str(args_node.target):
                        permute_exist= True
                        permute_idx=idx
                if permute_exist:
                    #remove_list.append(node.args[0].args[0].name)
                    if permute_idx==0:
                        #new_weight=getattr(fx_model,node.args[0].target).weight
                        pass
                    else:
                        remove_list.append(prev_node.args[permute_idx].name)
                        weight=get_value_from_module(fx_model,prev_node.args[0].target)
                        in_features, out_features = weight.shape[1], weight.shape[0]
                        torch_module = torch.nn.Linear(
                            in_features=in_features,
                            out_features=out_features,
                            bias= None,
                            )
                        with torch.no_grad():
                            torch_module.weight.data = weight
                        fx_model.add_module(prev_node.name+'module', torch_module)
                        prev_node.target=prev_node.name+'module'
                        prev_node.args=(prev_node.args[permute_idx],)
                        prev_node.op="call_module"
                        remove_list.append(node.name)
    for node in fx_model.graph.nodes:
        if node.name in remove_list:
            node.op='call_module'
            fx_model.add_module(node.name+'module', Identity_decompose())
            node.target=node.name+'module'
            new_args=()
            new_args = new_args + (node.args[0],)
            node.args=new_args
    model = torch.fx.GraphModule(fx_model, fx_model.graph)
    return model

def slice_replace(model):
    trace = CustomTracer()
    fx_trace = trace.trace(model)
    fx_model = fx.graph_module.GraphModule(trace.root, fx_trace)
    for node in fx_model.graph.nodes:
        if node.op=="call_module":
            module=get_value_from_module(fx_model,node.target)
            if 'OnnxSlice' in str(module):
                Revise=False
                new_args=()
                new_args_slice=()
                for s_node in node.args:
                    if isinstance(s_node,fx.Node) and s_node.op=="get_attr" and s_node.args==():
                        Revise=True
                        value=get_value_from_module(fx_model,s_node.target)
                        #new_args = new_args + (value,)
                        new_args_slice = new_args_slice + (value,)

                    else:
                        new_args = new_args + (s_node,)
                if Revise:
                    node.args=new_args
                    a=_get_slices(new_args_slice[0],new_args_slice[1],new_args_slice[2],new_args_slice[3])
                    m=AndesSlice(new_args_slice[0],new_args_slice[1],new_args_slice[2],new_args_slice[3],a[1])
                    setattr(fx_model, node.name + '_replace_module', m)
                    node.target=node.name + '_replace_module'
    model = torch.fx.GraphModule(fx_model, fx_model.graph)
    model.recompile()
    trace = CustomTracer()
    fx_trace = trace.trace(model)
    model = fx.graph_module.GraphModule(trace.root, fx_trace)
    return model

def module_replace(model):
    for name ,module in model.named_modules():
        if isinstance(module,torch.nn.SiLU):
            b=SiLU_decompose()
            set_layer(model, name, b)
        if isinstance(module,torch.nn.Mish):
            b=Mish_decompose()
            set_layer(model, name, b)
        if isinstance(module,torch.nn.Identity) or isinstance(module,torch.nn.Dropout):
            b=Identity_decompose()
            set_layer(model, name, b)
        if isinstance(module,torch.nn.GRUCell):
            b=GRU_decompose(module)
            set_layer(model, name, b)
        if isinstance(module,torch.nn.LSTMCell):
            b=LSTM_decompose(module)
            set_layer(model, name, b)
    model=hardsigmoid_replace(model)
    return model

def andes_preprocessing(model: torch.nn.Module):
    model=dag_process(model)
    model=transpose_simplify(model)
    for name,module in model.named_modules():
        if isinstance(module, node_converters.OnnxGlobalAveragePoolWithKnownInputShape):
            for idx in module.x_dims:
                kernel_size=module.input_shape[idx]
            a=torch.nn.Conv2d(module.input_shape[1],module.input_shape[1],kernel_size,stride=1,groups=64)
            a.weight.data=torch.ones(a.weight.data.size()).mul(1/(kernel_size*kernel_size))
            a.bias.data=torch.zeros(a.bias.data.size())
            '''a = torch.nn.Sequential(
                            torch.nn.AdaptiveAvgPool2d(1),
                            )'''
            #set_layer(model, name, a)
    #model=method2bimethod(model)
    
    model=operator2bimethod(model)
    model=input_replace(model)
    model=function_replace(model)
    model=module_replace(model)
    model=operator2bimethod(model)
    model=slice_replace(model)
    model=reshape_replace(model)
    '''for name,module in model.named_modules():
        if isinstance(module,torch.nn.BatchNorm1d):
            ln1=torch.nn.Linear(module.num_features,module.num_features,True)
            ln1.weight.data=torch.diag(torch.ones(module.num_features))
            ln1.bias.data=torch.zeros(module.num_features)
            #a=BN_decompose(module)
            a = torch.nn.Sequential(
                            ln1,
                            module
                            )
            set_layer(model, name, a)
        if isinstance(module, torch.nn.AdaptiveAvgPool2d):
            a = torch.nn.Sequential(
                            module,
                            torch.nn.Identity()
                            )
            a=torch.nn.Conv2d(64,64,7,stride=1,groups=64)
            a.weight.data=torch.ones(a.weight.data.size()).mul(1/49)
            a.bias.data=torch.zeros(a.bias.data.size())
            #set_layer(model, name, a)'''
    model=bn_alone_replace(model)
    return model



def out_process(model: torch.nn.Module) -> torch.nn.Module:
    trace = CustomTracer()
    fx_trace = trace.trace(model)
    fx_model = fx.graph_module.GraphModule(trace.root, fx_trace)
    cat_list=[]
    cat_insert_list=[]
    insert_list=[]# add
    origin_list=[]#add_insert
    for node in fx_model.graph.nodes:
        if 'cat' in node.name and 'insert' in node.name:
            cat_insert_list.append(node)
        if 'cat_' in node.name and not 'insert' in node.name or node.name=='cat':
            cat_list.append(node)
        if 'add' in node.name and 'insert' in node.name:
            cat_insert_list.append(node)
        if 'add_' in node.name and not 'insert' in node.name or node.name=='add':
            cat_list.append(node)
        if 'mul' in node.name and 'insert' in node.name:
            cat_insert_list.append(node)
        if 'mul_' in node.name and not 'insert' in node.name or node.name=='mul':
            cat_list.append(node)
    for node in fx_model.graph.nodes:
        if 'insert' in node.name:
            insert_list.append(node)
            origin_list.append(node.args[0])

    #Dealing with cat_insert_idd fail to hang in multi-branch
    print("dealing_with_output")
    for node in fx_model.graph.nodes:
        if 'output' in node.name:
            if isinstance(node.args[0],type(node)):
                if node.args[0] in cat_list:
                    #need revise
                    node.args=((cat_insert_list[cat_list.index(node.args[0])]),)
                continue
            new_args = []
            for node_output in node.args[0]:
                if node_output in cat_list:
                    new_args.append(cat_insert_list[cat_list.index(node_output)])
                elif isinstance(node_output,list) or isinstance(node_output,tuple):
                    new_args2 = []
                    for element in node_output:
                        if element in cat_list:
                            new_args2.append(cat_insert_list[cat_list.index(element)])
                        else:
                            new_args2.append(element)

                    new_args.append(tuple(new_args2))
                else:
                    new_args.append(node_output)
            node.args=(tuple(new_args),)
            '''if isinstance(node.args[0],list):
                node.args=(tuple(new_args),)'''
        else:
            for target_node in origin_list:
                # generate node.args
                if (target_node in node.args) and not (node in insert_list):
                    new_args = ()
                    for node_output in node.args:
                        if node_output in origin_list:
                            new_args=new_args+(insert_list[origin_list.index(node_output)],)
                        else:
                            new_args=new_args+(node_output,)
                    node.args=new_args
                #cat node.args
                if node.target == torch.cat   and (target_node in node.args[0]):
                    new_args = ()
                    for node_output in node.args[0]:
                        if node_output in origin_list:
                            new_args=new_args+(insert_list[origin_list.index(node_output)],)
                        else:
                            new_args=new_args+(node_output,)
                    new_argss = (new_args,node.args[1])
                    node.args=new_argss
    #Dealing with permute from list to tuple
    for node in fx_model.graph.nodes:
        if 'permute' in node.name:
            new_args = ()
            for node_output in node.args:
                if isinstance(node_output,list) or isinstance(node_output,tuple):
                    for value in node_output:
                        new_args = new_args + (value,)
                else:
                    new_args = new_args + (node_output,)
                node.args=new_args
    fx_model.recompile()
    return fx_model

def add_idd(model: torch.nn.Module) -> torch.nn.Module:
    model.eval()
    #init=torch.zeros([1,64])
    #input_data=torch.zeros([1,1,2,16,64])
    #b=model(input_data,init,init)
    trace = CustomTracer()
    fx_trace = trace.trace(model)
    fx_model = fx.graph_module.GraphModule(trace.root, fx_trace)
    new_graph = fx.Graph()
    for name,module in fx_model.named_modules():
        if 'insert_idd' in name:
            return model
    env = {}
    record_node = None
    pre_node_name = None
    record_node_list=[]
    record_node_success_list=[]
    replace_dict={}
    ignore_list=[]
    count = 0
    for node in fx_model.graph.nodes:
        new_node = new_graph.node_copy(node, lambda x: env[x.name])
        env[node.name] = new_node
        for target_list in idd_list:
            if node.op=='call_function' and target_list in str(node.target):
                pre_node_name = node.name+'_insert_idd'
                setattr(model, pre_node_name, torch.nn.Identity())
                new_node = new_graph.call_module(pre_node_name, args=(env[node.name],))
                replace_dict.update({node.name:new_node})
                ignore_list.append(new_node.name)
    '''for node in fx_model.graph.nodes:
        if (str(node.target) in mapping_insert_idd) or (node.target == torch.cat) or any([ta in str(node.target) for ta in mapping_insert_idd_general]):
            if record_node in node.args:
                new_node = new_graph.node_copy(node, lambda x: env[x.name])
                env[node.name] = new_node
                pre_node_name = node.name
                idd = {}
                idd[torch.nn.Identity] = pre_node_name + '_insert_idd'
                setattr(model, pre_node_name + '_insert_idd', torch.nn.Identity())
                new_node = new_graph.call_module(idd[torch.nn.Identity], args=(env[node.name],))
                if hasattr(node.args[0],'name'):
                    env[node.args[0].name] = new_node
                else:#constant
                    env[node.args[0]] = new_node
            else:
                new_node = new_graph.node_copy(node, lambda x: env[x.name])
                env[node.name] = new_node
                pre_node_name = node.name
                idd = {}
                idd[torch.nn.Identity] = pre_node_name + '_insert_idd'
                setattr(model, pre_node_name + '_insert_idd', torch.nn.Identity())
                new_node = new_graph.call_module(idd[torch.nn.Identity], args=(env[node.name],))
            record_node = node
            record_node_list.append(record_node)
            record_node_success_list.append(new_node)
            count += 1
        elif record_node in node.args:
            new_node = new_graph.node_copy(node, lambda x: env[x.name])
            env[node.name] = new_node
        else:
            new_node = new_graph.node_copy(node, lambda x: env[x.name])
            env[node.name] = new_node
    record_node = None
    successor_record_node = None'''
    new_graph.lint()
    for node in new_graph.nodes:
        new_args=()
        if not (node.name in ignore_list):
            for node_out in node.args:
                if isinstance(node_out,(tuple,list)):
                    new_args_sub=()
                    for element in node_out:
                        if isinstance(element,torch.fx.Node):
                            if element.name in replace_dict:
                                new_args_sub=new_args_sub+(replace_dict[element.name],)
                            else:
                                new_args_sub=new_args_sub+(element,)
                        else:
                            new_args_sub=new_args_sub+(element,)
                    new_args=new_args+(new_args_sub,)
                elif isinstance(node_out,torch.fx.Node):
                    if node_out.name in replace_dict:
                        new_args=new_args+(replace_dict[node_out.name],)
                    else:
                        new_args=new_args+(node_out,)
                else:
                    new_args=new_args+(node_out,)
            node.args=new_args
    new_graph.print_tabular()
    model=fx.GraphModule(model, new_graph)
    return model

def test_fromONNX(model: torch.nn.Module) -> torch.nn.Module:
    #model = copy.deepcopy(model)
    trace = CustomTracer()
    fx_trace = trace.trace(model)
    fx_model = fx.graph_module.GraphModule(trace.root, fx_trace)
    new_graph = fx.Graph()
    env = {}

    input_node = None
    target_node = None
    for node in fx_model.graph.nodes:
        if 'input_1' in node.name:
            input_node = node
        elif 'mul' in node.name:
            new_args = ()
            new_args = new_args + (node.args[0],)
            new_args = new_args + (node.args[0],)
            node.args = new_args
            #node = None
        elif 'sub' in node.name:
            new_args = ()
            new_args = new_args + (node.args[1],)
            new_args = new_args + (node.args[1],)
            node.args = new_args
        elif 'permute' in node.name:
            target_node = node
        elif 'conv_0' in node.name:
            if target_node in node.args:
                new_args = ()
                for ii in range(len(node.args)):
                    if target_node == node.args[ii]:
                        new_args = new_args + (input_node,)
                node.args = new_args
    for node in fx_model.graph.nodes:
        if 'permute' in node.name:
            if node.name == 'permute':
                fx_model.graph.erase_node(node)
    for node in fx_model.graph.nodes:
        if 'sub' in node.name:
            fx_model.graph.erase_node(node)
    for node in fx_model.graph.nodes:
        if 'onnx_initializer_1' in node.name:
            fx_model.graph.erase_node(node)
    for node in fx_model.graph.nodes:
        if 'mul' in node.name:
            if 'mat' in node.name:
                continue
            else:
                fx_model.graph.erase_node(node)
    for node in fx_model.graph.nodes:
        if 'onnx_initializer_0' in node.name:
            fx_model.graph.erase_node(node)
    print("ONNX_model pre_processing")
    print("Pass torch.fx")
    fx_model.graph.lint()
    return fx.GraphModule(model, fx_model.graph)

def see_fx_graph(model: torch.nn.Module) -> torch.nn.Module:
    #model = copy.deepcopy(model) 
    fx_model: fx.GraphModule = fx.symbolic_trace(model)
    new_graph = fx.Graph()
    env = {}
    
    #record_node = None
    #pre_node_name = None
    count = 0
    for node in fx_model.graph.nodes:
        count += 1
        new_node = new_graph.node_copy(node, lambda x: env[x.name])
        env[node.name] = new_node
    new_graph.lint()
    return fx.GraphModule(model, fx_model.graph)

def dag_process(model: torch.nn.Module) -> torch.nn.Module:
    #model = copy.deepcopy(model)
    trace = CustomTracer()
    fx_trace = trace.trace(model)
    fx_model = fx.graph_module.GraphModule(trace.root, fx_trace)
    new_graph = fx.Graph()
    env = {}
    prepare_dict = {}
    first_gemm = False
    
    record_node = None
    for node in fx_model.graph.nodes:
        if (not first_gemm):
            if ('pad' in node.name) or ('input' in node.name):
                if ('input' in node.name):
                    record_node = node
                else:
                    new_args = ()
                    new_args = new_args + (record_node,)
                    node.args = new_args
                prepare_dict[node.name] = 1
            elif (('conv' in node.name) or ('gemm' in node.name) or ('matmul' in node.name)):
                prepare_dict[node.name] = 0
                first_gemm = True
            else:
                prepare_dict[node.name] = -1
        else:
            prepare_dict[node.name] = 1
    
    record_node = None
    for node in fx_model.graph.nodes:
        if ('input' in node.name) or ('pad' in node.name):
            record_node = node
        if prepare_dict[node.name] == -1:
            if ('permute' in node.name) or ('reshape' in node.name) or ('to' in node.name) :
                new_args = ()
                new_args = new_args + (record_node,)
                node.args = new_args
            elif ('mul' in node.name) or ('sub' in node.name) or ('add' in node.name) or ('div' in node.name):
                new_args = ()
                new_args = new_args + (record_node,)
                new_args = new_args + (record_node,)
                node.args = new_args
            elif ('getitem' in node.name) or ('resize' in node.name) or ('cat' in node.name):
                new_args = ()
                new_args = new_args + (record_node,)
                new_args = new_args + (record_node,)
                node.args = new_args
        elif prepare_dict[node.name]==0:
            new_args = ()
            need_new_args = False
            for p_node in node.args:
                if prepare_dict[p_node.name] == -1:
                    need_new_args = True
                    new_args = new_args + p_node.args
                else:
                    new_args = new_args + (p_node,)
            if need_new_args:
                node.args = new_args
        
    for node in fx_model.graph.nodes:
        if prepare_dict[node.name] in [0,1]:        
            new_node = new_graph.node_copy(node, lambda x: env[x.name])
            env[node.name] = new_node
        else:
            fx_model.graph.erase_node(node)
        '''if 'output' in node.name:
            print("sss")
            new_args = ()
            if isinstance(node.args[0],list):
                for i in range(0,len(node.args[0])):
                    print(i)
                    new_args = new_args + (node.args[0][i],)  
                node.args = new_args'''
        #if str(node.target) =='<built-in function mul>':
        #    node.target = operator.mul
    
    prepare_dict = None
    del prepare_dict
    print("DAG_Input_Processing!!")
    
    new_graph.lint()
    return fx.GraphModule(model, fx_model.graph)


def dag_process(model: torch.nn.Module) -> torch.nn.Module:
    #model = copy.deepcopy(model)
    trace = CustomTracer()
    fx_trace = trace.trace(model)
    fx_model = fx.graph_module.GraphModule(trace.root, fx_trace)
    new_graph = fx.Graph()
    delete_dict={}
    input_dict={}
    delete=True
    model_cut=False
    input_count=0

    """
    Only support one branch model
    """
    def node_in_args(node):
        node_list=[]
        for prev_node in node.args:
            if isinstance(prev_node,(tuple,list)):
                for ele in prev_node:
                    if isinstance(ele,fx.Node):
                        node_list.append(ele.name)
            elif isinstance(prev_node,fx.Node):
                node_list.append(prev_node.name)
            else:
                pass
        return node_list
    def nodename_node(name,model):
        for node in model.graph.nodes:
            if node.name==name:
                return node
        return "not found"
    for node in fx_model.graph.nodes:
        if node.op=="placeholder":
            input_count=input_count+1
            input_dict.update({node.name:[node.name]})

        if node.op=="call_module":
            delete=False
        if node.op=="call_function" and not node.name in ['mul','add','sub','div'] :
            delete=False
        if node.op=="get_attr":
            delete_dict.update({node.name:False})
        else:
            args_list=node_in_args(node)
            for n in args_list:
                for key in input_dict.keys():
                    if n in input_dict[key]:
                        input_dict[key].append(node.name)
            delete_dict.update({node.name:delete})
        if node.op=="call_method" and node.target=="permute" and delete:
            args_list=node_in_args(node)
            source_count=0
            for n in args_list:
                for key in input_dict.keys():
                    if n in input_dict[key]:
                        new_input_node=node.name
                        success_node={new_input_node:nodename_node(key,fx_model)}
                        source_count=source_count+1
            if source_count==1:
                delete=False
                model_cut=True
            elif source_count>1:
                print("not supoort preprocessing")
                return fx_model
    if model_cut and input_count==1:
        print("erase target node")
        for node in fx_model.graph.nodes:
            new_args=()
            for prev_node in node.args:
                if isinstance(prev_node,(tuple,list)):
                    new_args_sub=()
                    for ele in prev_node:
                        if isinstance(ele,fx.Node):
                            if ele.name==new_input_node:
                                new_args_sub=new_args_sub+(ele.args[0],)
                            else:
                                new_args_sub=new_args_sub+(ele,)
                        else:
                            new_args_sub=new_args_sub+(ele,)
                    new_args=new_args+(new_args_sub,)
                elif isinstance(prev_node,fx.Node):
                    if prev_node.name==new_input_node:
                        new_args=new_args+(success_node[prev_node.name],)
                    else:
                        new_args=new_args+(prev_node,)
                else:
                    new_args=new_args+(prev_node,)
            node.args=new_args
            if delete_dict[node.name] and node.op != 'placeholder':
                node.args=()
        for node in fx_model.graph.nodes:
            if delete_dict[node.name] and node.op != 'placeholder':
                fx_model.graph.erase_node(node)
    model=fx.GraphModule(model, fx_model.graph)
    trace = CustomTracer()
    fx_trace = trace.trace(model)
    fx_model = fx.graph_module.GraphModule(trace.root, fx_trace)
    return fx_model
"""
need another function to add conv and solve the concat issue
"""
#def add_conv():
#   print()
#   return 0
