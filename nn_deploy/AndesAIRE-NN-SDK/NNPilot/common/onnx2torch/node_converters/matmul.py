__all__ = [
    'OnnxMatMul',
]

import torch
from torch import nn

from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OnnxMapping
from onnx2torch.utils.common import OnnxToTorchModule
from onnx2torch.utils.common import OperationConverterResult
from onnx2torch.utils.common import onnx_mapping_from_node


class OnnxMatMul(nn.Module, OnnxToTorchModule):  # pylint: disable=missing-class-docstring
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # pylint: disable=missing-function-docstring
        return torch.matmul(x, y)


@add_converter(operation_type='MatMul', version=1)
@add_converter(operation_type='MatMul', version=9)
@add_converter(operation_type='MatMul', version=13)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    input_name = node.input_values[0]
    weight_name = node.input_values[1]
    #print(node.input_values)
    #print(node.attributes)
    if weight_name in graph.initializers:
        weights = graph.initializers[weight_name]
        weights = weights.to_torch()
        weights = weights.T
        
        in_features, out_features = weights.shape[1], weights.shape[0]
        torch_module = nn.Linear(
            in_features=in_features,
            out_features=out_features,
            bias= None,
        )
        with torch.no_grad():
            torch_module.weight.data = weights
        
        return OperationConverterResult(
                torch_module=torch_module,
                onnx_mapping=OnnxMapping(
                    inputs=(input_name,),
                    outputs=node.output_values,
                ),
            )
    
    return OperationConverterResult(
        torch_module=OnnxMatMul(),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )
