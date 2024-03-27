__all__ = [
    'OnnxBinaryMathOperation_DI',
    'OnnxBinaryMathOperation_SI',
]

from typing import Optional

import torch
from torch import nn
import operator

from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OnnxToTorchModule
from onnx2torch.utils.common import OperationConverterResult
from onnx2torch.utils.common import old_style_broadcast
from onnx2torch.utils.common import onnx_mapping_from_node
from onnx2torch.utils.common import OnnxMapping

_TORCH_FUNCTION_FROM_ONNX_TYPE = {
    'Add': operator.add,
    'Sub': operator.sub,
    'Mul': operator.mul,
    'Div': operator.truediv,
}

class OnnxBinaryMathOperation_DI(nn.Module, OnnxToTorchModule):  # pylint: disable=missing-docstring
    def __init__(self, operation_type: str, broadcast: Optional[int] = None, axis: Optional[int] = None):
        super().__init__()

        self.broadcast = broadcast
        self.axis = axis
        self.math_op_function = _TORCH_FUNCTION_FROM_ONNX_TYPE[operation_type]

    def forward(  # pylint: disable=missing-function-docstring
        self,
        first: torch.Tensor,
        second: torch.Tensor,
    ) -> torch.Tensor:
        if self.broadcast == 1 and self.axis is not None:
            second = old_style_broadcast(first, second, self.axis)
        #a = self.math_op_function(first, second)
        #if isinstance(a, torch.Tensor):
        #    print(a.size())
        return self.math_op_function(first, second)

class OnnxBinaryMathOperation_SI(nn.Module, OnnxToTorchModule):  # pylint: disable=missing-docstring
    def __init__(self, operation_type: str, broadcast: Optional[int] = None, axis: Optional[int] = None):
        super().__init__()

        self.broadcast = broadcast
        self.axis = axis
        self.math_op_function = _TORCH_FUNCTION_FROM_ONNX_TYPE[operation_type]
        self.value = 0

    def forward(  # pylint: disable=missing-function-docstring
        self,
        first: torch.Tensor,
    ) -> torch.Tensor:
        return self.math_op_function(first, self.value.clone().detach())
        


@add_converter(operation_type='Add', version=1)
@add_converter(operation_type='Add', version=6)
@add_converter(operation_type='Add', version=7)
@add_converter(operation_type='Add', version=13)
@add_converter(operation_type='Add', version=14)
@add_converter(operation_type='Sub', version=1)
@add_converter(operation_type='Sub', version=6)
@add_converter(operation_type='Sub', version=7)
@add_converter(operation_type='Sub', version=13)
@add_converter(operation_type='Sub', version=14)
@add_converter(operation_type='Mul', version=1)
@add_converter(operation_type='Mul', version=6)
@add_converter(operation_type='Mul', version=7)
@add_converter(operation_type='Mul', version=13)
@add_converter(operation_type='Mul', version=14)
@add_converter(operation_type='Div', version=1)
@add_converter(operation_type='Div', version=6)
@add_converter(operation_type='Div', version=7)
@add_converter(operation_type='Div', version=13)
@add_converter(operation_type='Div', version=14)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    if node._input_values[1] in graph.initializers:
        constant_ = graph.initializers[node._input_values[1]]
        constant_ = constant_.to_torch()
        
        torch_module=OnnxBinaryMathOperation_SI(
                operation_type=node.operation_type,
                broadcast=node.attributes.get('broadcast', None),
                axis=node.attributes.get('axis', None),
                )
        torch_module.value = constant_
        
        return OperationConverterResult(
                torch_module=torch_module,
                onnx_mapping=OnnxMapping(
                    inputs=(node._input_values[0],),
                    outputs=node.output_values,
                ),
            )
    else:
        return OperationConverterResult(
            torch_module=OnnxBinaryMathOperation_DI(
                operation_type=node.operation_type,
                broadcast=node.attributes.get('broadcast', None),
                axis=node.attributes.get('axis', None),
            ),
            onnx_mapping=onnx_mapping_from_node(node=node),
        )
