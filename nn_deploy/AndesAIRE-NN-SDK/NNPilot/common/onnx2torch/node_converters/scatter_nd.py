__all__ = [
    'OnnxScatterND','OnnxScatter1D'
]

import torch
import torch._C as torch_C
from torch import nn

from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OperationConverterResult
from onnx2torch.utils.common import onnx_mapping_from_node, get_shape_from_value_info
from onnx2torch.utils.custom_export_to_onnx import CustomExportToOnnx
from onnx2torch.utils.custom_export_to_onnx import OnnxToTorchModuleWithCustomExport


class OnnxScatterND(nn.Module, OnnxToTorchModuleWithCustomExport):  # pylint: disable=missing-class-docstring
    def forward(  # pylint: disable=missing-function-docstring
        self,
        data: torch.Tensor,
        indices: torch.Tensor,
        updates: torch.Tensor,
    ) -> torch.Tensor:
        # There is no scatter nd for torch, use following formula:
        # https://github.com/onnx/onnx/blob/master/docs/Operators.md#ScatterND
        output = data.clone()

        if torch.onnx.is_in_onnx_export():
            return _ScatterNDExportToOnnx.set_output_and_apply(output, data, indices, updates)

        ind_dim = indices.dim()
        # last dimension is a partial-index into data
        indices = indices.reshape((-1, indices.shape[-1])).T.tolist()
        # update.shape = indices.shape[0:ind_dim-1] ++ data.shape[indices.shape[-1]:data.dim()-1]
        updates = updates.reshape((-1, *updates.shape[ind_dim - 1 :]))
        output[indices] = updates

        return output

class OnnxScatter1D(nn.Module, OnnxToTorchModuleWithCustomExport):  # pylint: disable=missing-class-docstring
    def forward(  # pylint: disable=missing-function-docstring
        self,
        data: torch.Tensor,
        indices: torch.Tensor,
        updates: torch.Tensor,
    ) -> torch.Tensor:
        # There is no scatter nd for torch, use following formula:
        # https://github.com/onnx/onnx/blob/master/docs/Operators.md#ScatterND
        output = updates
        return output

class _ScatterNDExportToOnnx(CustomExportToOnnx):  # pylint: disable=abstract-method
    @staticmethod
    def symbolic(graph: torch_C.Graph, *args) -> torch_C.Value:
        return graph.op('ScatterND', *args, outputs=1)


@add_converter(operation_type='ScatterND', version=11)
@add_converter(operation_type='ScatterND', version=13)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    for idx,target in enumerate(node._input_values):
        if target in graph._value_info:
            input_value_info = graph.value_info[node._input_values[idx]]
    input_shape = get_shape_from_value_info(input_value_info)
    out_value_info = graph.value_info[node._output_values[0]]
    output_shape = get_shape_from_value_info(out_value_info)
    if input_shape==output_shape:
        return OperationConverterResult(
        torch_module=OnnxScatter1D(),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )
    return OperationConverterResult(
        torch_module=OnnxScatterND(),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )
