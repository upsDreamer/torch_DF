__all__ = [
    'OnnxGather',
    'OnnxGatherElements',
]

from typing import List
from typing import Tuple
from typing import Union

import torch
import torch._C as torch_C
from torch import nn

from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OnnxToTorchModule
from onnx2torch.utils.common import OperationConverterResult
from onnx2torch.utils.common import onnx_mapping_from_node,get_const_value
from onnx2torch.utils.custom_export_to_onnx import CustomExportToOnnx
from onnx2torch.utils.custom_export_to_onnx import OnnxToTorchModuleWithCustomExport


class OnnxGatherElements(nn.Module, OnnxToTorchModule):  # pylint: disable=missing-docstring
    def __init__(self, axis: int = 0):
        super().__init__()
        self.axis = axis

    def forward(  # pylint: disable=missing-function-docstring
        self,
        input_tensor: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        return torch.gather(input_tensor, dim=self.axis, index=indices)


class OnnxGather(nn.Module, OnnxToTorchModuleWithCustomExport):
    """ONNX gather implementation (or numpy.take implementation)"""

    def __init__(self, axis: int = 0):
        super().__init__()
        self.axis = axis
        self.leaf_m_register=True

    @staticmethod
    def slice_from_axis(  # pylint: disable=missing-docstring
        input_tensor: torch.Tensor,
        axis: int,
        indices: torch.Tensor,
    ) -> Tuple[Union[slice, torch.Tensor], ...]:
        axis = input_tensor.dim() + axis if axis < 0 else axis
        skip_axis: List[Union[slice, torch.Tensor]] = [slice(None)] * axis
        skip_axis.append(indices)
        return tuple(skip_axis)

    def forward(  # pylint: disable=missing-function-docstring
        self, input_tensor: torch.Tensor, indices=None
    ) -> torch.Tensor:
        # pytorch Gather differs from onnx Gather, onnx gather work like numpy.take
        # But torch.take does not support different axis. So we make it by yourself
        # numpy.take is input_data[:, :, indices] where we pass NONE slices AXIS time
        slice_for_take = self.slice_from_axis(input_tensor, self.axis, indices)
        output = input_tensor[slice_for_take]
        if torch.onnx.is_in_onnx_export():
            return _GatherExportToOnnx.set_output_and_apply(output, input_tensor, indices, self.axis)

        return output

class AndesGather(nn.Module):
    """ONNX gather implementation (or numpy.take implementation)"""

    def __init__(self, axis: int = 0,indices=0):
        super().__init__()
        self.axis = axis
        self.indices=indices
        self.leaf_m_register=True

    @staticmethod
    def slice_from_axis(  # pylint: disable=missing-docstring
        input_tensor: torch.Tensor,
        axis: int,
        indices: torch.Tensor,
    ) -> Tuple[Union[slice, torch.Tensor], ...]:
        axis = input_tensor.dim() + axis if axis < 0 else axis
        skip_axis: List[Union[slice, torch.Tensor]] = [slice(None)] * axis
        skip_axis.append(indices)
        return tuple(skip_axis)

    def forward(  # pylint: disable=missing-function-docstring
            self, input_tensor: torch.Tensor
    ) -> torch.Tensor:
        # pytorch Gather differs from onnx Gather, onnx gather work like numpy.take
        # But torch.take does not support different axis. So we make it by yourself
        # numpy.take is input_data[:, :, indices] where we pass NONE slices AXIS time
        slice_for_take = self.slice_from_axis(input_tensor, self.axis, self.indices)
        output = input_tensor[slice_for_take]
        if torch.onnx.is_in_onnx_export():
            return _GatherExportToOnnx.set_output_and_apply(output, input_tensor, self.indices, self.axis)

        return output
    def extra_repr(self) -> str:
        return f'axis={self.axis}, indices={self.indices}'

class OnnxGather(nn.Module, OnnxToTorchModuleWithCustomExport):
    """ONNX gather implementation (or numpy.take implementation)"""

    def __init__(self, axis: int = 0,indices=0):
        super().__init__()
        self.axis = axis
        self.indices=indices
        self.interm=AndesGather(axis,indices)

    @staticmethod
    def slice_from_axis(  # pylint: disable=missing-docstring
        input_tensor: torch.Tensor,
        axis: int,
        indices: torch.Tensor,
    ) -> Tuple[Union[slice, torch.Tensor], ...]:
        axis = input_tensor.dim() + axis if axis < 0 else axis
        skip_axis: List[Union[slice, torch.Tensor]] = [slice(None)] * axis
        skip_axis.append(indices)
        return tuple(skip_axis)

    def forward(  # pylint: disable=missing-function-docstring
            self, input_tensor: torch.Tensor,indices: torch.Tensor
    ) -> torch.Tensor:
        # pytorch Gather differs from onnx Gather, onnx gather work like numpy.take
        # But torch.take does not support different axis. So we make it by yourself
        # numpy.take is input_data[:, :, indices] where we pass NONE slices AXIS time
        output=self.interm(input_tensor)
        if torch.onnx.is_in_onnx_export():
            return _GatherExportToOnnx.set_output_and_apply(output, input_tensor, self.indices, self.axis)

        return output

class _GatherExportToOnnx(CustomExportToOnnx):  # pylint: disable=abstract-method
    @staticmethod
    def symbolic(graph: torch_C.Graph, *args) -> torch_C.Value:
        input_tensor, indices, axis = args
        return graph.op('Gather', input_tensor, indices, axis_i=axis, outputs=1)


@add_converter(operation_type='Gather', version=1)
@add_converter(operation_type='Gather', version=11)
@add_converter(operation_type='Gather', version=13)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    axis = node.attributes.get('axis', 0)
    print(graph)
    print(node._input_values)
    indices=get_const_value(node._input_values[1],graph)
    torch_module = OnnxGather(
        axis=axis,
        indices=indices
    )

    return OperationConverterResult(
        torch_module=torch_module,
        onnx_mapping=onnx_mapping_from_node(node=node),
    )


@add_converter(operation_type='GatherElements', version=11)
@add_converter(operation_type='GatherElements', version=13)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    axis = node.attributes.get('axis', 0)
    indices = node.attributes.get('axis', 0)
    torch_module = OnnxGatherElements(
        axis=axis,
    )

    return OperationConverterResult(
        torch_module=torch_module,
        onnx_mapping=onnx_mapping_from_node(node=node),
    )
