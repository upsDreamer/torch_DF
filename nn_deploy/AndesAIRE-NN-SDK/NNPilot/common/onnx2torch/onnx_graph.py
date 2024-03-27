from collections import OrderedDict
from enum import Enum
from types import MappingProxyType
from typing import Mapping
from typing import Tuple

from onnx.onnx_ml_pb2 import GraphProto
from onnx.onnx_ml_pb2 import ValueInfoProto

from onnx2torch.onnx_node import OnnxNode
from onnx2torch.onnx_tensor import OnnxTensor


class ValueType(Enum):  # pylint: disable=missing-class-docstring
    GRAPH_INPUT = 0
    NODE_OUTPUT = 1
    GRAPH_INITIALIZER = 2
    UNKNOWN = 3
    EMPTY = 4


class OnnxGraph:  # pylint: disable=missing-class-docstring
    def __init__(self, onnx_graph_proto: GraphProto):
        #print(type(onnx_graph_proto))
        self._proto = onnx_graph_proto
        self._input_values = tuple(value_info.name for value_info in self._proto.input)
        self._output_values = tuple(value_info.name for value_info in self._proto.output)

        unique_names = []
        counters = {}
        for node in onnx_graph_proto.node:
            name = f'{node.domain}_{node.op_type}'.lstrip('_')
            name_counter = counters.setdefault(name, 0)
            counters[name] += 1
            unique_names.append(f'{name}_{name_counter}')

        self._nodes = OrderedDict(
            (name, OnnxNode(node, unique_name=name)) for name, node in zip(unique_names, onnx_graph_proto.node)
        )
        self._initializers = {initializer.name: OnnxTensor(initializer) for initializer in onnx_graph_proto.initializer}
        self._node_output_values = {
            output_name: (node, i) for node in self._nodes.values() for i, output_name in enumerate(node.output_values)
        }
        self._value_info = {value_info.name: value_info for value_info in onnx_graph_proto.value_info}
        for input_value_info in onnx_graph_proto.input:
            self._value_info[input_value_info.name] = input_value_info
        for output_value_info in onnx_graph_proto.output:
            self._value_info[output_value_info.name] = output_value_info
        
        delete_name_list = []
        delete_name_list_output = []
        # for Upsampling using
        for key_Resize in self._nodes:
            if 'Resize' in self._nodes[key_Resize].unique_name:
                previous_concat_name = None
                for i in range(0,len(self._nodes[key_Resize]._input_values)):
                    if 'Concat' in self._nodes[key_Resize]._input_values[i]:
                        previous_concat_name = self._nodes[key_Resize]._input_values[i]
                previous_concat = None
                for key_Concat in self._nodes:
                    if self._nodes[key_Concat].name in previous_concat_name:
                        previous_concat = self._nodes[key_Concat]
                        previous_slice_name = None
                        for i in range(0,len(self._nodes[key_Concat]._input_values)):
                            if 'Slice' in self._nodes[key_Concat]._input_values[i]:
                                previous_slice_name = self._nodes[key_Concat]._input_values[i]
                        previous_slice = None
                        for key_Slice in self._nodes:
                            if self._nodes[key_Slice].name in previous_slice_name:
                                delete_name_list_output.append(previous_slice_name)
                                previous_slice = self._nodes[key_Slice]
                                ini_list_C = []
                                ini_list_S = []
                                for input_ini_C in range(0,len(previous_concat._input_values)):
                                    if previous_concat._input_values[input_ini_C] in self.initializers:
                                        ini_list_C.append(previous_concat._input_values[input_ini_C])
                                for input_ini_S in range(0,len(previous_slice._input_values)):
                                    if previous_slice._input_values[input_ini_S] in self.initializers:
                                        ini_list_S.append(previous_slice._input_values[input_ini_S])
                                if len(ini_list_C) != 0:
                                    self._nodes[key_Resize]._input_values = self._nodes[key_Resize]._input_values + (ini_list_C[0],)
                                self._nodes[key_Resize]._input_values = self._nodes[key_Resize]._input_values + (ini_list_S[0],ini_list_S[1],)
                                if len(ini_list_C) != 0:
                                    self._nodes[key_Resize]._input_values = (self._nodes[key_Resize]._input_values[0],self._nodes[key_Resize]._input_values[4],)
                                    self._nodes[key_Concat]._input_values = ('input_1','input_1',)
                                    delete_name_list.append(key_Concat)
                                else:
                                    new_input = ()
                                    for i in range(0,len(self._nodes[key_Concat]._input_values)):
                                        if self._nodes[key_Concat]._input_values[i] != previous_slice_name:
                                            new_input = new_input + (self._nodes[key_Concat]._input_values[i],)
                                    self._nodes[key_Concat]._input_values = new_input
                                    self._nodes[key_Resize]._input_values = (self._nodes[key_Resize]._input_values[0],self._nodes[key_Resize]._input_values[4],self._nodes[key_Resize]._input_values[5],)
                                self._nodes[key_Slice]._input_values = ('input_1','input_1','input_1',)
                                delete_name_list.append(key_Slice)
                                ini_list_C = []
                                ini_list_S = []    
        for i_list in range(0,len(delete_name_list)):
            del self._nodes[delete_name_list[i_list]]
        for i_list in range(0,len(delete_name_list_output)):
            del self._node_output_values[delete_name_list_output[i_list]]
        i_list = None
        delete_name_list = None
        delete_name_list_output = None
        del i_list, delete_name_list, delete_name_list_output
    
    @property
    def proto(self) -> GraphProto:  # pylint: disable=missing-function-docstring
        return self._proto

    @property
    def value_info(self) -> Mapping[str, ValueInfoProto]:  # pylint: disable=missing-function-docstring
        return self._value_info

    @property
    def name(self) -> str:  # pylint: disable=missing-function-docstring
        return self._proto.name

    @property
    def input_values(self) -> Tuple[str, ...]:  # pylint: disable=missing-function-docstring
        return self._input_values

    @property
    def output_values(self) -> Tuple[str, ...]:  # pylint: disable=missing-function-docstring
        return self._output_values

    @property
    def nodes(self) -> Mapping[str, OnnxNode]:  # pylint: disable=missing-function-docstring
        return self._nodes

    @property
    def initializers(self) -> Mapping[str, OnnxTensor]:  # pylint: disable=missing-function-docstring
        return MappingProxyType(self._initializers)

    def value_type(self, value_name: str) -> ValueType:  # pylint: disable=missing-function-docstring
        if value_name in self._input_values:
            return ValueType.GRAPH_INPUT

        if value_name in self._node_output_values:
            return ValueType.NODE_OUTPUT

        if value_name in self._initializers:
            return ValueType.GRAPH_INITIALIZER

        if value_name == '':
            return ValueType.EMPTY

        return ValueType.UNKNOWN

    def value_as_node_output(  # pylint: disable=missing-function-docstring
        self,
        value_name: str,
    ) -> Tuple[OnnxNode, int]:
        return self._node_output_values[value_name]
