# Copyright (C) 2020 Arm Limited or its affiliates. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Description:
# Main entry point for the Andes Tflite Compiler.
#
# Provides command line interface, options parsing, and network loading. Before calling the compiler driver.
import argparse
import os
from collections import OrderedDict
from glob import glob

from andesTfliteCompiler import model_reader
from andesTfliteCompiler import architecture_features
from andesTfliteCompiler import graph_optimiser
from andesTfliteCompiler import compiler_config
from andesTfliteCompiler import extract_npu_subgraphs
from andesTfliteCompiler import pass_packing
from andesTfliteCompiler import tile_draw
from andesTfliteCompiler import libnn_file_writer
from andesTfliteCompiler import tile_tool_utils
from andesTfliteCompiler import mark_tensors
from andesTfliteCompiler import scheduler
from andesTfliteCompiler import npu_serialisation
from andesTfliteCompiler import rewrite_npu_command_stream
from andesTfliteCompiler import tensor_allocation
from andesTfliteCompiler import dla_performance
from andesTfliteCompiler import tflite_generator

from andesTfliteCompiler.rewrite_graph import verify_graph_health
from andesTfliteCompiler.errors import InputFileError

import logging
logging.basicConfig(level=logging.CRITICAL, format='%(levelname)s: %(message)s')

MEMORY_START_OFFSET = 16

def find_model_file(directory):
    tflite_files = glob(os.path.join(directory, '*.tflite'))
    if not tflite_files:# no tfilit file in the directory
        return None
    model_file = next((f for f in tflite_files if f.endswith('model.tflite')), None)
    if model_file is None: # it can't find any 'model.tflite' file
        if len(tflite_files) > 1: # it find more than one tflite file
            file_names = ', '.join(os.path.basename(f) for f in tflite_files)
            raise ValueError(f"Multiple .tflite files were found: {file_names}. Please remove or rename the .tflite file you want to use to 'model.tflite'.")
        return tflite_files[0] # only one tflite file in the directory ex: 'rn50.tflite'
    return model_file

def process(input_name, model_reader_options, compiler_options, arch_options, args):

    nng = model_reader.read_model(input_name, model_reader_options, arch_options)
    assert verify_graph_health(nng)
    nng = graph_optimiser.optimise_graph(nng, arch_options, verbose_graph=False)
    assert verify_graph_health(nng)
    nng.compute_best_peak_memory_usage(include_weight=compiler_options.include_weight, start_from=arch_options.MEMORY_START_OFFSET)
    nng = mark_tensors.mark_tensor_purpose(nng, arch_options, verbose_tensor_purpose=False)
    assert verify_graph_health(nng)
    pass_packing.pack_into_passes(nng, arch_options, verbose_packing=False)
    assert verify_graph_health(nng)
    extract_npu_subgraphs.extract_npu_subgraphs(nng, arch_options)
    assert verify_graph_health(nng)

    if not nng:
        raise InputFileError(input_name, "Input file could not be read")
    
    os.makedirs(arch_options.quantize_config.output_folder, exist_ok=True)

    scheduler_options = scheduler.SchedulerOptions(
        optimization_strategy="size",
        sram_target=arch_options.arena_cache_size,
        verbose_schedule=False
    )
    scheduler.schedule_passes(nng, arch_options, compiler_options, scheduler_options)

    tensor_allocation.execute_tensor_allocation(nng, arch_options, compiler_options, scheduler_options)
    alloc_draw = tile_draw.alloc_draw(arch_options.quantize_config.output_folder + "/allocation_information.txt", 
                                        arch_options.quantize_config.output_folder + "/feature_maps_memory_allocation.png",
                                        arch_options.quantize_config.output_folder + "/weights_memory_allocation.png")

    # Placeholders for scratch and flash tensors that are common for all Npu subgraphs
    scratch_tens_map = OrderedDict()
    flash_tens_map = OrderedDict()

    root_sg = nng.get_root_subgraph()
    scratch_tens_map = npu_serialisation.serialise_root_subgraph_scratch_tensors(
        root_sg, arch_options, scratch_tens_map
    )

    output_basename = os.path.join(arch_options.quantize_config.output_folder, os.path.splitext(os.path.basename(input_name))[0])
    report_filename = output_basename + "_andla_performance.xlsx"

    if arch_options.quantize_config.cpu_only == False:
        dla_performance.calc_performance_for_network(nng, arch_options, log_report=report_filename, print_verbose=True)

    for sg in nng.subgraphs:
        scratch_tens_map, flash_tens_map = npu_serialisation.serialise_npu_subgraph_into_tensors(
            sg, arch_options, scratch_tens_map, flash_tens_map
        )
    root_sg.flash_tensor_map = flash_tens_map
    rewrite_npu_command_stream.rewrite_npu(nng, arch_options)

    if arch_options.quantize_config.cpu_only == False:
        output_filename = output_basename + "_andla.tflite"
    else:
        output_filename = output_basename + "_nds.tflite"

    if not input_name.endswith("_nds.tflite"):
        tflite_generator.write_tflite(nng, output_filename, arch_options)

    if not arch_options.quantize_config.runtime in ["TFL", "TFLM"]:
        libnn_file_writer.record_para_bin(nng, arch_options, output_basename + '_para_bin/')
        libnn_file_writer.record_sfile(nng, arch_options, output_basename + "_para.S", output_basename + '_para_bin/')
        libnn_file_writer.record_headerfile(nng, arch_options, output_basename + ".h", compiler_options.cpu_tensor_alignment)
        libnn_file_writer.record_cfile(nng, arch_options, output_basename + ".c", output_basename + ".h", args)

    return nng


def initialize_options(args):
    model_reader_options = model_reader.ModelReaderOptions()
    compiler_options = compiler_config.CompilerOptions(
        include_weight=args.include_weight,
        tensor_allocator=None,
        verbose_allocation=False,
        cpu_tensor_alignment=16
    )
    base_address = tile_tool_utils.BaseAddress(args.base_address_index)
    quantize_config_path = args.quantize_config if args.model_pkg is None else os.path.join(args.model_pkg, "quantize_config.yaml")
    quantize_config = architecture_features.Quantize_config(quantize_config_path)
    quantize_config.add_config('cpu_only', args.cpu_only)
    quantize_config.add_config('x86', args.x86)
    quantize_config.add_config('debug_enable', args.debug_enable)
    quantize_config.add_config('working_folder', os.path.dirname(quantize_config_path))
    if args.model_pkg is not None and os.path.isdir(args.model_pkg):
        model_file = find_model_file(args.model_pkg)
        if model_file is None or not os.path.isfile(quantize_config_path):
            parser.error("The specified model package directory does not contain both 'quantize_config.yaml' and a '.tflite' file. Please verify the directory and try again.")
        args.output_dir = os.path.join(args.model_pkg, args.output_dir.lstrip('./'))
    else:
        if os.path.dirname(args.quantize_config) == os.path.dirname(args.model):
            args.output_dir = os.path.join(os.path.dirname(args.model), args.output_dir)
    quantize_config.add_config('output_folder', args.output_dir)
    quantize_config.add_config('operator_option', args.operator_option)
    quantize_config.add_config('runtime', args.runtime)
    quantize_config.add_config('base_address', base_address)
    quantize_config.add_config('channel_extension_op', args.ch_extend_op)
    quantize_config.add_config('single_op_subgraph', args.single_op_subgraph)

    arch_options = architecture_features.ArchitectureFeatures(
        main_compiler_config_files=args.config,
        system_config=args.system_config,
        memory_mode=args.memory_mode,
        cpu_config=args.cpu_config,
        verbose_config=False,
        arena_cache_size=None,
        quantize_config=quantize_config,
        memory_start_offset=MEMORY_START_OFFSET,
    )

    return model_reader_options, compiler_options, arch_options

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Neural network model compiler for AnDLA")
    # set network nargs to be optional to allow the support-ops-report CLI option to be used standalone
    parser.add_argument(
        "--model", type=str, default=None, nargs="?",
        help="Filename of the input TensorFlow Lite for Microcontrollers model",
    )
    parser.add_argument(
        "--output-dir", type=str, default="output", help="Output directory to write files to (default: %(default)s)"
    )
    parser.add_argument("-q", "--quantize-config", 
                        type=str, 
                        default="./andesTfliteCompiler/quantize_config.yaml", 
                        help="Quantize config generate by AITools"
    )
    parser.add_argument("--config",
                        type=str,
                        default="./andesTfliteCompiler/main_compiler.yaml", 
                        help ="System configuration file(s) in Python ConfigParser .yaml file format",
    )
    parser.add_argument("--system-config",
                        type=str,
                        default="AnDLA_RVP_Embedded",
                        help="System configuration to select from the configuration file"
    )
    parser.add_argument("--memory-mode",
                        type=str,
                        default="Dedicated_Sram",
                        help="Memory mode to select from the configuration file"
    )
    parser.add_argument("--cpu-config",
                        type=str,
                        default="RVP",
                        choices=["RVP", "RVV"],
                        help="CPU configuration",
    )
    parser.add_argument("--cpu-only",
                        action="store_true",
                        help="Option for libnn only (, which means without AnDLA,) output",
    )
    parser.add_argument("-o", "--operator-option",
                        type=str,
                        default="conv:general",
                        choices=["conv:deafult", "conv:general"],
                        help="Certain option of operator can be chosed",
    )
    parser.add_argument("--libnn-version",
                        type=str, 
                        default="./andesTfliteCompiler/nds_version.c", 
                        help="Quantize config generate by AITools"
    )
    parser.add_argument("-r", "--runtime",
                        type=str,
                        default="TFLM",
                        choices=["TFL", "TFLM", "CBACKEND"],
                        help="Runtime of tensorflow lite or tensorflow lite for microcontroller",
    )
    parser.add_argument("-b", "--base-address-index",
                        type=int,
                        nargs=8,
                        default=[1, 1, 2, 2, 2, 2, 3, 4],
                        metavar=('IFM', 'OFM', 'Weight', 'Bias', 'LUT', 'Scale_and_Shift', 'ex_IFM', 'ex_OFM'),
                        help="Base address index mapping of each type of tensor" 
                        + ", (default: [1, 1, 2, 2, 2, 2, 3, 4])",
    )

    parser.add_argument("--gen-package", action="store_true", help=" Generate NN_Demo")
    parser.add_argument("--debug-enable", action="store_true", help="Enable debug mode")
    parser.add_argument("--x86", action="store_true", help="generate x86 purec c backend")
    parser.add_argument("--include-weight", action="store_true", help=" Tenser allocation for weight and bias tensors")
    parser.add_argument("--ch-extend-op",  
                        action="store_true",
                        help="Option for insert channel extension inform op",
    )

    parser.add_argument("--single-op-subgraph", action="store_true", help="Make the optput andla tflite do single op in one custom op")

    parser.add_argument("--model-pkg", type=str, default=None, help="specify the model_package path device")
    args = parser.parse_args()
    if args.x86:
        if not args.cpu_only:
            parser.error("When targeting x86 C backend, 'cpu_only' must be set to True.")
        if args.include_weight:
            parser.error("When targeting x86 C backend, 'include_weight' should be set to False.")

    if args.model_pkg is None or not os.path.isdir(args.model_pkg):

        if args.model is None:
            parser.error("Error: model or model package is required")

        model_reader_options, compiler_options, arch_options = initialize_options(args)
        nng = process(args.model, model_reader_options, compiler_options, arch_options, args)
    else:
        quantize_config_path = os.path.join(args.model_pkg, "quantize_config.yaml")
        model_file = find_model_file(args.model_pkg)
        if model_file is None or not os.path.isfile(quantize_config_path):
            raise ValueError("Model or config missing in provided package directory.")
        model_reader_options, compiler_options, arch_options = initialize_options(args)
        nng = process(model_file, model_reader_options, compiler_options, arch_options, args)
    print("Compilation successful!")
