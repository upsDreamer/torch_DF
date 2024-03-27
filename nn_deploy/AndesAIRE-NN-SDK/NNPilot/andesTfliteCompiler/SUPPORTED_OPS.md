# Supported Ops
backend converter version: `0.1`

Summary table of constraints for:
- [TFLite](#tflite-summary-table)

## TFLite Summary Table

The table below contains TFLite operators that can be placed on the Ethos-U NPU.  
If the constraints are not met, then that operator will be scheduled on the CPU instead.  
For any other TFLite operator not listed, will be left untouched and scheduled on the CPU.  
Please check the supported operator list for your chosen runtime for further information.

| Operator | TFLite Constraints |
| --- | --- |
| ADD | [Generic](#tflite-generic-constraints), [Specific](#tflite-add-constraints) |
| AVERAGE_POOL_2D | [Generic](#tflite-generic-constraints), [Specific](#tflite-average_pool_2d-constraints) |
| CONCATENATION | [Generic](#tflite-generic-constraints) |
| CONV_2D | [Generic](#tflite-generic-constraints), [Specific](#tflite-conv_2d-constraints) |
| DEPTHWISE_CONV_2D | [Generic](#tflite-generic-constraints), [Specific](#tflite-depthwise_conv_2d-constraints) |
| FULLY_CONNECTED | [Generic](#tflite-generic-constraints), [Specific](#tflite-fully_connected-constraints) |
| LOGISTIC | [Generic](#tflite-generic-constraints) |
| MAX_POOL_2D | [Generic](#tflite-generic-constraints), [Specific](#tflite-max_pool_2d-constraints) |
| MUL | [Generic](#tflite-generic-constraints), [Specific](#tflite-mul-constraints) |
| PACK | [Generic](#tflite-generic-constraints) |
| PAD | [Generic](#tflite-generic-constraints), [Specific](#tflite-pad-constraints) |
| RELU | [Generic](#tflite-generic-constraints) |
| RELU6 | [Generic](#tflite-generic-constraints) |
| RELU_N1_TO_1 | [Generic](#tflite-generic-constraints) |
| RESHAPE | [Generic](#tflite-generic-constraints), [Specific](#tflite-reshape-constraints) |
| RESIZE_BILINEAR | [Generic](#tflite-generic-constraints), [Specific](#tflite-resize_bilinear-constraints) |
| SOFTMAX | [Generic](#tflite-generic-constraints) |
| SUB | [Generic](#tflite-generic-constraints), [Specific](#tflite-sub-constraints) |
| TANH | [Generic](#tflite-generic-constraints) |

### TFLite Generic Constraints

This is a list of constraints that all NPU operators must satisfy in order to be scheduled on the NPU.

- Tensors must be of type: int8
- Tensor dimensions must be in the range [1, 65535]
- The fused activation function (if present) must be one of type: LOGISTIC, RELU, RELU6, RELU_N1_TO_1, TANH
- If a fused activation function is present, the Output tensor must be one of type: int8

### TFLite ADD Constraints

This is a list of constraints that the ADD operator must satisfy in order to be scheduled on the NPU.

- Broadcasting is only allowed for rank indices with dimension 1, from either IFM1 or IFM2

### TFLite AVERAGE_POOL_2D Constraints

This is a list of constraints that the AVERAGE_POOL_2D operator must satisfy in order to be scheduled on the NPU.

- IFM Tensor batch size must be 1
- Stride values for both width and height must be in the range [1, 3]
- Kernel filter values for both width and height must be in the range [1, 8]
- VALID padding: Kernel filter height must be in the range [1, 256]
- VALID padding: Product of kernel filter width and height must be in the range [1, 65536]

### TFLite CONV_2D Constraints

This is a list of constraints that the CONV_2D operator must satisfy in order to be scheduled on the NPU.

- Stride values for both width and height must be in the range [1, 3]
- Dilation factor values for both width and height must be in the range [1, 2]
- Dilated kernel height must be in the range [1, 64]
- Product of dilated kernel width and height must be in the range [1, 4096]
- Weight tensor must be 8-bit
- Weight tensor must be constant
- The sum of the weights cannot exceed 8323072
- Optional Bias tensor must be of type: int32
- IFM Tensor batch size must be 1

### TFLite DEPTHWISE_CONV_2D Constraints

This is a list of constraints that the DEPTHWISE_CONV_2D operator must satisfy in order to be scheduled on the NPU.

- Stride values for both width and height must be in the range [1, 3]
- Dilation factor values for both width and height must be in the range [1, 2]
- Dilated kernel height must be in the range [1, 64]
- Product of dilated kernel width and height must be in the range [1, 4096]
- Weight tensor must be 8-bit
- Weight tensor must be constant
- The sum of the weights cannot exceed 8323072
- Optional Bias tensor must be of type: int32
- IFM Tensor batch size must be 1
- For depth multipliers > 1, IFM channels must be 1 and OFM channels must be equal to the depth multiplier

### TFLite FULLY_CONNECTED Constraints

This is a list of constraints that the FULLY_CONNECTED operator must satisfy in order to be scheduled on the NPU.

- Weight tensor must be 8-bit
- Weight tensor must be constant
- Optional Bias tensor must be of type: int32

### TFLite MAX_POOL_2D Constraints

This is a list of constraints that the MAX_POOL_2D operator must satisfy in order to be scheduled on the NPU.

- IFM Tensor batch size must be 1
- Stride values for both width and height must be in the range [1, 3]
- Kernel filter height must be in the range [1, 256]
- Product of kernel filter width and height must be in the range [1, 65536]

### TFLite MUL Constraints

This is a list of constraints that the MUL operator must satisfy in order to be scheduled on the NPU.

- Broadcasting is only allowed for rank indices with dimension 1, from either IFM1 or IFM2

### TFLite PAD Constraints

This is a list of constraints that the PAD operator must satisfy in order to be scheduled on the NPU.

- The padding tensor must have the shape [3,2] or [4,2]
- The pad tensor can only pad width and height
- Pad tensor must be of type: int32

### TFLite RESHAPE Constraints

This is a list of constraints that the RESHAPE operator must satisfy in order to be scheduled on the NPU.

- Shape must be constant

### TFLite RESIZE_BILINEAR Constraints

This is a list of constraints that the RESIZE_BILINEAR operator must satisfy in order to be scheduled on the NPU.

- The width and height of the IFM and OFM must match one of the following criteria:  
        IFM W and H must both be 1  
        IFM must match OFM  
        OFM W and H must be 2x IFM -1, if align_corners is True  
        OFM W and H must be 2x IFM, if align_corners is False
- half_pixel_centers are not supported

### TFLite SUB Constraints

This is a list of constraints that the SUB operator must satisfy in order to be scheduled on the NPU.

- Broadcasting is only allowed for rank indices with dimension 1, from either IFM1 or IFM2
