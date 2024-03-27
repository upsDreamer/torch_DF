.global LAYER0_weights
.type LAYER0_weights, @object
.global LAYER0_biases
.type LAYER0_biases, @object
.global LAYER2_weights
.type LAYER2_weights, @object
.global LAYER2_biases
.type LAYER2_biases, @object
.global LAYER4_weights
.type LAYER4_weights, @object
.global LAYER4_biases
.type LAYER4_biases, @object
.global LAYER21_weights
.type LAYER21_weights, @object
.global LAYER21_biases
.type LAYER21_biases, @object
.global LAYER23_weights
.type LAYER23_weights, @object
.global LAYER23_biases
.type LAYER23_biases, @object
.global LAYER35_weights
.type LAYER35_weights, @object
.global LAYER35_biases
.type LAYER35_biases, @object
.global LAYER41_weights
.type LAYER41_weights, @object
.global LAYER41_biases
.type LAYER41_biases, @object
.global LAYER54_weights
.type LAYER54_weights, @object
.global LAYER54_biases
.type LAYER54_biases, @object
.global LAYER56_weights
.type LAYER56_weights, @object
.global LAYER56_biases
.type LAYER56_biases, @object

.section .data.bin , "a", @progbits
.equ DMA_ADDR_ALIGN, 3

.align DMA_ADDR_ALIGN
LAYER0_weights:
	.incbin "./model_para_bin/LAYER0_weights.bin"

.align DMA_ADDR_ALIGN
LAYER0_biases:
	.incbin "./model_para_bin/LAYER0_biases.bin"

.align DMA_ADDR_ALIGN
LAYER2_weights:
	.incbin "./model_para_bin/LAYER2_weights.bin"

.align DMA_ADDR_ALIGN
LAYER2_biases:
	.incbin "./model_para_bin/LAYER2_biases.bin"

.align DMA_ADDR_ALIGN
LAYER4_weights:
	.incbin "./model_para_bin/LAYER4_weights.bin"

.align DMA_ADDR_ALIGN
LAYER4_biases:
	.incbin "./model_para_bin/LAYER4_biases.bin"

.align DMA_ADDR_ALIGN
LAYER21_weights:
	.incbin "./model_para_bin/LAYER21_weights.bin"

.align DMA_ADDR_ALIGN
LAYER21_biases:
	.incbin "./model_para_bin/LAYER21_biases.bin"

.align DMA_ADDR_ALIGN
LAYER23_weights:
	.incbin "./model_para_bin/LAYER23_weights.bin"

.align DMA_ADDR_ALIGN
LAYER23_biases:
	.incbin "./model_para_bin/LAYER23_biases.bin"

.align DMA_ADDR_ALIGN
LAYER35_weights:
	.incbin "./model_para_bin/LAYER35_weights.bin"

.align DMA_ADDR_ALIGN
LAYER35_biases:
	.incbin "./model_para_bin/LAYER35_biases.bin"

.align DMA_ADDR_ALIGN
LAYER41_weights:
	.incbin "./model_para_bin/LAYER41_weights.bin"

.align DMA_ADDR_ALIGN
LAYER41_biases:
	.incbin "./model_para_bin/LAYER41_biases.bin"

.align DMA_ADDR_ALIGN
LAYER54_weights:
	.incbin "./model_para_bin/LAYER54_weights.bin"

.align DMA_ADDR_ALIGN
LAYER54_biases:
	.incbin "./model_para_bin/LAYER54_biases.bin"

.align DMA_ADDR_ALIGN
LAYER56_weights:
	.incbin "./model_para_bin/LAYER56_weights.bin"

.align DMA_ADDR_ALIGN
LAYER56_biases:
	.incbin "./model_para_bin/LAYER56_biases.bin"

