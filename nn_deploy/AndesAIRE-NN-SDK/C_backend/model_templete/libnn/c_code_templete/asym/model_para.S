.global LAYER0_weights
.type LAYER0_weights, @object
.global LAYER0_biases
.type LAYER0_biases, @object
.global LAYER0_multipliers
.type LAYER0_multipliers, @object
.global LAYER0_shifts
.type LAYER0_shifts, @object
.global LAYER2_weights
.type LAYER2_weights, @object
.global LAYER2_biases
.type LAYER2_biases, @object
.global LAYER2_multipliers
.type LAYER2_multipliers, @object
.global LAYER2_shifts
.type LAYER2_shifts, @object
.global LAYER4_weights
.type LAYER4_weights, @object
.global LAYER4_biases
.type LAYER4_biases, @object
.global LAYER4_multipliers
.type LAYER4_multipliers, @object
.global LAYER4_shifts
.type LAYER4_shifts, @object
.global LAYER17_weights
.type LAYER17_weights, @object
.global LAYER17_biases
.type LAYER17_biases, @object
.global LAYER17_multipliers
.type LAYER17_multipliers, @object
.global LAYER17_shifts
.type LAYER17_shifts, @object
.global LAYER23_weights
.type LAYER23_weights, @object
.global LAYER23_biases
.type LAYER23_biases, @object
.global LAYER23_multipliers
.type LAYER23_multipliers, @object
.global LAYER23_shifts
.type LAYER23_shifts, @object
.global LAYER39_weights
.type LAYER39_weights, @object
.global LAYER39_biases
.type LAYER39_biases, @object
.global LAYER39_multipliers
.type LAYER39_multipliers, @object
.global LAYER39_shifts
.type LAYER39_shifts, @object
.global LAYER41_weights
.type LAYER41_weights, @object
.global LAYER41_biases
.type LAYER41_biases, @object
.global LAYER41_multipliers
.type LAYER41_multipliers, @object
.global LAYER41_shifts
.type LAYER41_shifts, @object
.global LAYER54_weights
.type LAYER54_weights, @object
.global LAYER54_biases
.type LAYER54_biases, @object
.global LAYER54_multipliers
.type LAYER54_multipliers, @object
.global LAYER54_shifts
.type LAYER54_shifts, @object
.global LAYER56_weights
.type LAYER56_weights, @object
.global LAYER56_biases
.type LAYER56_biases, @object
.global LAYER56_multipliers
.type LAYER56_multipliers, @object
.global LAYER56_shifts
.type LAYER56_shifts, @object

.section .data.bin , "a", @progbits
.equ DMA_ADDR_ALIGN, 3

.align DMA_ADDR_ALIGN
LAYER0_weights:
	.incbin "./model_para_bin/LAYER0_weights.bin"

.align DMA_ADDR_ALIGN
LAYER0_biases:
	.incbin "./model_para_bin/LAYER0_biases.bin"

.align DMA_ADDR_ALIGN
LAYER0_multipliers:
	.incbin "./model_para_bin/LAYER0_multipliers.bin"

.align DMA_ADDR_ALIGN
LAYER0_shifts:
	.incbin "./model_para_bin/LAYER0_shifts.bin"

.align DMA_ADDR_ALIGN
LAYER2_weights:
	.incbin "./model_para_bin/LAYER2_weights.bin"

.align DMA_ADDR_ALIGN
LAYER2_biases:
	.incbin "./model_para_bin/LAYER2_biases.bin"

.align DMA_ADDR_ALIGN
LAYER2_multipliers:
	.incbin "./model_para_bin/LAYER2_multipliers.bin"

.align DMA_ADDR_ALIGN
LAYER2_shifts:
	.incbin "./model_para_bin/LAYER2_shifts.bin"

.align DMA_ADDR_ALIGN
LAYER4_weights:
	.incbin "./model_para_bin/LAYER4_weights.bin"

.align DMA_ADDR_ALIGN
LAYER4_biases:
	.incbin "./model_para_bin/LAYER4_biases.bin"

.align DMA_ADDR_ALIGN
LAYER4_multipliers:
	.incbin "./model_para_bin/LAYER4_multipliers.bin"

.align DMA_ADDR_ALIGN
LAYER4_shifts:
	.incbin "./model_para_bin/LAYER4_shifts.bin"

.align DMA_ADDR_ALIGN
LAYER17_weights:
	.incbin "./model_para_bin/LAYER17_weights.bin"

.align DMA_ADDR_ALIGN
LAYER17_biases:
	.incbin "./model_para_bin/LAYER17_biases.bin"

.align DMA_ADDR_ALIGN
LAYER17_multipliers:
	.incbin "./model_para_bin/LAYER17_multipliers.bin"

.align DMA_ADDR_ALIGN
LAYER17_shifts:
	.incbin "./model_para_bin/LAYER17_shifts.bin"

.align DMA_ADDR_ALIGN
LAYER23_weights:
	.incbin "./model_para_bin/LAYER23_weights.bin"

.align DMA_ADDR_ALIGN
LAYER23_biases:
	.incbin "./model_para_bin/LAYER23_biases.bin"

.align DMA_ADDR_ALIGN
LAYER23_multipliers:
	.incbin "./model_para_bin/LAYER23_multipliers.bin"

.align DMA_ADDR_ALIGN
LAYER23_shifts:
	.incbin "./model_para_bin/LAYER23_shifts.bin"

.align DMA_ADDR_ALIGN
LAYER39_weights:
	.incbin "./model_para_bin/LAYER39_weights.bin"

.align DMA_ADDR_ALIGN
LAYER39_biases:
	.incbin "./model_para_bin/LAYER39_biases.bin"

.align DMA_ADDR_ALIGN
LAYER39_multipliers:
	.incbin "./model_para_bin/LAYER39_multipliers.bin"

.align DMA_ADDR_ALIGN
LAYER39_shifts:
	.incbin "./model_para_bin/LAYER39_shifts.bin"

.align DMA_ADDR_ALIGN
LAYER41_weights:
	.incbin "./model_para_bin/LAYER41_weights.bin"

.align DMA_ADDR_ALIGN
LAYER41_biases:
	.incbin "./model_para_bin/LAYER41_biases.bin"

.align DMA_ADDR_ALIGN
LAYER41_multipliers:
	.incbin "./model_para_bin/LAYER41_multipliers.bin"

.align DMA_ADDR_ALIGN
LAYER41_shifts:
	.incbin "./model_para_bin/LAYER41_shifts.bin"

.align DMA_ADDR_ALIGN
LAYER54_weights:
	.incbin "./model_para_bin/LAYER54_weights.bin"

.align DMA_ADDR_ALIGN
LAYER54_biases:
	.incbin "./model_para_bin/LAYER54_biases.bin"

.align DMA_ADDR_ALIGN
LAYER54_multipliers:
	.incbin "./model_para_bin/LAYER54_multipliers.bin"

.align DMA_ADDR_ALIGN
LAYER54_shifts:
	.incbin "./model_para_bin/LAYER54_shifts.bin"

.align DMA_ADDR_ALIGN
LAYER56_weights:
	.incbin "./model_para_bin/LAYER56_weights.bin"

.align DMA_ADDR_ALIGN
LAYER56_biases:
	.incbin "./model_para_bin/LAYER56_biases.bin"

.align DMA_ADDR_ALIGN
LAYER56_multipliers:
	.incbin "./model_para_bin/LAYER56_multipliers.bin"

.align DMA_ADDR_ALIGN
LAYER56_shifts:
	.incbin "./model_para_bin/LAYER56_shifts.bin"

