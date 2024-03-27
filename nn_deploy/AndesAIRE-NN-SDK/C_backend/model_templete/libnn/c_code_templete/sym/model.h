#ifndef __MODEL_H__
#define __MODEL_H__

#include "riscv_math_types.h"
#include "riscv_nn_activation.h"
#include "riscv_nn_basic.h"
#include "riscv_nn_convolution.h"
#include "riscv_nn_concatenation.h"
#include "riscv_nn_fully_connected.h"
#include "riscv_nn_pooling.h"
#include "riscv_nn_softmax.h"
#include "riscv_nn_util.h"
#include "nds32_basic_math.h"
#include "nds32_utils_math.h"

#include "perf_counter.h"

#define HP_SOFTMAX 1

#define LAYER0_IN_DIM	24
#define LAYER0_OUT_DIM	72

#define LAYER2_IN_DIM	42
#define LAYER2_OUT_DIM	24

#define LAYER3_IN_DIM	24
#define LAYER3_OUT_DIM	24

#define LAYER4_IN_DIM	24
#define LAYER4_OUT_DIM	72

#define LAYER6_OUT_DIM	24

#define LAYER7_IN_DIM	24
#define LAYER7_OUT_DIM	24

#define LAYER8_OUT_DIM	24

#define LAYER9_OUT_DIM	24

#define LAYER10_IN_DIM	24
#define LAYER10_OUT_DIM	24

#define LAYER11_OUT_DIM	24

#define LAYER12_IN_DIM	24
#define LAYER12_OUT_DIM	24

#define LAYER13_OUT_DIM	24

#define LAYER14_OUT_DIM	24

#define LAYER15_OUT_DIM	24

#define LAYER16_OUT_DIM	24

#define LAYER17_OUT_DIM	42

#define LAYER18_OUT_DIM	24

#define LAYER19_OUT_DIM	24

#define LAYER20_OUT_DIM_X	1
#define LAYER20_OUT_DIM_Y	1
#define LAYER20_OUT_CH	90

#define LAYER21_IN_DIM	90
#define LAYER21_OUT_DIM	144

#define LAYER23_IN_DIM	48
#define LAYER23_OUT_DIM	144

#define LAYER25_OUT_DIM	48

#define LAYER26_IN_DIM	48
#define LAYER26_OUT_DIM	48

#define LAYER27_OUT_DIM	48

#define LAYER28_OUT_DIM	48

#define LAYER29_IN_DIM	48
#define LAYER29_OUT_DIM	48

#define LAYER30_OUT_DIM	48

#define LAYER31_OUT_DIM	48

#define LAYER32_OUT_DIM	48

#define LAYER33_OUT_DIM	48

#define LAYER34_OUT_DIM	48

#define LAYER35_IN_DIM	96
#define LAYER35_OUT_DIM	288

#define LAYER37_OUT_DIM	48

#define LAYER38_OUT_DIM	42

#define LAYER39_OUT_DIM	24

#define LAYER40_OUT_DIM_X	1
#define LAYER40_OUT_DIM_Y	1
#define LAYER40_OUT_CH	114

#define LAYER41_IN_DIM	114
#define LAYER41_OUT_DIM	288

#define LAYER43_OUT_DIM	96

#define LAYER44_IN_DIM	96
#define LAYER44_OUT_DIM	96

#define LAYER45_OUT_DIM	96

#define LAYER46_OUT_DIM	96

#define LAYER47_OUT_DIM	96

#define LAYER48_IN_DIM	96
#define LAYER48_OUT_DIM	96

#define LAYER49_OUT_DIM	96

#define LAYER50_OUT_DIM	96

#define LAYER51_IN_DIM	96
#define LAYER51_OUT_DIM	96

#define LAYER52_OUT_DIM	96

#define LAYER53_OUT_DIM	96

#define LAYER54_IN_DIM	96
#define LAYER54_OUT_DIM	22

#define LAYER55_IN_DIM	22
#define LAYER55_OUT_DIM	22

#define LAYER56_IN_DIM	24
#define LAYER56_OUT_DIM	1

#define LAYER57_IN_DIM	1
#define LAYER57_OUT_DIM	1

q7_t buffer[1056] __attribute__ ((aligned(64)));
q15_t col_buffer[228] __attribute__ ((aligned(64)));

extern q7_t LAYER0_weights;
extern q31_t LAYER0_biases;
extern q7_t LAYER2_weights;
extern q31_t LAYER2_biases;
extern q7_t LAYER4_weights;
extern q31_t LAYER4_biases;
extern q7_t LAYER21_weights;
extern q31_t LAYER21_biases;
extern q7_t LAYER23_weights;
extern q31_t LAYER23_biases;
extern q7_t LAYER35_weights;
extern q31_t LAYER35_biases;
extern q7_t LAYER41_weights;
extern q31_t LAYER41_biases;
extern q7_t LAYER54_weights;
extern q31_t LAYER54_biases;
extern q7_t LAYER56_weights;
extern q31_t LAYER56_biases;

const q7_t *LAYER0_wt = &LAYER0_weights;
const q31_t *LAYER0_bias = &LAYER0_biases;
const q7_t *LAYER2_wt = &LAYER2_weights;
const q31_t *LAYER2_bias = &LAYER2_biases;
const q7_t *LAYER4_wt = &LAYER4_weights;
const q31_t *LAYER4_bias = &LAYER4_biases;
const q7_t *LAYER21_wt = &LAYER21_weights;
const q31_t *LAYER21_bias = &LAYER21_biases;
const q7_t *LAYER23_wt = &LAYER23_weights;
const q31_t *LAYER23_bias = &LAYER23_biases;
const q7_t *LAYER35_wt = &LAYER35_weights;
const q31_t *LAYER35_bias = &LAYER35_biases;
const q7_t *LAYER41_wt = &LAYER41_weights;
const q31_t *LAYER41_bias = &LAYER41_biases;
const q7_t *LAYER54_wt = &LAYER54_weights;
const q31_t *LAYER54_bias = &LAYER54_biases;
const q7_t *LAYER56_wt = &LAYER56_weights;
const q31_t *LAYER56_bias = &LAYER56_biases;


#define LAYER0_PRE_SHIFT	4
#define LAYER0_SCALAR	38931
#define LAYER0_SHIFT	22

#define LAYER2_PRE_SHIFT	1
#define LAYER2_SCALAR	36657
#define LAYER2_SHIFT	21

#define LAYER4_PRE_SHIFT	4
#define LAYER4_SCALAR	37011
#define LAYER4_SHIFT	22

#define LAYER6_SCALAR1	4342240
#define LAYER6_SCALAR2	4194304
#define LAYER6_PRE_SHIFT	23
#define LAYER6_SCALAR	5269988
#define LAYER6_SHIFT	22

//___________________________________________________________


#define LAYER8_SCALAR	1133945482
#define LAYER8_SHIFT	-7

//___________________________________________________________


#define LAYER9_SCALAR1	4342240
#define LAYER9_SCALAR2	4194304
#define LAYER9_PRE_SHIFT	23
#define LAYER9_SCALAR	5195843
#define LAYER9_SHIFT	22

//___________________________________________________________


#define LAYER11_SCALAR1	4585705
#define LAYER11_SCALAR2	4194304
#define LAYER11_PRE_SHIFT	23
#define LAYER11_SCALAR	5886387
#define LAYER11_SHIFT	22

//___________________________________________________________


#define LAYER13_SCALAR1	1073741824
#define LAYER13_SHIFT1	0
#define LAYER13_SCALAR2	2130706440
#define LAYER13_SHIFT2	2
#define LAYER13_SCALARO	1074537306
#define LAYER13_SHIFTO	21

//___________________________________________________________


#define LAYER14_SCALAR	1073906563
#define LAYER14_SHIFT	-6

//___________________________________________________________


#define LAYER15_SCALAR	1083674958
#define LAYER15_SHIFT	-7

//___________________________________________________________


#define LAYER16_SCALAR1	4194304
#define LAYER16_SCALAR2	4195257
#define LAYER16_PRE_SHIFT	23
#define LAYER16_SCALAR	8377162
#define LAYER16_SHIFT	22

//___________________________________________________________


#define LAYER17_SCALAR1	4194304
#define LAYER17_SCALAR2	4194304
#define LAYER17_PRE_SHIFT	23
#define LAYER17_SCALAR	6822823
#define LAYER17_SHIFT	20

//___________________________________________________________


#define LAYER18_SCALAR1	4194304
#define LAYER18_SCALAR2	4194304
#define LAYER18_PRE_SHIFT	23
#define LAYER18_SCALAR	5292441
#define LAYER18_SHIFT	25

//___________________________________________________________


#define LAYER19_SCALAR1	4194304
#define LAYER19_SCALAR2	4194304
#define LAYER19_PRE_SHIFT	23
#define LAYER19_SCALAR	5249813
#define LAYER19_SHIFT	25

//___________________________________________________________


#define LAYER21_PRE_SHIFT	3
#define LAYER21_SCALAR	39123
#define LAYER21_SHIFT	21

#define LAYER23_PRE_SHIFT	2
#define LAYER23_SCALAR	36021
#define LAYER23_SHIFT	22

#define LAYER25_SCALAR1	5992587
#define LAYER25_SCALAR2	4194304
#define LAYER25_PRE_SHIFT	23
#define LAYER25_SCALAR	4889985
#define LAYER25_SHIFT	22

//___________________________________________________________


#define LAYER27_SCALAR	1128033795
#define LAYER27_SHIFT	-6

//___________________________________________________________


#define LAYER28_SCALAR1	5992587
#define LAYER28_SCALAR2	4194304
#define LAYER28_PRE_SHIFT	23
#define LAYER28_SCALAR	7082444
#define LAYER28_SHIFT	22

//___________________________________________________________


#define LAYER30_SCALAR	2071203122
#define LAYER30_SHIFT	-7

//___________________________________________________________


#define LAYER31_SCALAR1	6295592
#define LAYER31_SCALAR2	2097152
#define LAYER31_PRE_SHIFT	22
#define LAYER31_SCALAR	2761828
#define LAYER31_SHIFT	22

//___________________________________________________________


#define LAYER32_SCALAR1	1073741824
#define LAYER32_SHIFT1	0
#define LAYER32_SCALAR2	2130706440
#define LAYER32_SHIFT2	2
#define LAYER32_SCALARO	1074528784
#define LAYER32_SHIFTO	21

//___________________________________________________________


#define LAYER33_SCALAR	1430397355
#define LAYER33_SHIFT	-5

//___________________________________________________________


#define LAYER34_SCALAR1	4194304
#define LAYER34_SCALAR2	8090637
#define LAYER34_PRE_SHIFT	23
#define LAYER34_SCALAR	4348776
#define LAYER34_SHIFT	22

//___________________________________________________________


#define LAYER35_PRE_SHIFT	5
#define LAYER35_SCALAR	37569
#define LAYER35_SHIFT	22

#define LAYER37_SCALAR1	4194304
#define LAYER37_SCALAR2	4194304
#define LAYER37_PRE_SHIFT	23
#define LAYER37_SCALAR	4486345
#define LAYER37_SHIFT	22

//___________________________________________________________


#define LAYER38_SCALAR1	4194304
#define LAYER38_SCALAR2	4194304
#define LAYER38_PRE_SHIFT	23
#define LAYER38_SCALAR	5662448
#define LAYER38_SHIFT	20

//___________________________________________________________


#define LAYER39_SCALAR1	4194304
#define LAYER39_SCALAR2	4194304
#define LAYER39_PRE_SHIFT	23
#define LAYER39_SCALAR	4392342
#define LAYER39_SHIFT	25

//___________________________________________________________


#define LAYER41_PRE_SHIFT	2
#define LAYER41_SCALAR	41839
#define LAYER41_SHIFT	22

#define LAYER43_SCALAR1	7136580
#define LAYER43_SCALAR2	4194304
#define LAYER43_PRE_SHIFT	23
#define LAYER43_SCALAR	4742183
#define LAYER43_SHIFT	22

//___________________________________________________________


#define LAYER45_SCALAR	1083254235
#define LAYER45_SHIFT	-7

//___________________________________________________________


#define LAYER46_SCALAR1	1073741824
#define LAYER46_SHIFT1	0
#define LAYER46_SCALAR2	2130706440
#define LAYER46_SHIFT2	2
#define LAYER46_SCALARO	1074528784
#define LAYER46_SHIFTO	21

//___________________________________________________________


#define LAYER47_SCALAR1	7136580
#define LAYER47_SCALAR2	4194304
#define LAYER47_PRE_SHIFT	23
#define LAYER47_SCALAR	4347582
#define LAYER47_SHIFT	22

//___________________________________________________________


#define LAYER49_SCALAR	1279699632
#define LAYER49_SHIFT	-7

//___________________________________________________________


#define LAYER50_SCALAR1	4252735
#define LAYER50_SCALAR2	2097152
#define LAYER50_PRE_SHIFT	22
#define LAYER50_SCALAR	2261583
#define LAYER50_SHIFT	21

//___________________________________________________________


#define LAYER52_SCALAR	1073741824
#define LAYER52_SHIFT	-6

//___________________________________________________________


#define LAYER53_SCALAR1	4194304
#define LAYER53_SCALAR2	4194304
#define LAYER53_PRE_SHIFT	23
#define LAYER53_SCALAR	8380416
#define LAYER53_SHIFT	22

//___________________________________________________________


#define LAYER54_PRE_SHIFT	5
#define LAYER54_SCALAR	45100
#define LAYER54_SHIFT	22

#define LAYER56_PRE_SHIFT	5
#define LAYER56_SCALAR	50598
#define LAYER56_SHIFT	22

#define LAYER3_RANGE_RADIUS	60
#define LAYER3_SCALAR	19145
#define LAYER3_LEFT_SHIFT	9

#define LAYER7_RANGE_RADIUS	120
#define LAYER7_SCALAR	22434
#define LAYER7_LEFT_SHIFT	8

#define LAYER10_RANGE_RADIUS	120
#define LAYER10_SCALAR	22754
#define LAYER10_LEFT_SHIFT	8

#define LAYER12_RANGE_RADIUS	120
#define LAYER12_SCALAR	19018
#define LAYER12_LEFT_SHIFT	8

#define LAYER26_RANGE_RADIUS	60
#define LAYER26_SCALAR	26481
#define LAYER26_LEFT_SHIFT	9

#define LAYER29_RANGE_RADIUS	60
#define LAYER29_SCALAR	18284
#define LAYER29_LEFT_SHIFT	9

#define LAYER44_RANGE_RADIUS	60
#define LAYER44_SCALAR	25835
#define LAYER44_LEFT_SHIFT	9

#define LAYER48_RANGE_RADIUS	60
#define LAYER48_SCALAR	28180
#define LAYER48_LEFT_SHIFT	9

#define LAYER51_RANGE_RADIUS	60
#define LAYER51_SCALAR	22727
#define LAYER51_LEFT_SHIFT	9

#define LAYER55_RANGE_RADIUS	120
#define LAYER55_SCALAR	24142
#define LAYER55_LEFT_SHIFT	8

#define LAYER57_RANGE_RADIUS	120
#define LAYER57_SCALAR	21518
#define LAYER57_LEFT_SHIFT	8

unsigned long long total_after_cycle, total_after_instret;

#endif /* __MODEL_H__ */