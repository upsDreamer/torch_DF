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

#define LAYER17_IN_DIM	48
#define LAYER17_OUT_DIM	144

#define LAYER19_OUT_DIM	24

#define LAYER20_OUT_DIM	42

#define LAYER21_OUT_DIM	24

#define LAYER22_OUT_DIM_X	1
#define LAYER22_OUT_DIM_Y	1
#define LAYER22_OUT_CH	90

#define LAYER23_IN_DIM	90
#define LAYER23_OUT_DIM	144

#define LAYER25_OUT_DIM	48

#define LAYER26_IN_DIM	48
#define LAYER26_OUT_DIM	48

#define LAYER27_OUT_DIM	48

#define LAYER28_IN_DIM	48
#define LAYER28_OUT_DIM	48

#define LAYER29_OUT_DIM	48

#define LAYER30_OUT_DIM	48

#define LAYER31_OUT_DIM	48

#define LAYER32_OUT_DIM	48

#define LAYER33_OUT_DIM	48

#define LAYER34_OUT_DIM	48

#define LAYER35_OUT_DIM	48

#define LAYER36_OUT_DIM	24

#define LAYER37_OUT_DIM	42

#define LAYER38_OUT_DIM_X	1
#define LAYER38_OUT_DIM_Y	1
#define LAYER38_OUT_CH	114

#define LAYER39_IN_DIM	114
#define LAYER39_OUT_DIM	288

#define LAYER41_IN_DIM	96
#define LAYER41_OUT_DIM	288

#define LAYER43_OUT_DIM	96

#define LAYER44_IN_DIM	96
#define LAYER44_OUT_DIM	96

#define LAYER45_OUT_DIM	96

#define LAYER46_OUT_DIM	96

#define LAYER47_IN_DIM	96
#define LAYER47_OUT_DIM	96

#define LAYER48_OUT_DIM	96

#define LAYER49_IN_DIM	96
#define LAYER49_OUT_DIM	96

#define LAYER50_OUT_DIM	96

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

extern q7_t LAYER0_weights;
extern q31_t LAYER0_biases;
extern q7_t LAYER2_weights;
extern q31_t LAYER2_biases;
extern q7_t LAYER4_weights;
extern q31_t LAYER4_biases;
extern q7_t LAYER17_weights;
extern q31_t LAYER17_biases;
extern q7_t LAYER23_weights;
extern q31_t LAYER23_biases;
extern q7_t LAYER39_weights;
extern q31_t LAYER39_biases;
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
const q7_t *LAYER17_wt = &LAYER17_weights;
const q31_t *LAYER17_bias = &LAYER17_biases;
const q7_t *LAYER23_wt = &LAYER23_weights;
const q31_t *LAYER23_bias = &LAYER23_biases;
const q7_t *LAYER39_wt = &LAYER39_weights;
const q31_t *LAYER39_bias = &LAYER39_biases;
const q7_t *LAYER41_wt = &LAYER41_weights;
const q31_t *LAYER41_bias = &LAYER41_biases;
const q7_t *LAYER54_wt = &LAYER54_weights;
const q31_t *LAYER54_bias = &LAYER54_biases;
const q7_t *LAYER56_wt = &LAYER56_weights;
const q31_t *LAYER56_bias = &LAYER56_biases;


extern q31_t LAYER0_multipliers;
extern q31_t LAYER0_shifts;
#define LAYER0_SCALAR	1364038745
#define LAYER0_SHIFT	-10

extern q31_t LAYER2_multipliers;
extern q31_t LAYER2_shifts;
#define LAYER2_SCALAR	2082858932
#define LAYER2_SHIFT	-7

extern q31_t LAYER4_multipliers;
extern q31_t LAYER4_shifts;
#define LAYER4_SCALAR	1346441343
#define LAYER4_SHIFT	-10

#define LAYER6_SCALAR1	1073741824
#define LAYER6_SHIFT1	0
#define LAYER6_SCALAR2	2126013355
#define LAYER6_SHIFT2	1
#define LAYER6_SCALARO	1593189243
#define LAYER6_SHIFTO	22

//___________________________________________________________


#define LAYER8_SCALAR	1234468034
#define LAYER8_SHIFT	-7

//___________________________________________________________


#define LAYER9_SCALAR1	1073741824
#define LAYER9_SHIFT1	0
#define LAYER9_SCALAR2	1849209047
#define LAYER9_SHIFT2	1
#define LAYER9_SCALARO	1585027555
#define LAYER9_SHIFTO	22

//___________________________________________________________


#define LAYER11_SCALAR1	1073741824
#define LAYER11_SHIFT1	0
#define LAYER11_SCALAR2	2126013355
#define LAYER11_SHIFT2	1
#define LAYER11_SCALARO	1344140608
#define LAYER11_SHIFTO	22

//___________________________________________________________


#define LAYER13_SCALAR1	1073741824
#define LAYER13_SHIFT1	0
#define LAYER13_SCALAR2	2139094913
#define LAYER13_SHIFT2	1
#define LAYER13_SCALARO	1074800096
#define LAYER13_SHIFTO	21

//___________________________________________________________


#define LAYER14_SCALAR	1075695492
#define LAYER14_SHIFT	-7

//___________________________________________________________


#define LAYER15_SCALAR	1080568264
#define LAYER15_SHIFT	-7

//___________________________________________________________


#define LAYER16_SCALAR1	2137816442
#define LAYER16_SHIFT1	1
#define LAYER16_SCALAR2	1073741824
#define LAYER16_SHIFT2	0
#define LAYER16_SCALARO	2143566592
#define LAYER16_SHIFTO	22

//___________________________________________________________


extern q31_t LAYER17_multipliers;
extern q31_t LAYER17_shifts;
#define LAYER17_SCALAR	1114646499
#define LAYER17_SHIFT	-9

#define LAYER19_SCALAR1	1073741824
#define LAYER19_SHIFT1	0
#define LAYER19_SCALAR2	1073741824
#define LAYER19_SHIFT2	0
#define LAYER19_SCALARO	1115696344
#define LAYER19_SHIFTO	25

//___________________________________________________________


#define LAYER20_SCALAR1	1073741824
#define LAYER20_SHIFT1	0
#define LAYER20_SCALAR2	1073741824
#define LAYER20_SHIFT2	0
#define LAYER20_SCALARO	1437270132
#define LAYER20_SHIFTO	20

//___________________________________________________________


#define LAYER21_SCALAR1	1073741824
#define LAYER21_SHIFT1	0
#define LAYER21_SCALAR2	1073741824
#define LAYER21_SHIFT2	0
#define LAYER21_SCALARO	1112424689
#define LAYER21_SHIFTO	25

//___________________________________________________________


extern q31_t LAYER23_multipliers;
extern q31_t LAYER23_shifts;
#define LAYER23_SCALAR	1489112512
#define LAYER23_SHIFT	-8

#define LAYER25_SCALAR1	1073741824
#define LAYER25_SHIFT1	0
#define LAYER25_SCALAR2	1522012013
#define LAYER25_SHIFT2	1
#define LAYER25_SCALARO	1311327327
#define LAYER25_SHIFTO	21

//___________________________________________________________


#define LAYER27_SCALAR1	1073741824
#define LAYER27_SHIFT1	0
#define LAYER27_SCALAR2	1522012013
#define LAYER27_SHIFT2	1
#define LAYER27_SCALARO	1933768969
#define LAYER27_SHIFTO	22

//___________________________________________________________


#define LAYER29_SCALAR	2039617122
#define LAYER29_SHIFT	-7

//___________________________________________________________


#define LAYER30_SCALAR1	1073741824
#define LAYER30_SHIFT1	0
#define LAYER30_SCALAR2	2139094913
#define LAYER30_SHIFT2	1
#define LAYER30_SCALARO	1074791537
#define LAYER30_SHIFTO	21

//___________________________________________________________


#define LAYER31_SCALAR1	1073741824
#define LAYER31_SHIFT1	0
#define LAYER31_SCALAR2	1602504644
#define LAYER31_SHIFT2	2
#define LAYER31_SCALARO	2088664236
#define LAYER31_SHIFTO	21

//___________________________________________________________


#define LAYER32_SCALAR	1210745684
#define LAYER32_SHIFT	-6

//___________________________________________________________


#define LAYER33_SCALAR	1818193355
#define LAYER33_SHIFT	-7

//___________________________________________________________


#define LAYER34_SCALAR1	2028732298
#define LAYER34_SHIFT1	2
#define LAYER34_SCALAR2	1073741824
#define LAYER34_SHIFT2	0
#define LAYER34_SCALARO	1342439714
#define LAYER34_SHIFTO	21

//___________________________________________________________


#define LAYER35_SCALAR1	1073741824
#define LAYER35_SHIFT1	0
#define LAYER35_SCALAR2	1073741824
#define LAYER35_SHIFT2	0
#define LAYER35_SCALARO	1136302046
#define LAYER35_SHIFTO	24

//___________________________________________________________


#define LAYER36_SCALAR1	1073741824
#define LAYER36_SHIFT1	0
#define LAYER36_SCALAR2	1073741824
#define LAYER36_SHIFT2	0
#define LAYER36_SCALARO	1112787081
#define LAYER36_SHIFTO	26

//___________________________________________________________


#define LAYER37_SCALAR1	1073741824
#define LAYER37_SHIFT1	0
#define LAYER37_SCALAR2	1073741824
#define LAYER37_SHIFT2	0
#define LAYER37_SCALARO	1433522341
#define LAYER37_SHIFTO	21

//___________________________________________________________


extern q31_t LAYER39_multipliers;
extern q31_t LAYER39_shifts;
#define LAYER39_SCALAR	1481082970
#define LAYER39_SHIFT	-7

extern q31_t LAYER41_multipliers;
extern q31_t LAYER41_shifts;
#define LAYER41_SCALAR	1358885828
#define LAYER41_SHIFT	-11

#define LAYER43_SCALAR1	1073741824
#define LAYER43_SHIFT1	0
#define LAYER43_SCALAR2	1212854058
#define LAYER43_SHIFT2	1
#define LAYER43_SCALARO	1980611132
#define LAYER43_SHIFTO	22

//___________________________________________________________


#define LAYER45_SCALAR	1432821068
#define LAYER45_SHIFT	-7

//___________________________________________________________


#define LAYER46_SCALAR1	1073741824
#define LAYER46_SHIFT1	0
#define LAYER46_SCALAR2	1817801480
#define LAYER46_SHIFT2	2
#define LAYER46_SCALARO	1892592007
#define LAYER46_SHIFTO	22

//___________________________________________________________


#define LAYER48_SCALAR1	1073741824
#define LAYER48_SHIFT1	0
#define LAYER48_SCALAR2	1212854058
#define LAYER48_SHIFT2	1
#define LAYER48_SCALARO	1966045311
#define LAYER48_SHIFTO	22

//___________________________________________________________


#define LAYER50_SCALAR	1075860353
#define LAYER50_SHIFT	-7

//___________________________________________________________


#define LAYER51_SCALAR1	1073741824
#define LAYER51_SHIFT1	0
#define LAYER51_SCALAR2	2139094913
#define LAYER51_SHIFT2	1
#define LAYER51_SCALARO	1074791537
#define LAYER51_SHIFTO	21

//___________________________________________________________


#define LAYER52_SCALAR	1075845106
#define LAYER52_SHIFT	-7

//___________________________________________________________


#define LAYER53_SCALAR1	2147453214
#define LAYER53_SHIFT1	1
#define LAYER53_SCALAR2	1073741824
#define LAYER53_SHIFT2	0
#define LAYER53_SCALARO	2143285308
#define LAYER53_SHIFTO	22

//___________________________________________________________


extern q31_t LAYER54_multipliers;
extern q31_t LAYER54_shifts;
#define LAYER54_SCALAR	2007379417
#define LAYER54_SHIFT	-11

extern q31_t LAYER56_multipliers;
extern q31_t LAYER56_shifts;
#define LAYER56_SCALAR	1814037284
#define LAYER56_SHIFT	-11

#define LAYER3_RANGE_RADIUS	60
#define LAYER3_SCALAR	21953
#define LAYER3_LEFT_SHIFT	9

#define LAYER7_RANGE_RADIUS	120
#define LAYER7_SCALAR	17715
#define LAYER7_LEFT_SHIFT	8

#define LAYER10_RANGE_RADIUS	120
#define LAYER10_SCALAR	17806
#define LAYER10_LEFT_SHIFT	8

#define LAYER12_RANGE_RADIUS	120
#define LAYER12_SCALAR	20997
#define LAYER12_LEFT_SHIFT	8

#define LAYER26_RANGE_RADIUS	60
#define LAYER26_SCALAR	18784
#define LAYER26_LEFT_SHIFT	9

#define LAYER28_RANGE_RADIUS	60
#define LAYER28_SCALAR	25475
#define LAYER28_LEFT_SHIFT	9

#define LAYER44_RANGE_RADIUS	60
#define LAYER44_SCALAR	25073
#define LAYER44_LEFT_SHIFT	9

#define LAYER47_RANGE_RADIUS	60
#define LAYER47_SCALAR	26239
#define LAYER47_LEFT_SHIFT	9

#define LAYER49_RANGE_RADIUS	60
#define LAYER49_SCALAR	25258
#define LAYER49_LEFT_SHIFT	9

#define LAYER55_RANGE_RADIUS	120
#define LAYER55_SCALAR	17682
#define LAYER55_LEFT_SHIFT	8

#define LAYER57_RANGE_RADIUS	120
#define LAYER57_SCALAR	19567
#define LAYER57_LEFT_SHIFT	8

unsigned long long total_after_cycle, total_after_instret;

#endif /* __MODEL_H__ */