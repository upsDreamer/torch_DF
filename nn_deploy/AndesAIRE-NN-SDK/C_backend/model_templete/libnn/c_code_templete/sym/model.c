#include "model.h"

// libnn version: nds_ver_310_master_8e018642
// andestfliteCompiler sha: c1f189b6082fa82dbb643a215105018aeb81b9c1
// HW ID: 0x10, Config: 0xc01222f9
#if !defined(NULL)
#define NULL 0
#endif

#define OUT_PLACE 0
#undef DUMP_LAYER_ENABLE
#define DUMP_LAYER_ENABLE 0
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>

void run_nn(q7_t *in_data_0, q7_t *in_data_1, q7_t *in_data_2, q7_t *in_data_3, q7_t *out_data_0, q7_t *out_data_1, q7_t *out_data_2, q7_t *out_data_3, q7_t *out_data_4)
{
    #if DUMP_LAYER_ENABLE
        FILE *write_ptr;
        if  ((write_ptr = fopen("./buffer_tmp", "r"))) {
            fclose(write_ptr);
            printf("dir exists\n");
        }
        else {
            printf("dir doesn't exist\n");
        }
    #endif // DUMP_LAYER_ENABLE

    // LAYER0
    // 140468142601936_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER0\n");
    #endif

    riscv_nn_fc_s8_s8_s8_sym_bias(in_data_1, LAYER0_wt, LAYER0_IN_DIM, LAYER0_OUT_DIM, LAYER0_PRE_SHIFT, LAYER0_SCALAR, LAYER0_SHIFT, LAYER0_bias, buffer + 432, (q15_t*)col_buffer);

    #if DUMP_LAYER_ENABLE
        write_ptr = fopen("./buffer_tmp/LAYER0.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER0.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 432, LAYER0_OUT_DIM, 1, write_ptr);
            fclose(write_ptr);
        }
    #endif

    // =================================================================================================================


    // LAYER1
    // torch.chunk_2copycopy_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER1\n");
    #endif

    riscv_nn_split_s8_x(buffer + 432, 72, 1, 1, 1, buffer + 224, 24, 0);
    riscv_nn_split_s8_x(buffer + 432, 72, 1, 1, 1, buffer + 288, 24, 24);
    riscv_nn_split_s8_x(buffer + 432, 72, 1, 1, 1, buffer + 256, 24, 48);

    #if DUMP_LAYER_ENABLE
        write_ptr = fopen("./buffer_tmp/LAYER1_split_0.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER1_split_0.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 224, 1 * 1 * 1 * 24, 1, write_ptr);
            fclose(write_ptr);
        }
        write_ptr = fopen("./buffer_tmp/LAYER1_split_1.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER1_split_1.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 288, 1 * 1 * 1 * 24, 1, write_ptr);
            fclose(write_ptr);
        }
        write_ptr = fopen("./buffer_tmp/LAYER1_split_2.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER1_split_2.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 256, 1 * 1 * 1 * 24, 1, write_ptr);
            fclose(write_ptr);
        }
    #endif

    // =================================================================================================================


    // LAYER2
    // 140468143011920_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER2\n");
    #endif

    riscv_nn_fc_s8_s8_s8_sym_bias(in_data_0, LAYER2_wt, LAYER2_IN_DIM, LAYER2_OUT_DIM, LAYER2_PRE_SHIFT, LAYER2_SCALAR, LAYER2_SHIFT, LAYER2_bias, buffer + 192, (q15_t*)col_buffer);

    #if DUMP_LAYER_ENABLE
        write_ptr = fopen("./buffer_tmp/LAYER2.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER2.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 192, LAYER2_OUT_DIM, 1, write_ptr);
            fclose(write_ptr);
        }
    #endif

    // =================================================================================================================


    // LAYER3
    // 140468048980688_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER3\n");
    #endif

    riscv_nn_tanh_s8(0, LAYER3_RANGE_RADIUS, LAYER3_SCALAR, LAYER3_LEFT_SHIFT, LAYER3_OUT_DIM, buffer + 192, buffer + 192);

    #if DUMP_LAYER_ENABLE
        write_ptr = fopen("./buffer_tmp/LAYER3.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER3.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 192, LAYER3_OUT_DIM, 1, write_ptr);
            fclose(write_ptr);
        }
    #endif

    // =================================================================================================================


    // LAYER4
    // 140468048981008_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER4\n");
    #endif

    riscv_nn_fc_s8_s8_s8_sym_bias(buffer + 192, LAYER4_wt, LAYER4_IN_DIM, LAYER4_OUT_DIM, LAYER4_PRE_SHIFT, LAYER4_SCALAR, LAYER4_SHIFT, LAYER4_bias, buffer + 432, (q15_t*)col_buffer);

    #if DUMP_LAYER_ENABLE
        write_ptr = fopen("./buffer_tmp/LAYER4.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER4.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 432, LAYER4_OUT_DIM, 1, write_ptr);
            fclose(write_ptr);
        }
    #endif

    // =================================================================================================================


    // LAYER5
    // torch.chunk_9_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER5\n");
    #endif

    riscv_nn_split_s8_x(buffer + 432, 72, 1, 1, 1, buffer + 352, 24, 0);
    riscv_nn_split_s8_x(buffer + 432, 72, 1, 1, 1, buffer + 512, 24, 24);
    riscv_nn_split_s8_x(buffer + 432, 72, 1, 1, 1, buffer + 320, 24, 48);

    #if DUMP_LAYER_ENABLE
        write_ptr = fopen("./buffer_tmp/LAYER5_split_0.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER5_split_0.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 352, 1 * 1 * 1 * 24, 1, write_ptr);
            fclose(write_ptr);
        }
        write_ptr = fopen("./buffer_tmp/LAYER5_split_1.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER5_split_1.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 512, 1 * 1 * 1 * 24, 1, write_ptr);
            fclose(write_ptr);
        }
        write_ptr = fopen("./buffer_tmp/LAYER5_split_2.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER5_split_2.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 320, 1 * 1 * 1 * 24, 1, write_ptr);
            fclose(write_ptr);
        }
    #endif

    // =================================================================================================================


    // LAYER6
    // torch.add_11_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER6\n");
    #endif

    riscv_nn_add_s8_sym_round(buffer + 512, buffer + 288, LAYER6_SCALAR1, LAYER6_SCALAR2, LAYER6_OUT_DIM, LAYER6_PRE_SHIFT, LAYER6_SCALAR, LAYER6_SHIFT, buffer + 432);

    #if DUMP_LAYER_ENABLE
        write_ptr = fopen("./buffer_tmp/LAYER6.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER6.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 432, LAYER6_OUT_DIM, 1, write_ptr);
            fclose(write_ptr);
        }
    #endif

    // =================================================================================================================


    // LAYER7
    // 140468142569872_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER7\n");
    #endif

    riscv_nn_sigmoid_s8(0, LAYER7_RANGE_RADIUS, LAYER7_SCALAR, LAYER7_LEFT_SHIFT, LAYER7_OUT_DIM, buffer + 432, buffer + 432);

    #if DUMP_LAYER_ENABLE
        write_ptr = fopen("./buffer_tmp/LAYER7.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER7.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 432, LAYER7_OUT_DIM, 1, write_ptr);
            fclose(write_ptr);
        }
    #endif

    // =================================================================================================================


    // LAYER8
    // torch.mul_18_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER8\n");
    #endif

    riscv_nn_ew_mul_s8_asym(buffer + 432, buffer + 256, 128, 0, buffer + 288, 0, LAYER8_SCALAR, LAYER8_SHIFT, -128, 127, LAYER8_OUT_DIM);

    #if DUMP_LAYER_ENABLE
        write_ptr = fopen("./buffer_tmp/LAYER8.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER8.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 288, LAYER8_OUT_DIM, 1, write_ptr);
            fclose(write_ptr);
        }
    #endif

    // =================================================================================================================


    // LAYER9
    // torch.add_26_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER9\n");
    #endif

    riscv_nn_add_s8_sym_round(buffer + 352, buffer + 224, LAYER9_SCALAR1, LAYER9_SCALAR2, LAYER9_OUT_DIM, LAYER9_PRE_SHIFT, LAYER9_SCALAR, LAYER9_SHIFT, buffer + 256);

    #if DUMP_LAYER_ENABLE
        write_ptr = fopen("./buffer_tmp/LAYER9.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER9.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 256, LAYER9_OUT_DIM, 1, write_ptr);
            fclose(write_ptr);
        }
    #endif

    // =================================================================================================================


    // LAYER10
    // 140468049666896_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER10\n");
    #endif

    riscv_nn_sigmoid_s8(0, LAYER10_RANGE_RADIUS, LAYER10_SCALAR, LAYER10_LEFT_SHIFT, LAYER10_OUT_DIM, buffer + 256, buffer + 256);

    #if DUMP_LAYER_ENABLE
        write_ptr = fopen("./buffer_tmp/LAYER10.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER10.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 256, LAYER10_OUT_DIM, 1, write_ptr);
            fclose(write_ptr);
        }
    #endif

    // =================================================================================================================


    // LAYER11
    // torch.add_22_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER11\n");
    #endif

    riscv_nn_add_s8_sym_round(buffer + 320, buffer + 288, LAYER11_SCALAR1, LAYER11_SCALAR2, LAYER11_OUT_DIM, LAYER11_PRE_SHIFT, LAYER11_SCALAR, LAYER11_SHIFT, buffer + 224);

    #if DUMP_LAYER_ENABLE
        write_ptr = fopen("./buffer_tmp/LAYER11.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER11.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 224, LAYER11_OUT_DIM, 1, write_ptr);
            fclose(write_ptr);
        }
    #endif

    // =================================================================================================================


    // LAYER12
    // 140468048695824_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER12\n");
    #endif

    riscv_nn_tanh_s8(0, LAYER12_RANGE_RADIUS, LAYER12_SCALAR, LAYER12_LEFT_SHIFT, LAYER12_OUT_DIM, buffer + 224, buffer + 224);

    #if DUMP_LAYER_ENABLE
        write_ptr = fopen("./buffer_tmp/LAYER12.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER12.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 224, LAYER12_OUT_DIM, 1, write_ptr);
            fclose(write_ptr);
        }
    #endif

    // =================================================================================================================


    // LAYER13
    // torch.sub_32_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER13\n");
    #endif

    riscv_nn_ew_rsubc_s8_asym(buffer + 256, 532676608, 128, LAYER13_SCALAR2, LAYER13_SHIFT2, 23, buffer + 320, 0, LAYER13_SCALARO, LAYER13_SHIFTO, -128, 127, LAYER13_OUT_DIM);

    #if DUMP_LAYER_ENABLE
        write_ptr = fopen("./buffer_tmp/LAYER13.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER13.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 320, LAYER13_OUT_DIM, 1, write_ptr);
            fclose(write_ptr);
        }
    #endif

    // =================================================================================================================


    // LAYER14
    // torch.mul_34_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER14\n");
    #endif

    riscv_nn_ew_mul_s8_asym(buffer + 320, buffer + 224, 0, 0, buffer + 288, 0, LAYER14_SCALAR, LAYER14_SHIFT, -128, 127, LAYER14_OUT_DIM);

    #if DUMP_LAYER_ENABLE
        write_ptr = fopen("./buffer_tmp/LAYER14.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER14.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 288, LAYER14_OUT_DIM, 1, write_ptr);
            fclose(write_ptr);
        }
    #endif

    // =================================================================================================================


    // LAYER15
    // torch.mul_36_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER15\n");
    #endif

    riscv_nn_ew_mul_s8_asym(buffer + 256, in_data_1, 128, 0, buffer + 224, 0, LAYER15_SCALAR, LAYER15_SHIFT, -128, 127, LAYER15_OUT_DIM);

    #if DUMP_LAYER_ENABLE
        write_ptr = fopen("./buffer_tmp/LAYER15.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER15.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 224, LAYER15_OUT_DIM, 1, write_ptr);
            fclose(write_ptr);
        }
    #endif

    // =================================================================================================================


    // LAYER16
    // torch.add_38_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER16\n");
    #endif

    riscv_nn_add_s8_sym_round(buffer + 224, buffer + 288, LAYER16_SCALAR1, LAYER16_SCALAR2, LAYER16_OUT_DIM, LAYER16_PRE_SHIFT, LAYER16_SCALAR, LAYER16_SHIFT, out_data_0);

    #if DUMP_LAYER_ENABLE
        write_ptr = fopen("./buffer_tmp/LAYER16.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER16.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(out_data_0, LAYER16_OUT_DIM, 1, write_ptr);
            fclose(write_ptr);
        }
    #endif

    // =================================================================================================================


    // LAYER17
    // 140467114074320_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER17\n");
    #endif

    memset(buffer + 224, 0, LAYER17_OUT_DIM);
    riscv_nn_add_s8_sym_round(in_data_0, buffer + 224, LAYER17_SCALAR1, LAYER17_SCALAR2, LAYER17_OUT_DIM, LAYER17_PRE_SHIFT, LAYER17_SCALAR, LAYER17_SHIFT, buffer + 288);

    #if DUMP_LAYER_ENABLE
        write_ptr = fopen("./buffer_tmp/LAYER17.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER17.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 288, LAYER17_OUT_DIM, 1, write_ptr);
            fclose(write_ptr);
        }
    #endif

    // =================================================================================================================


    // LAYER18
    // 140467114077264_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER18\n");
    #endif

    memset(buffer + 224, 0, LAYER18_OUT_DIM);
    riscv_nn_add_s8_sym_round(out_data_0, buffer + 224, LAYER18_SCALAR1, LAYER18_SCALAR2, LAYER18_OUT_DIM, LAYER18_PRE_SHIFT, LAYER18_SCALAR, LAYER18_SHIFT, buffer + 336);

    #if DUMP_LAYER_ENABLE
        write_ptr = fopen("./buffer_tmp/LAYER18.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER18.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 336, LAYER18_OUT_DIM, 1, write_ptr);
            fclose(write_ptr);
        }
    #endif

    // =================================================================================================================


    // LAYER19
    // 140467114076688_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER19\n");
    #endif

    memset(buffer + 224, 0, LAYER19_OUT_DIM);
    riscv_nn_add_s8_sym_round(buffer + 192, buffer + 224, LAYER19_SCALAR1, LAYER19_SCALAR2, LAYER19_OUT_DIM, LAYER19_PRE_SHIFT, LAYER19_SCALAR, LAYER19_SHIFT, buffer + 432);

    #if DUMP_LAYER_ENABLE
        write_ptr = fopen("./buffer_tmp/LAYER19.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER19.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 432, LAYER19_OUT_DIM, 1, write_ptr);
            fclose(write_ptr);
        }
    #endif

    // =================================================================================================================


    // LAYER20
    // torch.cat_49_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER20\n");
    #endif

    riscv_nn_concate_s8_x(buffer + 432, 24, 1, 1, 1, buffer + 192, 90, 0);
    riscv_nn_concate_s8_x(buffer + 336, 24, 1, 1, 1, buffer + 192, 90, 24);
    riscv_nn_concate_s8_x(buffer + 288, 42, 1, 1, 1, buffer + 192, 90, 48);

    #if DUMP_LAYER_ENABLE
        write_ptr = fopen("./buffer_tmp/LAYER20.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER20.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 192, LAYER20_OUT_DIM_X * LAYER20_OUT_DIM_Y * LAYER20_OUT_CH, 1, write_ptr);
            fclose(write_ptr);
        }
    #endif

    // =================================================================================================================


    // LAYER21
    // 140468048981264_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER21\n");
    #endif

    riscv_nn_fc_s8_s8_s8_sym_bias(buffer + 192, LAYER21_wt, LAYER21_IN_DIM, LAYER21_OUT_DIM, LAYER21_PRE_SHIFT, LAYER21_SCALAR, LAYER21_SHIFT, LAYER21_bias, buffer + 432, (q15_t*)col_buffer);

    #if DUMP_LAYER_ENABLE
        write_ptr = fopen("./buffer_tmp/LAYER21.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER21.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 432, LAYER21_OUT_DIM, 1, write_ptr);
            fclose(write_ptr);
        }
    #endif

    // =================================================================================================================


    // LAYER22
    // torch.chunk_55_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER22\n");
    #endif

    riscv_nn_split_s8_x(buffer + 432, 144, 1, 1, 1, buffer + 240, 48, 0);
    riscv_nn_split_s8_x(buffer + 432, 144, 1, 1, 1, buffer + 336, 48, 48);
    riscv_nn_split_s8_x(buffer + 432, 144, 1, 1, 1, buffer + 192, 48, 96);

    #if DUMP_LAYER_ENABLE
        write_ptr = fopen("./buffer_tmp/LAYER22_split_0.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER22_split_0.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 240, 1 * 1 * 1 * 48, 1, write_ptr);
            fclose(write_ptr);
        }
        write_ptr = fopen("./buffer_tmp/LAYER22_split_1.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER22_split_1.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 336, 1 * 1 * 1 * 48, 1, write_ptr);
            fclose(write_ptr);
        }
        write_ptr = fopen("./buffer_tmp/LAYER22_split_2.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER22_split_2.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 192, 1 * 1 * 1 * 48, 1, write_ptr);
            fclose(write_ptr);
        }
    #endif

    // =================================================================================================================


    // LAYER23
    // 140468143042448_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER23\n");
    #endif

    riscv_nn_fc_s8_s8_s8_sym_bias(in_data_2, LAYER23_wt, LAYER23_IN_DIM, LAYER23_OUT_DIM, LAYER23_PRE_SHIFT, LAYER23_SCALAR, LAYER23_SHIFT, LAYER23_bias, buffer + 528, (q15_t*)col_buffer);

    #if DUMP_LAYER_ENABLE
        write_ptr = fopen("./buffer_tmp/LAYER23.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER23.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 528, LAYER23_OUT_DIM, 1, write_ptr);
            fclose(write_ptr);
        }
    #endif

    // =================================================================================================================


    // LAYER24
    // torch.chunk_2_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER24\n");
    #endif

    riscv_nn_split_s8_x(buffer + 528, 144, 1, 1, 1, buffer + 288, 48, 0);
    riscv_nn_split_s8_x(buffer + 528, 144, 1, 1, 1, buffer + 480, 48, 48);
    riscv_nn_split_s8_x(buffer + 528, 144, 1, 1, 1, buffer + 432, 48, 96);

    #if DUMP_LAYER_ENABLE
        write_ptr = fopen("./buffer_tmp/LAYER24_split_0.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER24_split_0.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 288, 1 * 1 * 1 * 48, 1, write_ptr);
            fclose(write_ptr);
        }
        write_ptr = fopen("./buffer_tmp/LAYER24_split_1.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER24_split_1.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 480, 1 * 1 * 1 * 48, 1, write_ptr);
            fclose(write_ptr);
        }
        write_ptr = fopen("./buffer_tmp/LAYER24_split_2.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER24_split_2.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 432, 1 * 1 * 1 * 48, 1, write_ptr);
            fclose(write_ptr);
        }
    #endif

    // =================================================================================================================


    // LAYER25
    // torch.add_57_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER25\n");
    #endif

    riscv_nn_add_s8_sym_round(buffer + 336, buffer + 480, LAYER25_SCALAR1, LAYER25_SCALAR2, LAYER25_OUT_DIM, LAYER25_PRE_SHIFT, LAYER25_SCALAR, LAYER25_SHIFT, buffer + 528);

    #if DUMP_LAYER_ENABLE
        write_ptr = fopen("./buffer_tmp/LAYER25.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER25.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 528, LAYER25_OUT_DIM, 1, write_ptr);
            fclose(write_ptr);
        }
    #endif

    // =================================================================================================================


    // LAYER26
    // 140468049259600_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER26\n");
    #endif

    riscv_nn_sigmoid_s8(0, LAYER26_RANGE_RADIUS, LAYER26_SCALAR, LAYER26_LEFT_SHIFT, LAYER26_OUT_DIM, buffer + 528, buffer + 528);

    #if DUMP_LAYER_ENABLE
        write_ptr = fopen("./buffer_tmp/LAYER26.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER26.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 528, LAYER26_OUT_DIM, 1, write_ptr);
            fclose(write_ptr);
        }
    #endif

    // =================================================================================================================


    // LAYER27
    // torch.mul_64_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER27\n");
    #endif

    riscv_nn_ew_mul_s8_asym(buffer + 528, buffer + 432, 128, 0, buffer + 336, 0, LAYER27_SCALAR, LAYER27_SHIFT, -128, 127, LAYER27_OUT_DIM);

    #if DUMP_LAYER_ENABLE
        write_ptr = fopen("./buffer_tmp/LAYER27.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER27.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 336, LAYER27_OUT_DIM, 1, write_ptr);
            fclose(write_ptr);
        }
    #endif

    // =================================================================================================================


    // LAYER28
    // torch.add_68_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER28\n");
    #endif

    riscv_nn_add_s8_sym_round(buffer + 240, buffer + 288, LAYER28_SCALAR1, LAYER28_SCALAR2, LAYER28_OUT_DIM, LAYER28_PRE_SHIFT, LAYER28_SCALAR, LAYER28_SHIFT, buffer + 432);

    #if DUMP_LAYER_ENABLE
        write_ptr = fopen("./buffer_tmp/LAYER28.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER28.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 432, LAYER28_OUT_DIM, 1, write_ptr);
            fclose(write_ptr);
        }
    #endif

    // =================================================================================================================


    // LAYER29
    // 140468049258704_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER29\n");
    #endif

    riscv_nn_sigmoid_s8(0, LAYER29_RANGE_RADIUS, LAYER29_SCALAR, LAYER29_LEFT_SHIFT, LAYER29_OUT_DIM, buffer + 432, buffer + 432);

    #if DUMP_LAYER_ENABLE
        write_ptr = fopen("./buffer_tmp/LAYER29.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER29.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 432, LAYER29_OUT_DIM, 1, write_ptr);
            fclose(write_ptr);
        }
    #endif

    // =================================================================================================================


    // LAYER30
    // torch.mul_76_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER30\n");
    #endif

    riscv_nn_ew_mul_s8_asym(buffer + 432, in_data_2, 128, 0, buffer + 240, 0, LAYER30_SCALAR, LAYER30_SHIFT, -128, 127, LAYER30_OUT_DIM);

    #if DUMP_LAYER_ENABLE
        write_ptr = fopen("./buffer_tmp/LAYER30.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER30.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 240, LAYER30_OUT_DIM, 1, write_ptr);
            fclose(write_ptr);
        }
    #endif

    // =================================================================================================================


    // LAYER31
    // torch.add_79_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER31\n");
    #endif

    riscv_nn_add_s8_sym_round(buffer + 192, buffer + 336, LAYER31_SCALAR1, LAYER31_SCALAR2, LAYER31_OUT_DIM, LAYER31_PRE_SHIFT, LAYER31_SCALAR, LAYER31_SHIFT, buffer + 144);
    riscv_nn_relu_s8(buffer + 144, LAYER31_OUT_DIM);

    #if DUMP_LAYER_ENABLE
        write_ptr = fopen("./buffer_tmp/LAYER31.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER31.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 144, LAYER31_OUT_DIM, 1, write_ptr);
            fclose(write_ptr);
        }
    #endif

    // =================================================================================================================


    // LAYER32
    // torch.sub_74_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER32\n");
    #endif

    riscv_nn_ew_rsubc_s8_asym(buffer + 432, 532676608, 128, LAYER32_SCALAR2, LAYER32_SHIFT2, 23, buffer + 192, 0, LAYER32_SCALARO, LAYER32_SHIFTO, -128, 127, LAYER32_OUT_DIM);

    #if DUMP_LAYER_ENABLE
        write_ptr = fopen("./buffer_tmp/LAYER32.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER32.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 192, LAYER32_OUT_DIM, 1, write_ptr);
            fclose(write_ptr);
        }
    #endif

    // =================================================================================================================


    // LAYER33
    // torch.mul_82_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER33\n");
    #endif

    riscv_nn_ew_mul_s8_asym(buffer + 192, buffer + 144, 0, 0, buffer + 288, 0, LAYER33_SCALAR, LAYER33_SHIFT, -128, 127, LAYER33_OUT_DIM);

    #if DUMP_LAYER_ENABLE
        write_ptr = fopen("./buffer_tmp/LAYER33.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER33.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 288, LAYER33_OUT_DIM, 1, write_ptr);
            fclose(write_ptr);
        }
    #endif

    // =================================================================================================================


    // LAYER34
    // torch.add_84_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER34\n");
    #endif

    riscv_nn_add_s8_sym_round(buffer + 240, buffer + 288, LAYER34_SCALAR1, LAYER34_SCALAR2, LAYER34_OUT_DIM, LAYER34_PRE_SHIFT, LAYER34_SCALAR, LAYER34_SHIFT, out_data_2);

    #if DUMP_LAYER_ENABLE
        write_ptr = fopen("./buffer_tmp/LAYER34.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER34.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(out_data_2, LAYER34_OUT_DIM, 1, write_ptr);
            fclose(write_ptr);
        }
    #endif

    // =================================================================================================================


    // LAYER35
    // 140468142577040_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER35\n");
    #endif

    riscv_nn_fc_s8_s8_s8_sym_bias(in_data_3, LAYER35_wt, LAYER35_IN_DIM, LAYER35_OUT_DIM, LAYER35_PRE_SHIFT, LAYER35_SCALAR, LAYER35_SHIFT, LAYER35_bias, buffer + 576, (q15_t*)col_buffer);

    #if DUMP_LAYER_ENABLE
        write_ptr = fopen("./buffer_tmp/LAYER35.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER35.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 576, LAYER35_OUT_DIM, 1, write_ptr);
            fclose(write_ptr);
        }
    #endif

    // =================================================================================================================


    // LAYER36
    // torch.chunk_2copy_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER36\n");
    #endif

    riscv_nn_split_s8_x(buffer + 576, 288, 1, 1, 1, buffer + 480, 96, 0);
    riscv_nn_split_s8_x(buffer + 576, 288, 1, 1, 1, buffer + 288, 96, 96);
    riscv_nn_split_s8_x(buffer + 576, 288, 1, 1, 1, buffer + 192, 96, 192);

    #if DUMP_LAYER_ENABLE
        write_ptr = fopen("./buffer_tmp/LAYER36_split_0.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER36_split_0.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 480, 1 * 1 * 1 * 96, 1, write_ptr);
            fclose(write_ptr);
        }
        write_ptr = fopen("./buffer_tmp/LAYER36_split_1.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER36_split_1.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 288, 1 * 1 * 1 * 96, 1, write_ptr);
            fclose(write_ptr);
        }
        write_ptr = fopen("./buffer_tmp/LAYER36_split_2.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER36_split_2.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 192, 1 * 1 * 1 * 96, 1, write_ptr);
            fclose(write_ptr);
        }
    #endif

    // =================================================================================================================


    // LAYER37
    // 140467114189264_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER37\n");
    #endif

    memset(buffer + 576, 0, LAYER37_OUT_DIM);
    riscv_nn_add_s8_sym_round(out_data_2, buffer + 576, LAYER37_SCALAR1, LAYER37_SCALAR2, LAYER37_OUT_DIM, LAYER37_PRE_SHIFT, LAYER37_SCALAR, LAYER37_SHIFT, buffer + 432);

    #if DUMP_LAYER_ENABLE
        write_ptr = fopen("./buffer_tmp/LAYER37.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER37.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 432, LAYER37_OUT_DIM, 1, write_ptr);
            fclose(write_ptr);
        }
    #endif

    // =================================================================================================================


    // LAYER38
    // 140467114190672_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER38\n");
    #endif

    memset(buffer + 576, 0, LAYER38_OUT_DIM);
    riscv_nn_add_s8_sym_round(in_data_0, buffer + 576, LAYER38_SCALAR1, LAYER38_SCALAR2, LAYER38_OUT_DIM, LAYER38_PRE_SHIFT, LAYER38_SCALAR, LAYER38_SHIFT, buffer + 704);

    #if DUMP_LAYER_ENABLE
        write_ptr = fopen("./buffer_tmp/LAYER38.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER38.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 704, LAYER38_OUT_DIM, 1, write_ptr);
            fclose(write_ptr);
        }
    #endif

    // =================================================================================================================


    // LAYER39
    // 140467114192592_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER39\n");
    #endif

    memset(buffer + 576, 0, LAYER39_OUT_DIM);
    riscv_nn_add_s8_sym_round(out_data_0, buffer + 576, LAYER39_SCALAR1, LAYER39_SCALAR2, LAYER39_OUT_DIM, LAYER39_PRE_SHIFT, LAYER39_SCALAR, LAYER39_SHIFT, buffer + 384);

    #if DUMP_LAYER_ENABLE
        write_ptr = fopen("./buffer_tmp/LAYER39.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER39.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 384, LAYER39_OUT_DIM, 1, write_ptr);
            fclose(write_ptr);
        }
    #endif

    // =================================================================================================================


    // LAYER40
    // torch.cat_90_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER40\n");
    #endif

    riscv_nn_concate_s8_x(buffer + 384, 24, 1, 1, 1, buffer + 576, 114, 0);
    riscv_nn_concate_s8_x(buffer + 432, 48, 1, 1, 1, buffer + 576, 114, 24);
    riscv_nn_concate_s8_x(buffer + 704, 42, 1, 1, 1, buffer + 576, 114, 72);

    #if DUMP_LAYER_ENABLE
        write_ptr = fopen("./buffer_tmp/LAYER40.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER40.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 576, LAYER40_OUT_DIM_X * LAYER40_OUT_DIM_Y * LAYER40_OUT_CH, 1, write_ptr);
            fclose(write_ptr);
        }
    #endif

    // =================================================================================================================


    // LAYER41
    // 140468048980752_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER41\n");
    #endif

    riscv_nn_fc_s8_s8_s8_sym_bias(buffer + 576, LAYER41_wt, LAYER41_IN_DIM, LAYER41_OUT_DIM, LAYER41_PRE_SHIFT, LAYER41_SCALAR, LAYER41_SHIFT, LAYER41_bias, buffer + 768, (q15_t*)col_buffer);

    #if DUMP_LAYER_ENABLE
        write_ptr = fopen("./buffer_tmp/LAYER41.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER41.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 768, LAYER41_OUT_DIM, 1, write_ptr);
            fclose(write_ptr);
        }
    #endif

    // =================================================================================================================


    // LAYER42
    // torch.chunk_96_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER42\n");
    #endif

    riscv_nn_split_s8_x(buffer + 768, 288, 1, 1, 1, buffer + 672, 96, 0);
    riscv_nn_split_s8_x(buffer + 768, 288, 1, 1, 1, buffer + 576, 96, 96);
    riscv_nn_split_s8_x(buffer + 768, 288, 1, 1, 1, buffer + 384, 96, 192);

    #if DUMP_LAYER_ENABLE
        write_ptr = fopen("./buffer_tmp/LAYER42_split_0.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER42_split_0.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 672, 1 * 1 * 1 * 96, 1, write_ptr);
            fclose(write_ptr);
        }
        write_ptr = fopen("./buffer_tmp/LAYER42_split_1.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER42_split_1.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 576, 1 * 1 * 1 * 96, 1, write_ptr);
            fclose(write_ptr);
        }
        write_ptr = fopen("./buffer_tmp/LAYER42_split_2.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER42_split_2.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 384, 1 * 1 * 1 * 96, 1, write_ptr);
            fclose(write_ptr);
        }
    #endif

    // =================================================================================================================


    // LAYER43
    // torch.add_113_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER43\n");
    #endif

    riscv_nn_add_s8_sym_round(buffer + 672, buffer + 480, LAYER43_SCALAR1, LAYER43_SCALAR2, LAYER43_OUT_DIM, LAYER43_PRE_SHIFT, LAYER43_SCALAR, LAYER43_SHIFT, buffer + 768);

    #if DUMP_LAYER_ENABLE
        write_ptr = fopen("./buffer_tmp/LAYER43.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER43.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 768, LAYER43_OUT_DIM, 1, write_ptr);
            fclose(write_ptr);
        }
    #endif

    // =================================================================================================================


    // LAYER44
    // 140468142571600_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER44\n");
    #endif

    riscv_nn_sigmoid_s8(0, LAYER44_RANGE_RADIUS, LAYER44_SCALAR, LAYER44_LEFT_SHIFT, LAYER44_OUT_DIM, buffer + 768, buffer + 768);

    #if DUMP_LAYER_ENABLE
        write_ptr = fopen("./buffer_tmp/LAYER44.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER44.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 768, LAYER44_OUT_DIM, 1, write_ptr);
            fclose(write_ptr);
        }
    #endif

    // =================================================================================================================


    // LAYER45
    // torch.mul_123_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER45\n");
    #endif

    riscv_nn_ew_mul_s8_asym(buffer + 768, in_data_3, 128, 0, buffer + 480, 0, LAYER45_SCALAR, LAYER45_SHIFT, -128, 127, LAYER45_OUT_DIM);

    #if DUMP_LAYER_ENABLE
        write_ptr = fopen("./buffer_tmp/LAYER45.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER45.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 480, LAYER45_OUT_DIM, 1, write_ptr);
            fclose(write_ptr);
        }
    #endif

    // =================================================================================================================


    // LAYER46
    // torch.sub_119_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER46\n");
    #endif

    riscv_nn_ew_rsubc_s8_asym(buffer + 768, 532676608, 128, LAYER46_SCALAR2, LAYER46_SHIFT2, 23, buffer + 48, 0, LAYER46_SCALARO, LAYER46_SHIFTO, -128, 127, LAYER46_OUT_DIM);

    #if DUMP_LAYER_ENABLE
        write_ptr = fopen("./buffer_tmp/LAYER46.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER46.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 48, LAYER46_OUT_DIM, 1, write_ptr);
            fclose(write_ptr);
        }
    #endif

    // =================================================================================================================


    // LAYER47
    // torch.add_98_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER47\n");
    #endif

    riscv_nn_add_s8_sym_round(buffer + 576, buffer + 288, LAYER47_SCALAR1, LAYER47_SCALAR2, LAYER47_OUT_DIM, LAYER47_PRE_SHIFT, LAYER47_SCALAR, LAYER47_SHIFT, buffer + 672);

    #if DUMP_LAYER_ENABLE
        write_ptr = fopen("./buffer_tmp/LAYER47.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER47.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 672, LAYER47_OUT_DIM, 1, write_ptr);
            fclose(write_ptr);
        }
    #endif

    // =================================================================================================================


    // LAYER48
    // 140468049259024_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER48\n");
    #endif

    riscv_nn_sigmoid_s8(0, LAYER48_RANGE_RADIUS, LAYER48_SCALAR, LAYER48_LEFT_SHIFT, LAYER48_OUT_DIM, buffer + 672, buffer + 672);

    #if DUMP_LAYER_ENABLE
        write_ptr = fopen("./buffer_tmp/LAYER48.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER48.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 672, LAYER48_OUT_DIM, 1, write_ptr);
            fclose(write_ptr);
        }
    #endif

    // =================================================================================================================


    // LAYER49
    // torch.mul_105_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER49\n");
    #endif

    riscv_nn_ew_mul_s8_asym(buffer + 672, buffer + 192, 128, 0, buffer + 288, 0, LAYER49_SCALAR, LAYER49_SHIFT, -128, 127, LAYER49_OUT_DIM);

    #if DUMP_LAYER_ENABLE
        write_ptr = fopen("./buffer_tmp/LAYER49.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER49.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 288, LAYER49_OUT_DIM, 1, write_ptr);
            fclose(write_ptr);
        }
    #endif

    // =================================================================================================================


    // LAYER50
    // torch.add_109_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER50\n");
    #endif

    riscv_nn_add_s8_sym_round(buffer + 384, buffer + 288, LAYER50_SCALAR1, LAYER50_SCALAR2, LAYER50_OUT_DIM, LAYER50_PRE_SHIFT, LAYER50_SCALAR, LAYER50_SHIFT, buffer + 192);

    #if DUMP_LAYER_ENABLE
        write_ptr = fopen("./buffer_tmp/LAYER50.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER50.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 192, LAYER50_OUT_DIM, 1, write_ptr);
            fclose(write_ptr);
        }
    #endif

    // =================================================================================================================


    // LAYER51
    // 140468049674512_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER51\n");
    #endif

    riscv_nn_tanh_s8(0, LAYER51_RANGE_RADIUS, LAYER51_SCALAR, LAYER51_LEFT_SHIFT, LAYER51_OUT_DIM, buffer + 192, buffer + 192);

    #if DUMP_LAYER_ENABLE
        write_ptr = fopen("./buffer_tmp/LAYER51.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER51.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 192, LAYER51_OUT_DIM, 1, write_ptr);
            fclose(write_ptr);
        }
    #endif

    // =================================================================================================================


    // LAYER52
    // torch.mul_121_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER52\n");
    #endif

    riscv_nn_ew_mul_s8_asym(buffer + 48, buffer + 192, 0, 0, buffer + 288, 0, LAYER52_SCALAR, LAYER52_SHIFT, -128, 127, LAYER52_OUT_DIM);

    #if DUMP_LAYER_ENABLE
        write_ptr = fopen("./buffer_tmp/LAYER52.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER52.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 288, LAYER52_OUT_DIM, 1, write_ptr);
            fclose(write_ptr);
        }
    #endif

    // =================================================================================================================


    // LAYER53
    // torch.add_125_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER53\n");
    #endif

    riscv_nn_add_s8_sym_round(buffer + 480, buffer + 288, LAYER53_SCALAR1, LAYER53_SCALAR2, LAYER53_OUT_DIM, LAYER53_PRE_SHIFT, LAYER53_SCALAR, LAYER53_SHIFT, out_data_3);

    #if DUMP_LAYER_ENABLE
        write_ptr = fopen("./buffer_tmp/LAYER53.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER53.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(out_data_3, LAYER53_OUT_DIM, 1, write_ptr);
            fclose(write_ptr);
        }
    #endif

    // =================================================================================================================


    // LAYER54
    // 140468048980624_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER54\n");
    #endif

    riscv_nn_fc_s8_s8_s8_sym_bias(out_data_3, LAYER54_wt, LAYER54_IN_DIM, LAYER54_OUT_DIM, LAYER54_PRE_SHIFT, LAYER54_SCALAR, LAYER54_SHIFT, LAYER54_bias, buffer + 192, (q15_t*)col_buffer);

    #if DUMP_LAYER_ENABLE
        write_ptr = fopen("./buffer_tmp/LAYER54.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER54.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 192, LAYER54_OUT_DIM, 1, write_ptr);
            fclose(write_ptr);
        }
    #endif

    // =================================================================================================================


    // LAYER55
    // 140468048944400_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER55\n");
    #endif

    riscv_nn_sigmoid_s8(0, LAYER55_RANGE_RADIUS, LAYER55_SCALAR, LAYER55_LEFT_SHIFT, LAYER55_OUT_DIM, buffer + 192, out_data_4);

    #if DUMP_LAYER_ENABLE
        write_ptr = fopen("./buffer_tmp/LAYER55.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER55.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(out_data_4, LAYER55_OUT_DIM, 1, write_ptr);
            fclose(write_ptr);
        }
    #endif

    // =================================================================================================================


    // LAYER56
    // 140468048981712_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER56\n");
    #endif

    riscv_nn_fc_s8_s8_s8_sym_bias(out_data_0, LAYER56_wt, LAYER56_IN_DIM, LAYER56_OUT_DIM, LAYER56_PRE_SHIFT, LAYER56_SCALAR, LAYER56_SHIFT, LAYER56_bias, buffer + 224, (q15_t*)col_buffer);

    #if DUMP_LAYER_ENABLE
        write_ptr = fopen("./buffer_tmp/LAYER56.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER56.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 224, LAYER56_OUT_DIM, 1, write_ptr);
            fclose(write_ptr);
        }
    #endif

    // =================================================================================================================


    // LAYER57
    // 140468049259088_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER57\n");
    #endif

    riscv_nn_sigmoid_s8(0, LAYER57_RANGE_RADIUS, LAYER57_SCALAR, LAYER57_LEFT_SHIFT, LAYER57_OUT_DIM, buffer + 224, out_data_1);

    #if DUMP_LAYER_ENABLE
        write_ptr = fopen("./buffer_tmp/LAYER57.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER57.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(out_data_1, LAYER57_OUT_DIM, 1, write_ptr);
            fclose(write_ptr);
        }
    #endif

    // =================================================================================================================


}
