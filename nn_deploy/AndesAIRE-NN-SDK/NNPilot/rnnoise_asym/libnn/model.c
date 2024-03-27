#include "model.h"

// libnn version: nds_ver_310_master_8e018642
// andestfliteCompiler sha: N/A
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
    // 139723433430800_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER0\n");
    #endif

    riscv_nn_fc_s8_s8_s8_asym_bias(in_data_0, LAYER0_wt, LAYER0_IN_DIM, LAYER0_OUT_DIM, 1, 4, 0, LAYER0_SCALAR, LAYER0_SHIFT, -17, LAYER0_bias, buffer + 384, -128, 127, NULL);

    #if DUMP_LAYER_ENABLE
        write_ptr = fopen("./buffer_tmp/LAYER0.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER0.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 384, LAYER0_OUT_DIM, 1, write_ptr);
            fclose(write_ptr);
        }
    #endif

    // =================================================================================================================


    // LAYER1
    // 139723433413216_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER1\n");
    #endif

    riscv_nn_tanh_s8(17, LAYER1_RANGE_RADIUS, LAYER1_SCALAR, LAYER1_LEFT_SHIFT, LAYER1_OUT_DIM, buffer + 384, buffer + 384);

    #if DUMP_LAYER_ENABLE
        write_ptr = fopen("./buffer_tmp/LAYER1.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER1.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 384, LAYER1_OUT_DIM, 1, write_ptr);
            fclose(write_ptr);
        }
    #endif

    // =================================================================================================================


    // LAYER2
    // 139723433430992_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER2\n");
    #endif

    riscv_nn_fc_s8_s8_s8_asym_bias(buffer + 384, LAYER2_wt, LAYER2_IN_DIM, LAYER2_OUT_DIM, 1, 0, 0, LAYER2_SCALAR, LAYER2_SHIFT, -6, LAYER2_bias, buffer + 416, -128, 127, NULL);

    #if DUMP_LAYER_ENABLE
        write_ptr = fopen("./buffer_tmp/LAYER2.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER2.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 416, LAYER2_OUT_DIM, 1, write_ptr);
            fclose(write_ptr);
        }
    #endif

    // =================================================================================================================


    // LAYER3
    // torch.chunk_9_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER3\n");
    #endif

    riscv_nn_split_s8_x(buffer + 416, 72, 1, 1, 1, buffer + 272, 24, 0);
    riscv_nn_split_s8_x(buffer + 416, 72, 1, 1, 1, buffer + 336, 24, 24);
    riscv_nn_split_s8_x(buffer + 416, 72, 1, 1, 1, buffer + 240, 24, 48);

    #if DUMP_LAYER_ENABLE
        write_ptr = fopen("./buffer_tmp/LAYER3_split_0.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER3_split_0.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 272, 1 * 1 * 1 * 24, 1, write_ptr);
            fclose(write_ptr);
        }
        write_ptr = fopen("./buffer_tmp/LAYER3_split_1.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER3_split_1.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 336, 1 * 1 * 1 * 24, 1, write_ptr);
            fclose(write_ptr);
        }
        write_ptr = fopen("./buffer_tmp/LAYER3_split_2.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER3_split_2.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 240, 1 * 1 * 1 * 24, 1, write_ptr);
            fclose(write_ptr);
        }
    #endif

    // =================================================================================================================


    // LAYER4
    // 139723433430896_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER4\n");
    #endif

    riscv_nn_fc_s8_s8_s8_asym_bias(in_data_1, LAYER4_wt, LAYER4_IN_DIM, LAYER4_OUT_DIM, 1, 0, 0, LAYER4_SCALAR, LAYER4_SHIFT, -4, LAYER4_bias, buffer + 448, -128, 127, NULL);

    #if DUMP_LAYER_ENABLE
        write_ptr = fopen("./buffer_tmp/LAYER4.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER4.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 448, LAYER4_OUT_DIM, 1, write_ptr);
            fclose(write_ptr);
        }
    #endif

    // =================================================================================================================


    // LAYER5
    // torch.chunk_2copycopy_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER5\n");
    #endif

    riscv_nn_split_s8_x(buffer + 448, 72, 1, 1, 1, buffer + 304, 24, 0);
    riscv_nn_split_s8_x(buffer + 448, 72, 1, 1, 1, buffer + 528, 24, 24);
    riscv_nn_split_s8_x(buffer + 448, 72, 1, 1, 1, buffer + 416, 24, 48);

    #if DUMP_LAYER_ENABLE
        write_ptr = fopen("./buffer_tmp/LAYER5_split_0.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER5_split_0.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 304, 1 * 1 * 1 * 24, 1, write_ptr);
            fclose(write_ptr);
        }
        write_ptr = fopen("./buffer_tmp/LAYER5_split_1.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER5_split_1.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 528, 1 * 1 * 1 * 24, 1, write_ptr);
            fclose(write_ptr);
        }
        write_ptr = fopen("./buffer_tmp/LAYER5_split_2.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER5_split_2.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 416, 1 * 1 * 1 * 24, 1, write_ptr);
            fclose(write_ptr);
        }
    #endif

    // =================================================================================================================


    // LAYER6
    // torch.add_11_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER6\n");
    #endif

    riscv_nn_ew_add_s8_asym(buffer + 336, buffer + 528, 6, LAYER6_SCALAR1, LAYER6_SHIFT1, 4, LAYER6_SCALAR2, LAYER6_SHIFT2, 23, buffer + 448, -33, LAYER6_SCALARO, LAYER6_SHIFTO, -128, 127, LAYER6_OUT_DIM);

    #if DUMP_LAYER_ENABLE
        write_ptr = fopen("./buffer_tmp/LAYER6.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER6.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 448, LAYER6_OUT_DIM, 1, write_ptr);
            fclose(write_ptr);
        }
    #endif

    // =================================================================================================================


    // LAYER7
    // 139723433234832_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER7\n");
    #endif

    riscv_nn_sigmoid_s8(33, LAYER7_RANGE_RADIUS, LAYER7_SCALAR, LAYER7_LEFT_SHIFT, LAYER7_OUT_DIM, buffer + 448, buffer + 448);

    #if DUMP_LAYER_ENABLE
        write_ptr = fopen("./buffer_tmp/LAYER7.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER7.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 448, LAYER7_OUT_DIM, 1, write_ptr);
            fclose(write_ptr);
        }
    #endif

    // =================================================================================================================


    // LAYER8
    // torch.mul_18_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER8\n");
    #endif

    riscv_nn_ew_mul_s8_asym(buffer + 448, buffer + 416, 128, 4, buffer + 336, -15, LAYER8_SCALAR, LAYER8_SHIFT, -128, 127, LAYER8_OUT_DIM);

    #if DUMP_LAYER_ENABLE
        write_ptr = fopen("./buffer_tmp/LAYER8.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER8.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 336, LAYER8_OUT_DIM, 1, write_ptr);
            fclose(write_ptr);
        }
    #endif

    // =================================================================================================================


    // LAYER9
    // torch.add_26_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER9\n");
    #endif

    riscv_nn_ew_add_s8_asym(buffer + 272, buffer + 304, 6, LAYER9_SCALAR1, LAYER9_SHIFT1, 4, LAYER9_SCALAR2, LAYER9_SHIFT2, 23, buffer + 416, 3, LAYER9_SCALARO, LAYER9_SHIFTO, -128, 127, LAYER9_OUT_DIM);

    #if DUMP_LAYER_ENABLE
        write_ptr = fopen("./buffer_tmp/LAYER9.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER9.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 416, LAYER9_OUT_DIM, 1, write_ptr);
            fclose(write_ptr);
        }
    #endif

    // =================================================================================================================


    // LAYER10
    // 139723433234976_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER10\n");
    #endif

    riscv_nn_sigmoid_s8(-3, LAYER10_RANGE_RADIUS, LAYER10_SCALAR, LAYER10_LEFT_SHIFT, LAYER10_OUT_DIM, buffer + 416, buffer + 416);

    #if DUMP_LAYER_ENABLE
        write_ptr = fopen("./buffer_tmp/LAYER10.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER10.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 416, LAYER10_OUT_DIM, 1, write_ptr);
            fclose(write_ptr);
        }
    #endif

    // =================================================================================================================


    // LAYER11
    // torch.sub_32_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER11\n");
    #endif

    riscv_nn_ew_rsubc_s8_asym(buffer + 416, 1069547520, 128, LAYER11_SCALAR2, LAYER11_SHIFT2, 23, buffer + 272, -128, LAYER11_SCALARO, LAYER11_SHIFTO, -128, 127, LAYER11_OUT_DIM);

    #if DUMP_LAYER_ENABLE
        write_ptr = fopen("./buffer_tmp/LAYER11.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER11.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 272, LAYER11_OUT_DIM, 1, write_ptr);
            fclose(write_ptr);
        }
    #endif

    // =================================================================================================================


    // LAYER12
    // torch.add_22_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER12\n");
    #endif

    riscv_nn_ew_add_s8_asym(buffer + 240, buffer + 336, 6, LAYER12_SCALAR1, LAYER12_SHIFT1, 15, LAYER12_SCALAR2, LAYER12_SHIFT2, 23, buffer + 304, -7, LAYER12_SCALARO, LAYER12_SHIFTO, -128, 127, LAYER12_OUT_DIM);

    #if DUMP_LAYER_ENABLE
        write_ptr = fopen("./buffer_tmp/LAYER12.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER12.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 304, LAYER12_OUT_DIM, 1, write_ptr);
            fclose(write_ptr);
        }
    #endif

    // =================================================================================================================


    // LAYER13
    // 139723433236944_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER13\n");
    #endif

    riscv_nn_tanh_s8(7, LAYER13_RANGE_RADIUS, LAYER13_SCALAR, LAYER13_LEFT_SHIFT, LAYER13_OUT_DIM, buffer + 304, buffer + 304);

    #if DUMP_LAYER_ENABLE
        write_ptr = fopen("./buffer_tmp/LAYER13.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER13.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 304, LAYER13_OUT_DIM, 1, write_ptr);
            fclose(write_ptr);
        }
    #endif

    // =================================================================================================================


    // LAYER14
    // torch.mul_36_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER14\n");
    #endif

    riscv_nn_ew_mul_s8_asym(buffer + 416, in_data_1, 128, 0, buffer + 240, -1, LAYER14_SCALAR, LAYER14_SHIFT, -128, 127, LAYER14_OUT_DIM);

    #if DUMP_LAYER_ENABLE
        write_ptr = fopen("./buffer_tmp/LAYER14.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER14.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 240, LAYER14_OUT_DIM, 1, write_ptr);
            fclose(write_ptr);
        }
    #endif

    // =================================================================================================================


    // LAYER15
    // torch.mul_34_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER15\n");
    #endif

    riscv_nn_ew_mul_s8_asym(buffer + 272, buffer + 304, 128, 0, buffer + 336, -1, LAYER15_SCALAR, LAYER15_SHIFT, -128, 127, LAYER15_OUT_DIM);

    #if DUMP_LAYER_ENABLE
        write_ptr = fopen("./buffer_tmp/LAYER15.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER15.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 336, LAYER15_OUT_DIM, 1, write_ptr);
            fclose(write_ptr);
        }
    #endif

    // =================================================================================================================


    // LAYER16
    // torch.add_38_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER16\n");
    #endif

    riscv_nn_ew_add_s8_asym(buffer + 240, buffer + 336, 1, LAYER16_SCALAR1, LAYER16_SHIFT1, 1, LAYER16_SCALAR2, LAYER16_SHIFT2, 23, out_data_0, 0, LAYER16_SCALARO, LAYER16_SHIFTO, -128, 127, LAYER16_OUT_DIM);

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
    // 139723433430320_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER17\n");
    #endif

    riscv_nn_fc_s8_s8_s8_asym_bias(in_data_2, LAYER17_wt, LAYER17_IN_DIM, LAYER17_OUT_DIM, 1, 128, 0, LAYER17_SCALAR, LAYER17_SHIFT, -9, LAYER17_bias, buffer + 416, -128, 127, NULL);

    #if DUMP_LAYER_ENABLE
        write_ptr = fopen("./buffer_tmp/LAYER17.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER17.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 416, LAYER17_OUT_DIM, 1, write_ptr);
            fclose(write_ptr);
        }
    #endif

    // =================================================================================================================


    // LAYER18
    // torch.chunk_2_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER18\n");
    #endif

    riscv_nn_split_s8_x(buffer + 416, 144, 1, 1, 1, buffer + 288, 48, 0);
    riscv_nn_split_s8_x(buffer + 416, 144, 1, 1, 1, buffer + 336, 48, 48);
    riscv_nn_split_s8_x(buffer + 416, 144, 1, 1, 1, buffer + 240, 48, 96);

    #if DUMP_LAYER_ENABLE
        write_ptr = fopen("./buffer_tmp/LAYER18_split_0.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER18_split_0.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 288, 1 * 1 * 1 * 48, 1, write_ptr);
            fclose(write_ptr);
        }
        write_ptr = fopen("./buffer_tmp/LAYER18_split_1.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER18_split_1.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 336, 1 * 1 * 1 * 48, 1, write_ptr);
            fclose(write_ptr);
        }
        write_ptr = fopen("./buffer_tmp/LAYER18_split_2.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER18_split_2.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 240, 1 * 1 * 1 * 48, 1, write_ptr);
            fclose(write_ptr);
        }
    #endif

    // =================================================================================================================


    // LAYER19
    // 139728129448832_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER19\n");
    #endif

    memset(buffer + 416, -4, LAYER19_OUT_DIM);
    riscv_nn_ew_add_s8_asym(in_data_0, buffer + 416, 4, LAYER19_SCALAR1, LAYER19_SHIFT1, 4, LAYER19_SCALAR2, LAYER19_SHIFT2, 23, buffer + 480, -52, LAYER19_SCALARO, LAYER19_SHIFTO, -128, 127, LAYER19_OUT_DIM);

    #if DUMP_LAYER_ENABLE
        write_ptr = fopen("./buffer_tmp/LAYER19.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER19.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 480, LAYER19_OUT_DIM, 1, write_ptr);
            fclose(write_ptr);
        }
    #endif

    // =================================================================================================================


    // LAYER20
    // 139728129447008_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER20\n");
    #endif

    memset(buffer + 416, 0, LAYER20_OUT_DIM);
    riscv_nn_ew_add_s8_asym(buffer + 384, buffer + 416, 0, LAYER20_SCALAR1, LAYER20_SHIFT1, 0, LAYER20_SCALAR2, LAYER20_SHIFT2, 23, buffer + 528, -52, LAYER20_SCALARO, LAYER20_SHIFTO, -128, 127, LAYER20_OUT_DIM);

    #if DUMP_LAYER_ENABLE
        write_ptr = fopen("./buffer_tmp/LAYER20.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER20.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 528, LAYER20_OUT_DIM, 1, write_ptr);
            fclose(write_ptr);
        }
    #endif

    // =================================================================================================================


    // LAYER21
    // 139728129447392_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER21\n");
    #endif

    memset(buffer + 384, 0, LAYER21_OUT_DIM);
    riscv_nn_ew_add_s8_asym(out_data_0, buffer + 384, 0, LAYER21_SCALAR1, LAYER21_SHIFT1, 0, LAYER21_SCALAR2, LAYER21_SHIFT2, 23, buffer + 560, -52, LAYER21_SCALARO, LAYER21_SHIFTO, -128, 127, LAYER21_OUT_DIM);

    #if DUMP_LAYER_ENABLE
        write_ptr = fopen("./buffer_tmp/LAYER21.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER21.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 560, LAYER21_OUT_DIM, 1, write_ptr);
            fclose(write_ptr);
        }
    #endif

    // =================================================================================================================


    // LAYER22
    // torch.cat_49_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER22\n");
    #endif

    riscv_nn_concate_s8_x(buffer + 528, 24, 1, 1, 1, buffer + 384, 90, 0);
    riscv_nn_concate_s8_x(buffer + 560, 24, 1, 1, 1, buffer + 384, 90, 24);
    riscv_nn_concate_s8_x(buffer + 480, 42, 1, 1, 1, buffer + 384, 90, 48);

    #if DUMP_LAYER_ENABLE
        write_ptr = fopen("./buffer_tmp/LAYER22.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER22.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 384, LAYER22_OUT_DIM_X * LAYER22_OUT_DIM_Y * LAYER22_OUT_CH, 1, write_ptr);
            fclose(write_ptr);
        }
    #endif

    // =================================================================================================================


    // LAYER23
    // 139723433411968_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER23\n");
    #endif

    riscv_nn_fc_s8_s8_s8_asym_bias(buffer + 384, LAYER23_wt, LAYER23_IN_DIM, LAYER23_OUT_DIM, 1, 52, 0, LAYER23_SCALAR, LAYER23_SHIFT, -9, LAYER23_bias, buffer + 480, -128, 127, NULL);

    #if DUMP_LAYER_ENABLE
        write_ptr = fopen("./buffer_tmp/LAYER23.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER23.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 480, LAYER23_OUT_DIM, 1, write_ptr);
            fclose(write_ptr);
        }
    #endif

    // =================================================================================================================


    // LAYER24
    // torch.chunk_55_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER24\n");
    #endif

    riscv_nn_split_s8_x(buffer + 480, 144, 1, 1, 1, buffer + 432, 48, 0);
    riscv_nn_split_s8_x(buffer + 480, 144, 1, 1, 1, buffer + 624, 48, 48);
    riscv_nn_split_s8_x(buffer + 480, 144, 1, 1, 1, buffer + 384, 48, 96);

    #if DUMP_LAYER_ENABLE
        write_ptr = fopen("./buffer_tmp/LAYER24_split_0.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER24_split_0.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 432, 1 * 1 * 1 * 48, 1, write_ptr);
            fclose(write_ptr);
        }
        write_ptr = fopen("./buffer_tmp/LAYER24_split_1.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER24_split_1.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 624, 1 * 1 * 1 * 48, 1, write_ptr);
            fclose(write_ptr);
        }
        write_ptr = fopen("./buffer_tmp/LAYER24_split_2.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER24_split_2.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 384, 1 * 1 * 1 * 48, 1, write_ptr);
            fclose(write_ptr);
        }
    #endif

    // =================================================================================================================


    // LAYER25
    // torch.add_57_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER25\n");
    #endif

    riscv_nn_ew_add_s8_asym(buffer + 624, buffer + 336, 9, LAYER25_SCALAR1, LAYER25_SHIFT1, 9, LAYER25_SCALAR2, LAYER25_SHIFT2, 23, buffer + 480, -3, LAYER25_SCALARO, LAYER25_SHIFTO, -128, 127, LAYER25_OUT_DIM);

    #if DUMP_LAYER_ENABLE
        write_ptr = fopen("./buffer_tmp/LAYER25.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER25.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 480, LAYER25_OUT_DIM, 1, write_ptr);
            fclose(write_ptr);
        }
    #endif

    // =================================================================================================================


    // LAYER26
    // 139723433235216_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER26\n");
    #endif

    riscv_nn_sigmoid_s8(3, LAYER26_RANGE_RADIUS, LAYER26_SCALAR, LAYER26_LEFT_SHIFT, LAYER26_OUT_DIM, buffer + 480, buffer + 480);

    #if DUMP_LAYER_ENABLE
        write_ptr = fopen("./buffer_tmp/LAYER26.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER26.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 480, LAYER26_OUT_DIM, 1, write_ptr);
            fclose(write_ptr);
        }
    #endif

    // =================================================================================================================


    // LAYER27
    // torch.add_68_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER27\n");
    #endif

    riscv_nn_ew_add_s8_asym(buffer + 432, buffer + 288, 9, LAYER27_SCALAR1, LAYER27_SHIFT1, 9, LAYER27_SCALAR2, LAYER27_SHIFT2, 23, buffer + 336, 6, LAYER27_SCALARO, LAYER27_SHIFTO, -128, 127, LAYER27_OUT_DIM);

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
    // 139723433235312_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER28\n");
    #endif

    riscv_nn_sigmoid_s8(-6, LAYER28_RANGE_RADIUS, LAYER28_SCALAR, LAYER28_LEFT_SHIFT, LAYER28_OUT_DIM, buffer + 336, buffer + 336);

    #if DUMP_LAYER_ENABLE
        write_ptr = fopen("./buffer_tmp/LAYER28.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER28.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 336, LAYER28_OUT_DIM, 1, write_ptr);
            fclose(write_ptr);
        }
    #endif

    // =================================================================================================================


    // LAYER29
    // torch.mul_64_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER29\n");
    #endif

    riscv_nn_ew_mul_s8_asym(buffer + 480, buffer + 240, 128, 9, buffer + 288, -5, LAYER29_SCALAR, LAYER29_SHIFT, -128, 127, LAYER29_OUT_DIM);

    #if DUMP_LAYER_ENABLE
        write_ptr = fopen("./buffer_tmp/LAYER29.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER29.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 288, LAYER29_OUT_DIM, 1, write_ptr);
            fclose(write_ptr);
        }
    #endif

    // =================================================================================================================


    // LAYER30
    // torch.sub_74_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER30\n");
    #endif

    riscv_nn_ew_rsubc_s8_asym(buffer + 336, 1069547520, 128, LAYER30_SCALAR2, LAYER30_SHIFT2, 23, buffer + 240, -128, LAYER30_SCALARO, LAYER30_SHIFTO, -128, 127, LAYER30_OUT_DIM);

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

    riscv_nn_ew_add_s8_asym(buffer + 384, buffer + 288, 9, LAYER31_SCALAR1, LAYER31_SHIFT1, 5, LAYER31_SCALAR2, LAYER31_SHIFT2, 23, buffer + 432, -128, LAYER31_SCALARO, LAYER31_SHIFTO, -128, 127, LAYER31_OUT_DIM);

    #if DUMP_LAYER_ENABLE
        write_ptr = fopen("./buffer_tmp/LAYER31.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER31.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 432, LAYER31_OUT_DIM, 1, write_ptr);
            fclose(write_ptr);
        }
    #endif

    // =================================================================================================================


    // LAYER32
    // torch.mul_82_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER32\n");
    #endif

    riscv_nn_ew_mul_s8_asym(buffer + 240, buffer + 432, 128, 128, buffer + 384, -128, LAYER32_SCALAR, LAYER32_SHIFT, -128, 127, LAYER32_OUT_DIM);

    #if DUMP_LAYER_ENABLE
        write_ptr = fopen("./buffer_tmp/LAYER32.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER32.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 384, LAYER32_OUT_DIM, 1, write_ptr);
            fclose(write_ptr);
        }
    #endif

    // =================================================================================================================


    // LAYER33
    // 139728128855344_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER33\n");
    #endif

    memset(buffer + 240, -4, LAYER33_OUT_DIM);
    riscv_nn_ew_add_s8_asym(in_data_0, buffer + 240, 4, LAYER33_SCALAR1, LAYER33_SHIFT1, 4, LAYER33_SCALAR2, LAYER33_SHIFT2, 23, buffer + 288, 38, LAYER33_SCALARO, LAYER33_SHIFTO, -128, 127, LAYER33_OUT_DIM);

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
    // torch.mul_76_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER34\n");
    #endif

    riscv_nn_ew_mul_s8_asym(buffer + 336, in_data_2, 128, 128, buffer + 192, -128, LAYER34_SCALAR, LAYER34_SHIFT, -128, 127, LAYER34_OUT_DIM);

    #if DUMP_LAYER_ENABLE
        write_ptr = fopen("./buffer_tmp/LAYER34.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER34.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 192, LAYER34_OUT_DIM, 1, write_ptr);
            fclose(write_ptr);
        }
    #endif

    // =================================================================================================================


    // LAYER35
    // torch.add_84_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER35\n");
    #endif

    riscv_nn_ew_add_s8_asym(buffer + 192, buffer + 384, 128, LAYER35_SCALAR1, LAYER35_SHIFT1, 128, LAYER35_SCALAR2, LAYER35_SHIFT2, 23, out_data_2, -128, LAYER35_SCALARO, LAYER35_SHIFTO, -128, 127, LAYER35_OUT_DIM);

    #if DUMP_LAYER_ENABLE
        write_ptr = fopen("./buffer_tmp/LAYER35.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER35.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(out_data_2, LAYER35_OUT_DIM, 1, write_ptr);
            fclose(write_ptr);
        }
    #endif

    // =================================================================================================================


    // LAYER36
    // 139723433430416_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER36\n");
    #endif

    riscv_nn_fc_s8_s8_s8_asym_bias(in_data_3, LAYER36_wt, LAYER36_IN_DIM, LAYER36_OUT_DIM, 1, 0, 0, LAYER36_SCALAR, LAYER36_SHIFT, -15, LAYER36_bias, buffer + 576, -128, 127, NULL);

    #if DUMP_LAYER_ENABLE
        write_ptr = fopen("./buffer_tmp/LAYER36.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER36.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 576, LAYER36_OUT_DIM, 1, write_ptr);
            fclose(write_ptr);
        }
    #endif

    // =================================================================================================================


    // LAYER37
    // torch.chunk_2copy_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER37\n");
    #endif

    riscv_nn_split_s8_x(buffer + 576, 288, 1, 1, 1, buffer + 192, 96, 0);
    riscv_nn_split_s8_x(buffer + 576, 288, 1, 1, 1, buffer + 480, 96, 96);
    riscv_nn_split_s8_x(buffer + 576, 288, 1, 1, 1, buffer + 384, 96, 192);

    #if DUMP_LAYER_ENABLE
        write_ptr = fopen("./buffer_tmp/LAYER37_split_0.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER37_split_0.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 192, 1 * 1 * 1 * 96, 1, write_ptr);
            fclose(write_ptr);
        }
        write_ptr = fopen("./buffer_tmp/LAYER37_split_1.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER37_split_1.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 480, 1 * 1 * 1 * 96, 1, write_ptr);
            fclose(write_ptr);
        }
        write_ptr = fopen("./buffer_tmp/LAYER37_split_2.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER37_split_2.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 384, 1 * 1 * 1 * 96, 1, write_ptr);
            fclose(write_ptr);
        }
    #endif

    // =================================================================================================================


    // LAYER38
    // 139728128853376_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER38\n");
    #endif

    memset(buffer + 576, 0, LAYER38_OUT_DIM);
    riscv_nn_ew_add_s8_asym(out_data_0, buffer + 576, 0, LAYER38_SCALAR1, LAYER38_SHIFT1, 0, LAYER38_SCALAR2, LAYER38_SHIFT2, 23, buffer + 336, 38, LAYER38_SCALARO, LAYER38_SHIFTO, -128, 127, LAYER38_OUT_DIM);

    #if DUMP_LAYER_ENABLE
        write_ptr = fopen("./buffer_tmp/LAYER38.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER38.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 336, LAYER38_OUT_DIM, 1, write_ptr);
            fclose(write_ptr);
        }
    #endif

    // =================================================================================================================


    // LAYER39
    // 139728128855008_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER39\n");
    #endif

    memset(buffer + 576, -128, LAYER39_OUT_DIM);
    riscv_nn_ew_add_s8_asym(out_data_2, buffer + 576, 128, LAYER39_SCALAR1, LAYER39_SHIFT1, 128, LAYER39_SCALAR2, LAYER39_SHIFT2, 23, buffer + 704, 38, LAYER39_SCALARO, LAYER39_SHIFTO, -128, 127, LAYER39_OUT_DIM);

    #if DUMP_LAYER_ENABLE
        write_ptr = fopen("./buffer_tmp/LAYER39.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER39.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 704, LAYER39_OUT_DIM, 1, write_ptr);
            fclose(write_ptr);
        }
    #endif

    // =================================================================================================================


    // LAYER40
    // torch.cat_90_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER40\n");
    #endif

    riscv_nn_concate_s8_x(buffer + 336, 24, 1, 1, 1, buffer + 576, 114, 0);
    riscv_nn_concate_s8_x(buffer + 704, 48, 1, 1, 1, buffer + 576, 114, 24);
    riscv_nn_concate_s8_x(buffer + 288, 42, 1, 1, 1, buffer + 576, 114, 72);

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
    // 139723433412352_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER41\n");
    #endif

    riscv_nn_fc_s8_s8_s8_asym_bias(buffer + 576, LAYER41_wt, LAYER41_IN_DIM, LAYER41_OUT_DIM, 1, -38, 0, LAYER41_SCALAR, LAYER41_SHIFT, -20, LAYER41_bias, buffer + 768, -128, 127, NULL);

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

    riscv_nn_split_s8_x(buffer + 768, 288, 1, 1, 1, buffer + 288, 96, 0);
    riscv_nn_split_s8_x(buffer + 768, 288, 1, 1, 1, buffer + 672, 96, 96);
    riscv_nn_split_s8_x(buffer + 768, 288, 1, 1, 1, buffer + 576, 96, 192);

    #if DUMP_LAYER_ENABLE
        write_ptr = fopen("./buffer_tmp/LAYER42_split_0.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER42_split_0.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 288, 1 * 1 * 1 * 96, 1, write_ptr);
            fclose(write_ptr);
        }
        write_ptr = fopen("./buffer_tmp/LAYER42_split_1.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER42_split_1.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 672, 1 * 1 * 1 * 96, 1, write_ptr);
            fclose(write_ptr);
        }
        write_ptr = fopen("./buffer_tmp/LAYER42_split_2.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER42_split_2.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 576, 1 * 1 * 1 * 96, 1, write_ptr);
            fclose(write_ptr);
        }
    #endif

    // =================================================================================================================


    // LAYER43
    // torch.add_98_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER43\n");
    #endif

    riscv_nn_ew_add_s8_asym(buffer + 672, buffer + 480, 20, LAYER43_SCALAR1, LAYER43_SHIFT1, 15, LAYER43_SCALAR2, LAYER43_SHIFT2, 23, buffer + 768, -25, LAYER43_SCALARO, LAYER43_SHIFTO, -128, 127, LAYER43_OUT_DIM);

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
    // 139723433235408_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER44\n");
    #endif

    riscv_nn_sigmoid_s8(25, LAYER44_RANGE_RADIUS, LAYER44_SCALAR, LAYER44_LEFT_SHIFT, LAYER44_OUT_DIM, buffer + 768, buffer + 768);

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
    // torch.mul_105_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER45\n");
    #endif

    riscv_nn_ew_mul_s8_asym(buffer + 768, buffer + 384, 128, 15, buffer + 480, -14, LAYER45_SCALAR, LAYER45_SHIFT, -128, 127, LAYER45_OUT_DIM);

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
    // torch.add_109_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER46\n");
    #endif

    riscv_nn_ew_add_s8_asym(buffer + 576, buffer + 480, 20, LAYER46_SCALAR1, LAYER46_SHIFT1, 14, LAYER46_SCALAR2, LAYER46_SHIFT2, 23, buffer + 384, 0, LAYER46_SCALARO, LAYER46_SHIFTO, -128, 127, LAYER46_OUT_DIM);

    #if DUMP_LAYER_ENABLE
        write_ptr = fopen("./buffer_tmp/LAYER46.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER46.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 384, LAYER46_OUT_DIM, 1, write_ptr);
            fclose(write_ptr);
        }
    #endif

    // =================================================================================================================


    // LAYER47
    // 139723433236656_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER47\n");
    #endif

    riscv_nn_tanh_s8(0, LAYER47_RANGE_RADIUS, LAYER47_SCALAR, LAYER47_LEFT_SHIFT, LAYER47_OUT_DIM, buffer + 384, buffer + 384);

    #if DUMP_LAYER_ENABLE
        write_ptr = fopen("./buffer_tmp/LAYER47.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER47.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 384, LAYER47_OUT_DIM, 1, write_ptr);
            fclose(write_ptr);
        }
    #endif

    // =================================================================================================================


    // LAYER48
    // torch.add_113_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER48\n");
    #endif

    riscv_nn_ew_add_s8_asym(buffer + 288, buffer + 192, 20, LAYER48_SCALAR1, LAYER48_SHIFT1, 15, LAYER48_SCALAR2, LAYER48_SHIFT2, 23, buffer + 480, -5, LAYER48_SCALARO, LAYER48_SHIFTO, -128, 127, LAYER48_OUT_DIM);

    #if DUMP_LAYER_ENABLE
        write_ptr = fopen("./buffer_tmp/LAYER48.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER48.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 480, LAYER48_OUT_DIM, 1, write_ptr);
            fclose(write_ptr);
        }
    #endif

    // =================================================================================================================


    // LAYER49
    // 139723433235504_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER49\n");
    #endif

    riscv_nn_sigmoid_s8(5, LAYER49_RANGE_RADIUS, LAYER49_SCALAR, LAYER49_LEFT_SHIFT, LAYER49_OUT_DIM, buffer + 480, buffer + 480);

    #if DUMP_LAYER_ENABLE
        write_ptr = fopen("./buffer_tmp/LAYER49.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER49.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 480, LAYER49_OUT_DIM, 1, write_ptr);
            fclose(write_ptr);
        }
    #endif

    // =================================================================================================================


    // LAYER50
    // torch.sub_119_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER50\n");
    #endif

    riscv_nn_ew_rsubc_s8_asym(buffer + 480, 1069547520, 128, LAYER50_SCALAR2, LAYER50_SHIFT2, 23, buffer + 288, -128, LAYER50_SCALARO, LAYER50_SHIFTO, -128, 127, LAYER50_OUT_DIM);

    #if DUMP_LAYER_ENABLE
        write_ptr = fopen("./buffer_tmp/LAYER50.bin", "wb");
        if (write_ptr == NULL) {
            printf("Error opening file for writing: LAYER50.bin\n");
            // Handle the error or exit the program
        } else {
            fwrite(buffer + 288, LAYER50_OUT_DIM, 1, write_ptr);
            fclose(write_ptr);
        }
    #endif

    // =================================================================================================================


    // LAYER51
    // torch.mul_121_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER51\n");
    #endif

    riscv_nn_ew_mul_s8_asym(buffer + 288, buffer + 384, 128, 0, buffer + 192, -1, LAYER51_SCALAR, LAYER51_SHIFT, -128, 127, LAYER51_OUT_DIM);

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
    // torch.mul_123_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER52\n");
    #endif

    riscv_nn_ew_mul_s8_asym(buffer + 480, in_data_3, 128, 0, buffer + 288, -1, LAYER52_SCALAR, LAYER52_SHIFT, -128, 127, LAYER52_OUT_DIM);

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

    riscv_nn_ew_add_s8_asym(buffer + 288, buffer + 192, 1, LAYER53_SCALAR1, LAYER53_SHIFT1, 1, LAYER53_SCALAR2, LAYER53_SHIFT2, 23, out_data_3, 0, LAYER53_SCALARO, LAYER53_SHIFTO, -128, 127, LAYER53_OUT_DIM);

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
    // 139723433412688_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER54\n");
    #endif

    riscv_nn_fc_s8_s8_s8_asym_bias(out_data_3, LAYER54_wt, LAYER54_IN_DIM, LAYER54_OUT_DIM, 1, 0, 0, LAYER54_SCALAR, LAYER54_SHIFT, -49, LAYER54_bias, buffer + 192, -128, 127, NULL);

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
    // 139723433235600_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER55\n");
    #endif

    riscv_nn_sigmoid_s8(49, LAYER55_RANGE_RADIUS, LAYER55_SCALAR, LAYER55_LEFT_SHIFT, LAYER55_OUT_DIM, buffer + 192, out_data_4);

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
    // 139723433430752_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER56\n");
    #endif

    riscv_nn_fc_s8_s8_s8_asym_bias(out_data_0, LAYER56_wt, LAYER56_IN_DIM, LAYER56_OUT_DIM, 1, 0, 0, LAYER56_SCALAR, LAYER56_SHIFT, -8, LAYER56_bias, buffer + 224, -128, 127, NULL);

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
    // 139723433235120_out
    #if DUMP_LAYER_ENABLE
        printf("LAYER57\n");
    #endif

    riscv_nn_sigmoid_s8(8, LAYER57_RANGE_RADIUS, LAYER57_SCALAR, LAYER57_LEFT_SHIFT, LAYER57_OUT_DIM, buffer + 224, out_data_1);

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
