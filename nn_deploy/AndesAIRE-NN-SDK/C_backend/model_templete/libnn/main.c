#include "nds_math_types.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "platform.h"
#include "perf_counter.h"
#include <string.h>
#include <stdlib.h>
#include <limits.h>

extern int enable_cache(void);
extern void run_nn(q7_t* in_data_0, q7_t* in_data_1, q7_t* in_data_2, q7_t* in_data_3, q7_t* out_data_0, q7_t* out_data_1, q7_t* out_data_2, q7_t* out_data_3, q7_t* out_data_4);

#ifdef CFG_NDS_RVV
#include "nds32_intrinsic.h"
void enable_vs()
{
    //Enable VPU
    uint32_t csr_mstatus;
    const uint32_t mask_vs = (3L << 9);
    const uint32_t vs_clean = (2L << 9);

    csr_mstatus = __nds__read_and_clear_csr(mask_vs, NDS_MSTATUS);
    csr_mstatus |=  vs_clean;
    __nds__write_csr(csr_mstatus, NDS_MSTATUS);

    //Read again to verify it.
    csr_mstatus = __nds__read_csr(NDS_MSTATUS);
}
#endif // CFG_NDS_RVV

#define num_data 1 // setting test data number
#define start_idx 0 // setting test data starting index
#define end_idx ((start_idx) + (num_data))

unsigned long long after_cycle, after_instret;

FILE *fptr;
size_t dump_read;
size_t dump_write;

int main()
{
    int num_samples = 0;
    q7_t *silence_table;

    q7_t input_0[1 * 42] = {0};
	q7_t input_1[1 * 24] = {0};
	q7_t input_2[1 * 48] = {0};
	q7_t input_3[1 * 96] = {0};


	q7_t output_0[1 * 24] = {0};
	q7_t output_1[1 *  1] = {0};
	q7_t output_2[1 * 48] = {0};
#ifdef LIBNN_TYPE_SYM
#else //ASYM
    memset(output_2, -128, sizeof(output_2));
#endif
	q7_t output_3[1 * 96] = {0};
	q7_t output_4[1 * 22] = {0};

#ifdef CFG_NDS_RVV
    enable_vs();
#endif // CFG_NDS_RVV

    enable_cache();

    // init_nn();

    char test_bin[256]    = {0};
    char sample_bin[256]  = {0};
    char silence_bin[256] = {0};
    char out_bin[256]     = {0};
    printf("Start executing ...\n");
    for (int i = start_idx; i < end_idx; i++)
    {
#ifdef LIBNN_TYPE_SYM
        snprintf(sample_bin, sizeof(sample_bin), "./input_data/sym/test_%d/num_samples.bin", i);
        snprintf(silence_bin, sizeof(silence_bin), "./input_data/sym/test_%d/silence_table.bin", i);
#else
        snprintf(sample_bin, sizeof(sample_bin), "./input_data/asym/test_%d/num_samples.bin", i);
        snprintf(silence_bin, sizeof(silence_bin), "./input_data/asym/test_%d/silence_table.bin", i);
#endif    
        if ((fptr = fopen(sample_bin, "rb")) == NULL)
        {
            printf("Fail to open file at %s !!!\n", sample_bin);
            exit(0);
        }
        dump_read = fread(&num_samples, sizeof(num_samples), 1, fptr);
        fclose(fptr);
        printf("i: %d , num_samples: %d\n", i, num_samples);

        silence_table = malloc((unsigned int) num_samples);

        if ((fptr = fopen(silence_bin, "rb")) == NULL)
        {
            printf("Fail to open file at %s !!!\n", silence_bin);
            exit(0);
        }
        dump_read = fread(silence_table, (size_t) num_samples, 1, fptr);
        fclose(fptr);

        for (int j = 0; j < num_samples; j++)
        {   
#ifdef LIBNN_TYPE_SYM    
            snprintf(test_bin, sizeof(test_bin), "./input_data/sym/test_%d/feature_%d.bin", i, j);
#else
            snprintf(test_bin, sizeof(test_bin), "./input_data/asym/test_%d/feature_%d.bin", i, j);
#endif
            snprintf(out_bin, sizeof(out_bin), "./output/out_%d_%d.bin", i, j);
                
            if (silence_table[j] != 1) {
                if ((fptr = fopen(test_bin, "rb")) == NULL)
                {
                    printf("Fail to open file at %s !!!\n", test_bin);
                    exit(0);
                }
                dump_read = fread(input_0, sizeof(input_0), 1, fptr);
                fclose(fptr);

                memcpy(input_1, output_0, sizeof(output_0));
                memcpy(input_2, output_2, sizeof(output_2));
                memcpy(input_3, output_3, sizeof(output_3));

                // prepare_nn(image_data);
                reset_perf_counter();
                run_nn(input_0, input_1, input_2, input_3, output_0, output_1, output_2, output_3, output_4);
                after_cycle = rdmcycle();
                after_instret = rdminstret();
#ifdef LIBNN_TYPE_SYM                
                printf("RNNoise-sym-int8 model: %lld instructions in %lld cycles\n", after_instret, after_cycle);
#else
                printf("RNNoise-asym-int8 model: %lld instructions in %lld cycles\n", after_instret, after_cycle);
#endif
                // finish_nn(output);

                if ((fptr = fopen(out_bin, "wb")) == NULL)
                {
                    printf("Fail to open file at %s !!!\n", out_bin);
                    exit(0);
                }
                dump_write = fwrite(output_4, sizeof(output_4), 1, fptr);
                fclose(fptr);
                // break;
            }
            else 
            {   
                printf("feature %d_%d silence!\n", i, j);
            }   
        }
    }

    printf("done\n");

    return 0;
}

