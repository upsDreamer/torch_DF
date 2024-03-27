import numpy as np
import copy
from scipy import stats
from numba import jit

@jit(nopython=True,fastmath=True)
def speed_up_sum(NParray):
    summing = 0
    for i in range(len(NParray)):
        summing += NParray[i]
    return summing

@jit(nopython=True,fastmath=True)
def merge_hist(target_bin, sliced_nd_hist, quantized_bins, num_merged_bins):
    for j in range(target_bin): # original target_bin
        start = j * num_merged_bins
        stop = start + num_merged_bins
        quantized_bins[j] = speed_up_sum(sliced_nd_hist[start:stop])

@jit(nopython=True,fastmath=True)        
def expand_hist(target_bin, num_merged_bins, is_nonzeros, q, quantized_bins):
    for j in range(target_bin): # original target_bin
        start = j * num_merged_bins
        if j == target_bin - 1:
            stop = -1
        else:
            stop = start + num_merged_bins
        norm = speed_up_sum(is_nonzeros[start:stop])
        if norm != 0:
            q[start:stop] = float(quantized_bins[j]) / float(norm)


def threshold_distribution(distribution, target_bin=128):
    """
    Return the best threshold value. 
    Ref: https://github.com//apache/incubator-mxnet/blob/master/python/mxnet/contrib/quantization.py
    Args:
        distribution: list, activations has been processed by histogram and normalize,size is 2048
        target_bin: int, the num of bin that is used by quantize, Int8 default value is 128
    Returns:
        target_threshold: int, num of bin with the minimum KL 
    """   
    distribution = distribution[1:] # It seems that the line is needed for finding a reasonable threshold.
    length = distribution.size
    threshold_sum = sum(distribution[target_bin:])
    kl_divergence = np.zeros(length - target_bin)

    for threshold in range(target_bin, length):
        sliced_nd_hist = copy.deepcopy(distribution[:threshold])

        # generate reference distribution p
        p = sliced_nd_hist.copy()
        p[threshold-1] += threshold_sum
        threshold_sum = threshold_sum - distribution[threshold]

        # is_nonzeros[k] indicates whether hist[k] is nonzero
        is_nonzeros = (p != 0).astype(np.int64)
        # 
        quantized_bins = np.zeros(target_bin, dtype=np.int64)
        # calculate how many bins should be merged to generate quantized distribution q
        num_merged_bins = sliced_nd_hist.size // target_bin
        
        # merge hist into num_quantized_bins bins
        merge_hist(target_bin, sliced_nd_hist, quantized_bins, num_merged_bins)
        quantized_bins[-1] += speed_up_sum(sliced_nd_hist[target_bin * num_merged_bins:])
        
        # expand quantized_bins into p.size bins
        q = np.zeros(sliced_nd_hist.size, dtype=np.float64)
        expand_hist(target_bin, num_merged_bins, is_nonzeros, q, quantized_bins)
        q[p == 0] = 0
        # p = _smooth_distribution(p) # with some bugs, need to fix
        # q = _smooth_distribution(q)
        p[p == 0] = 0.0001
        q[q == 0] = 0.0001
        
        # calculate kl_divergence between q and p
        kl_divergence[threshold - target_bin] = stats.entropy(p, q)

    min_kl_divergence = np.argmin(kl_divergence)
    threshold_value = min_kl_divergence + target_bin

    return threshold_value


# Approximate version 
def apprx_bounds_distribution(distribution, target_bin = 256):
    distribution = distribution[1:]
    length = distribution.size
    upper_sum = sum(distribution[target_bin:])
    lower_sum = 0
    kl_divergence = np.zeros(length - target_bin)
    
    for upper in range(target_bin, length):
        sliced_nd_hist = copy.deepcopy(distribution[:upper])

        # generate reference distribution p
        p = sliced_nd_hist.copy()
        p[upper-1] += upper_sum
        upper_sum = upper_sum - distribution[upper]

        # is_nonzeros[k] indicates whether hist[k] is nonzero
        is_nonzeros = (p != 0).astype(np.int64)
        # 
        quantized_bins = np.zeros(target_bin, dtype=np.int64)
        # calculate how many bins should be merged to generate quantized distribution q
        num_merged_bins = sliced_nd_hist.size // target_bin
        
        # merge hist into num_quantized_bins bins
        merge_hist(target_bin, sliced_nd_hist, quantized_bins, num_merged_bins)
        quantized_bins[-1] += speed_up_sum(sliced_nd_hist[target_bin * num_merged_bins:]) 
        # expand quantized_bins into p.size bins
        q = np.zeros(sliced_nd_hist.size, dtype=np.float64)
        expand_hist(target_bin, num_merged_bins, is_nonzeros, q, quantized_bins)
        q[p == 0] = 0
        # p = _smooth_distribution(p) # with some bugs, need to fix
        # q = _smooth_distribution(q)
        p[p == 0] = 0.0001
        q[q == 0] = 0.0001
        
        # calculate kl_divergence between q and p
        kl_divergence[upper - target_bin] = stats.entropy(p, q)

    min_kl_divergence = np.argmin(kl_divergence)
    upper_range = min_kl_divergence + target_bin
    kl_divergence = None
    del kl_divergence
    
    if upper_range > target_bin:
        kl_divergence = np.zeros(upper_range - target_bin)
        for lower in range(0, upper_range-target_bin):
            sliced_nd_hist = copy.deepcopy(distribution[lower:upper_range])
    
            # generate reference distribution p
            p = sliced_nd_hist.copy()
            p[0] += lower_sum
            lower_sum = lower_sum + distribution[lower]
    
            # is_nonzeros[k] indicates whether hist[k] is nonzero
            is_nonzeros = (p != 0).astype(np.int64)
            # 
            quantized_bins = np.zeros(target_bin, dtype=np.int64)
            # calculate how many bins should be merged to generate quantized distribution q
            num_merged_bins = sliced_nd_hist.size // target_bin
            
            # merge hist into num_quantized_bins bins
            merge_hist(target_bin, sliced_nd_hist, quantized_bins, num_merged_bins)
            quantized_bins[-1] += speed_up_sum(sliced_nd_hist[target_bin * num_merged_bins:]) 
            # expand quantized_bins into p.size bins
            q = np.zeros(sliced_nd_hist.size, dtype=np.float64)
            expand_hist(target_bin, num_merged_bins, is_nonzeros, q, quantized_bins)
            q[p == 0] = 0
            # p = _smooth_distribution(p) # with some bugs, need to fix
            # q = _smooth_distribution(q)
            p[p == 0] = 0.0001
            q[q == 0] = 0.0001
            
            # calculate kl_divergence between q and p
            kl_divergence[lower] = stats.entropy(p, q)
    
        min_kl_divergence = np.argmin(kl_divergence)
        lower_range = min_kl_divergence
    else:
        lower_range = 0
    
    return lower_range, upper_range

# mind the boundary condition
def bounds_distribution(distribution, target_bin = 256):
    distribution = distribution[1:]
    length = distribution.size  
    min_kl_divergence = np.finfo(np.float32).max
    upper_value = 2047
    lower_value = 0
    
    upper_sum = sum(distribution[target_bin:])
    for upper_bound in range(target_bin, length):
        lower_sum = 0#sum(distribution[:lower_bound])#
        #print(upper_bound)
        for lower_bound in range(0,(upper_bound - target_bin)):
            sliced_nd_hist = copy.deepcopy(distribution[lower_bound:upper_bound])

            # generate reference distribution p
            p = sliced_nd_hist.copy()
            p[len(p)-1] += upper_sum # I had take away upper_bound, instead i use len(p).
            p[0] += lower_sum # I had take away lower_bound, instead i use 0.
            lower_sum = lower_sum + distribution[lower_bound] 
    
            # is_nonzeros[k] indicates whether hist[k] is nonzero
            is_nonzeros = (p != 0).astype(np.int64)
            quantized_bins = np.zeros(target_bin, dtype=np.int64) 
            # calculate how many bins should be merged to generate quantized distribution q
            num_merged_bins = sliced_nd_hist.size // target_bin
            
            # merge hist into num_quantized_bins bins
            merge_hist(target_bin, sliced_nd_hist, quantized_bins, num_merged_bins)
            quantized_bins[-1] += speed_up_sum(sliced_nd_hist[upper_bound * num_merged_bins:]) 
            
            # expand quantized_bins into p.size bins
            q = np.zeros(sliced_nd_hist.size, dtype=np.float64)
            expand_hist(target_bin, num_merged_bins, is_nonzeros, q, quantized_bins)
            q[p == 0] = 0          
            p[p == 0] = 0.0001
            q[q == 0] = 0.0001
            
            # calculate kl_divergence between q and p
            kl_divergence = stats.entropy(p, q)
            if (kl_divergence < min_kl_divergence):
                min_kl_divergence = kl_divergence
                upper_value = upper_bound
                lower_value = lower_bound
        upper_sum = upper_sum - distribution[upper_bound]
    return lower_value, upper_value