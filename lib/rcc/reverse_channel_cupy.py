import cupy as cp
import numpy as np

# Load the CUDA module
cuda_code = open('lib/rcc/reverse_channel_cuda.cu', 'r').read()
cuda_module = cp.RawModule(code=cuda_code)

# Get the kernel functions
reverse_channel_encode_kernel = cuda_module.get_function("reverse_channel_encode_kernel")
generate_sample_kernel = cuda_module.get_function("generate_sample_kernel")

def generate_sample(dim, shared_seed, sample_seed):
    sample_out = cp.empty(dim, dtype=cp.float32)
    
    generate_sample_kernel(
        (1, 1, 1),
        (1, 1, 1),
        (cp.int32(dim),
         cp.uint64(shared_seed),
         cp.uint64(sample_seed),
         sample_out)
    )
    
    return sample_out.get()

def reverse_channel_encode_cupy(mu_q_in, K, shared_seed=0):
    mu_q = cp.asarray(mu_q_in, dtype=cp.float32)
    dim = mu_q.shape[0]

    # Allocate memory on GPU
    log_w = cp.empty(K, dtype=cp.float32)
    max_log_w = cp.array([-cp.inf], dtype=cp.float32)

    # Set up grid and block dimensions
    block_size = 256
    grid_size = (K + block_size - 1) // block_size

    # Generate vector of random exponentials
    t = cp.random.exponential(scale=1.0, size=K)
    # take the log of the cumsum of those
    log_cumsum_t = cp.log(cp.cumsum(t))
    
    # Launch main kernel
    reverse_channel_encode_kernel(
        (grid_size, 1, 1),
        (block_size, 1, 1),
        (mu_q,
        cp.int32(dim),
        cp.uint64(K),
        cp.uint64(shared_seed),
        log_w,
        max_log_w)
    )
    cp.cuda.stream.get_current_stream().synchronize()
    
    s = log_cumsum_t - log_w
    
    winning_seed = cp.argmin(s).item()
    sample = generate_sample(dim, shared_seed, winning_seed)

    return winning_seed, sample.astype(np.float16)

def reverse_channel_decode_cupy(dim, shared_seed, winning_seed):
    sample = generate_sample(dim, shared_seed, winning_seed)
    return sample.astype(np.float16)
