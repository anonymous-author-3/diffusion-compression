import numpy as np
from lib.rcc.reverse_channel_cupy import reverse_channel_encode_cupy, reverse_channel_decode_cupy

def partition_mu(dim, chunk_sizes, shared_seed=0):
    '''
    return an array of shape (dim,) which determines which chunk each dimension belongs to.
    the values in the array correspond to the indices of the chunk.
    '''
    total_bits = sum(chunk_sizes)
    chunk_ndims = []
    for chunk_size in chunk_sizes[:-1]:
        chunk_ndims.append(int(dim * chunk_size / total_bits))
    chunk_ndims.append(dim - sum(chunk_ndims))

    partition_indices = np.concatenate([np.full(ndims, i) for i, ndims in enumerate(chunk_ndims)])
    rng = np.random.default_rng(shared_seed)
    rng.shuffle(partition_indices)

    return partition_indices

def combine_partitions(partition_indices, partitions):
    combined = np.zeros_like(partition_indices, dtype=partitions[0].dtype)
    for i, partition in enumerate(partitions):
        combined[partition_indices == i] = partition
    return combined

def chunk_and_encode(mu_q, mu_p, chunk_sizes, shared_seed=0):
    partition_indices = partition_mu(len(mu_q), chunk_sizes, shared_seed)

    partitions = []
    seeds = []
    for i, chunk_size in enumerate(chunk_sizes):
        chunk_mask = partition_indices == i
        mu_q_chunk = mu_q[chunk_mask]
        mu_p_chunk = mu_p[chunk_mask]
        chunk_shared_seed = hash((shared_seed, i)) % (2**32)
        seed, partition = reverse_channel_encode(mu_q_chunk, mu_p_chunk, K=int(2**chunk_size), shared_seed=chunk_shared_seed)
        seeds.append(seed)
        partitions.append(partition)
    
    return tuple(seeds), combine_partitions(partition_indices, partitions)

def decode_from_chunks(mu_p, seeds, chunk_sizes, shared_seed=0):
    partition_indices = partition_mu(len(mu_p), chunk_sizes, shared_seed)

    partitions = []
    for i, (seed, chunk_size) in enumerate(zip(seeds, chunk_sizes)):
        mu_p_chunk = mu_p[partition_indices == i]
        chunk_shared_seed = hash((shared_seed, i)) % (2**32)
        partition = reverse_channel_decode(mu_p_chunk, seed, shared_seed=chunk_shared_seed)
        partitions.append(partition)
    return combine_partitions(partition_indices, partitions)

def reverse_channel_encode(mu_q, mu_p, K=None, shared_seed=0):
    diff = (mu_q - mu_p).astype(np.float32)  # Convert to float32
    seed, sample = reverse_channel_encode_cupy(diff, K, shared_seed)
    return seed, (sample + mu_p.astype(np.float16))  # Convert back to float16


def reverse_channel_decode(mu_p, seed, shared_seed=0):
    '''
    Given an isotropic gaussian with unit variance centered at mu_q,
    and a random seed, generate a sample from the distribution q.
    '''
    sample = reverse_channel_decode_cupy(len(mu_p), shared_seed, seed)
    return (sample.astype(np.float16) + mu_p.astype(np.float16)).astype(np.float16)
