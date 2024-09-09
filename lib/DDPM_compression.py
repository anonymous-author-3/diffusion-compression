import torch
import numpy as np
import tqdm
from lib.rcc.reverse_channel_coding import chunk_and_encode, decode_from_chunks


def sample_from(mu, sigma):
    return mu + sigma * torch.randn(*mu.shape).to('cuda')

def P(
        model,
        scheduler,
        timestep: int,
        x_t: torch.FloatTensor,
    ):
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] or `tuple`.

        Returns:
            [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] is returned, otherwise a
                tuple is returned where the first element is the sample tensor.

        """
        
        t = timestep
        model_output = model(x_t, t).sample
        sample = x_t

        prev_t = scheduler.previous_timestep(t)

        # 1. compute alphas, betas
        alpha_prod_t = scheduler.alphas_cumprod[t]
        alpha_prod_t_prev = scheduler.alphas_cumprod[prev_t] if prev_t >= 0 else scheduler.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        # 2. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
        if scheduler.config.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        elif scheduler.config.prediction_type == "sample":
            pred_original_sample = model_output
        elif scheduler.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
        else:
            raise ValueError(
                f"prediction_type given as {scheduler.config.prediction_type} must be one of `epsilon`, `sample` or"
                " `v_prediction`  for the DDPMScheduler."
            )

        # 3. Clip or threshold "predicted x_0"
        if scheduler.config.thresholding:
            pred_original_sample = scheduler._threshold_sample(pred_original_sample)
        elif scheduler.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -scheduler.config.clip_sample_range, scheduler.config.clip_sample_range
            )

        # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
        current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t

        # 5. Compute predicted previous sample µ_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        mu = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample

        sigma = scheduler._get_variance(t) ** 0.5
        
        return mu, sigma, pred_original_sample

def Q(scheduler, timestep, x_t, x_0):
    t = timestep
    sample = x_t

    prev_t = scheduler.previous_timestep(t)

    # 1. compute alphas, betas
    alpha_prod_t = scheduler.alphas_cumprod[t]
    alpha_prod_t_prev = scheduler.alphas_cumprod[prev_t] if prev_t >= 0 else scheduler.one
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev
    current_alpha_t = alpha_prod_t / alpha_prod_t_prev
    current_beta_t = 1 - current_alpha_t

    pred_original_sample = x_0
    
    # 3. Clip or threshold "predicted x_0"
    if scheduler.config.thresholding:
        pred_original_sample = scheduler._threshold_sample(pred_original_sample)
    elif scheduler.config.clip_sample:
        pred_original_sample = pred_original_sample.clamp(
            -scheduler.config.clip_sample_range, scheduler.config.clip_sample_range
        )

    # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
    # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
    pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
    current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t

    # 5. Compute predicted previous sample µ_t
    # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
    mu = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample

    sigma = scheduler._get_variance(t) ** 0.5

    return mu, sigma


def distribute_apples(m, n):
    '''
    Given m apples and n buckets, return how many apples to put in each bucket, to distribute as evenly as possible.
    '''
    if n == 0:
        return []
    
    base_apples = m // n
    extra_apples = m % n
    
    distribution = [base_apples] * n
    
    for i in range(extra_apples):
        distribution[i] += 1
    
    return tuple(distribution)

def get_chunk_sizes(Dkl, max_size=8, chunk_padding_bits=2):
    # TODO: Make sure we've added in 2 bits of padding per chunk
    # so split up as if the max size was 2 less
    n = int(np.ceil(Dkl))
    num_chunks = int(np.ceil(n / (max_size - chunk_padding_bits)))
    return distribute_apples(n + chunk_padding_bits * num_chunks, num_chunks)

def zipf_rcc_encode_image(
        image,
        model,
        scheduler,
        max_chunk_size=8,
        chunk_padding_bits=2,
        sample_shape=None,
        D_kl_per_step=None):
    
    computed_chunk_sizes = []
    zipf_s_vals = []
    zipf_n_vals = []
    ground_truth=image
    rng = np.random.default_rng(0)
    if sample_shape is None:
        sample_shape = (1, model.config.in_channels, model.config.sample_size, model.config.sample_size)
    sample = torch.tensor(rng.standard_normal(sample_shape)).to('cuda').to(image.dtype)

    all_seeds = []
    progress = tqdm.tqdm(scheduler.timesteps)
    index=0;
    for t in progress:
        with torch.no_grad():
            mu_p, sigma_p, pred_original_sample = P(model, scheduler, t, sample)
        mu_q, sigma_q = Q(scheduler, t, sample, ground_truth)
        
        Dkl = D_kl_per_step[index]
        index += 1
        
        chunk_sizes = get_chunk_sizes(Dkl, max_chunk_size, chunk_padding_bits)
        computed_chunk_sizes.append(chunk_sizes)
        progress.set_description(f"D_kl: {Dkl:.02f}, chunks: {chunk_sizes}")
        chunk_size_sum = sum(chunk_sizes)
        for chunk_size in chunk_sizes:
            zipf_n_vals.append(2**chunk_size)

            chunk_dkl = Dkl * chunk_size / chunk_size_sum
            s = 1 + 1 / (chunk_dkl + np.exp(-1) * np.log(np.e + 1))
            zipf_s_vals.append(s)
       
        q = (mu_q.flatten().detach().cpu()/sigma_q.item()).numpy()
        p = (mu_p.flatten().detach().cpu()/sigma_p.item()).numpy()
        seeds, next_sample = chunk_and_encode(q, p, chunk_sizes, shared_seed=int(t))
        sample = torch.tensor(next_sample).reshape(mu_q.shape).to(sample.device).to(sample.dtype) * sigma_p
        all_seeds.extend(seeds)

    return all_seeds, zipf_s_vals, zipf_n_vals


def get_zipf_params(D_kl_per_step, max_chunk_size, chunk_padding_bits):
    computed_chunk_sizes = []
    zipf_s_vals = []
    zipf_n_vals = []

    for Dkl in D_kl_per_step:
        chunk_sizes = get_chunk_sizes(Dkl, max_chunk_size, chunk_padding_bits)
        computed_chunk_sizes.append(chunk_sizes)
        chunk_size_sum = sum(chunk_sizes)
        for chunk_size in chunk_sizes:
            zipf_n_vals.append(2**chunk_size)

            chunk_dkl = Dkl * chunk_size / chunk_size_sum
            s = 1 + 1 / (chunk_dkl + np.exp(-1) * np.log(np.e + 1))
            zipf_s_vals.append(s)
    return computed_chunk_sizes, zipf_s_vals, zipf_n_vals

def rcc_decode(model, scheduler, seeds, chunk_sizes_per_step, sample_shape):
    sample_shape = (1, model.config.in_channels, sample_shape[0], sample_shape[1])

    seed_tuples = []
    next_seed_idx = 0
    for chunk_sizes in chunk_sizes_per_step:
        seed_tuple = seeds[next_seed_idx: next_seed_idx + len(chunk_sizes)]
        seed_tuples.append(seed_tuple)
        next_seed_idx += len(chunk_sizes)
    
    rng = np.random.default_rng(0)
    sample = torch.tensor(rng.standard_normal(sample_shape)).to('cuda').to(torch.float16)
    for seed_tuple, chunk_sizes, t in tqdm.tqdm(zip(seed_tuples, chunk_sizes_per_step, scheduler.timesteps[:-1])):
        with torch.no_grad():
            mu_p, sigma_p, pred_original_sample = P(model, scheduler, t, sample)
        
        p = (mu_p / sigma_p).flatten().detach().cpu().numpy()
        sample = torch.tensor(decode_from_chunks(p, seed_tuple, chunk_sizes, int(t))).reshape(mu_p.shape).to(sample.device).to(sample.dtype) * sigma_p
    return sample

def find_first_index(tensor, x):
    indices = (tensor <= x).nonzero()
    return indices[0].item() if indices.nelement() > 0 else -1

def denoise(noisy_sample, noisy_sample_timestep, model, scheduler):
    sample = noisy_sample.to('cuda')
    
    t_idx = find_first_index(scheduler.timesteps, noisy_sample_timestep)
    for t in scheduler.timesteps[t_idx:]:
        with torch.no_grad():
            residual = model(sample, t).sample
        sample = scheduler.step(residual, t, sample).prev_sample

    return sample

