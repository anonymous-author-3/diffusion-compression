import numpy as np
import torch
from lib.diffc.utils.chunk_coding import get_chunk_sizes, chunk_and_encode
from lib.diffc.utils.dkl import D_kl
from lib.diffc.utils.q import Q

class EncodingCallback():
    def __init__(self, starting_latent, target_latent, max_chunk_size=16, chunk_padding_bits=2, D_kl_per_step=None):
        pass
        self.seed_tuples = []
        self.computed_chunk_sizes = []
        self.last_latent = starting_latent
        self.target_latent = target_latent
        self.max_chunk_size = max_chunk_size
        self.chunk_padding_bits = chunk_padding_bits
        self.D_kl_per_step = D_kl_per_step
        self.Dkls = []
        self.zipf_n_vals = []
        self.zipf_s_vals = []
    
    def __call__(self, pipe, step_index, timestep, callback_kwargs):
        if timestep == pipe.scheduler.timesteps[-1]:
            return {}
        sigma = (pipe.scheduler._get_variance(timestep) ** 0.5).item()
        sample = self.last_latent
        q_mu, q_sigma = Q(pipe.scheduler, timestep, sample, self.target_latent)
        q = (q_mu.flatten().detach().cpu()/sigma).numpy()
        p = (callback_kwargs['latents'].flatten().detach().cpu()/sigma).numpy()
        
        if self.D_kl_per_step == [0]:
            Dkl = D_kl(torch.tensor(q).to(torch.float32), torch.tensor(p).to(torch.float32), torch.tensor(1), torch.tensor(1)) # this is in bits, not nats
            print(f"timestep: {int(timestep)}, Dkl: {Dkl:.01f}")
        else:
            actual_dkl = D_kl(torch.tensor(q).to(torch.float32), torch.tensor(p).to(torch.float32), torch.tensor(1), torch.tensor(1)) # this is in bits, not nats
            Dkl = self.D_kl_per_step[step_index]
            print(f"timestep: {int(timestep)}, nominal Dkl: {Dkl:.01f}, actual: {actual_dkl:.01f}")
        self.Dkls.append(Dkl)
        
        chunk_sizes = get_chunk_sizes(Dkl, self.max_chunk_size, self.chunk_padding_bits)
        chunk_size_sum = sum(chunk_sizes)
        for chunk_size in chunk_sizes:
            self.zipf_n_vals.append(2**chunk_size)

            chunk_dkl = Dkl * chunk_size / chunk_size_sum
            s = 1 + 1 / (chunk_dkl + np.exp(-1) * np.log(np.e + 1))
            self.zipf_s_vals.append(s)

        self.computed_chunk_sizes.append(chunk_sizes)

        seeds, next_sample = chunk_and_encode(q, p, chunk_sizes=chunk_sizes, shared_seed=int(timestep))

        self.seed_tuples.append(seeds)
        next_sample = torch.tensor(next_sample * sigma).reshape(self.last_latent.shape).to(self.last_latent.dtype).to(self.last_latent.device)
        self.last_latent = next_sample
        return {'latents': next_sample}

def SD_encode(
        pipeline,
        prompt,
        guidance_scale,
        target_image,
        timesteps=None,
        max_chunk_size=16,
        chunk_padding_bits=2,
        D_kl_per_step=[0],
        seed=0):
    '''
    Encode an image using Stable Diffusion
    pipeline should be an SD pipeline with the DDPMNoVarianceScheduler
    '''
    
    generator = torch.Generator()
    generator.manual_seed(seed)
    torch.manual_seed(seed)
    starting_latent = torch.randn(target_image.shape, device=target_image.device, dtype=target_image.dtype)    
    callback = EncodingCallback(
        starting_latent,
        target_image,
        max_chunk_size,
        chunk_padding_bits,
        D_kl_per_step)

    pipeline(
        prompt=prompt,
        latents=starting_latent,
        generator=generator,
        callback_on_step_end=callback,
        timesteps=timesteps,
        denoising_end = .999,
        guidance_scale=guidance_scale,
        callback_on_step_end_tensor_inputs=["latents"],
        output_type="latent")
    
    return callback.Dkls, callback.seed_tuples, callback.computed_chunk_sizes, callback.zipf_s_vals, callback.zipf_n_vals
    