import torch
from lib.diffc.utils.chunk_coding import decode_from_chunks

class DecodingCallback():
    def __init__(self, starting_latent, chunk_sizes_per_step, seed_tuples, recon_steps):
        self.starting_latent = starting_latent
        self.chunk_sizes_per_step = chunk_sizes_per_step
        self.seed_tuples = seed_tuples
        self.recon_steps_iter = iter(recon_steps)
        self.next_recon_step = next(self.recon_steps_iter, None)
        self.recon_latents = []
        self.recon_step_indices = []
 
    def __call__(self, pipe, step_index, timestep, callback_kwargs):
        if timestep == pipe.scheduler.timesteps[-1]:
            return {}
        sigma = (pipe.scheduler._get_variance(timestep) ** 0.5).item()
        
        latents = callback_kwargs['latents']

        p = (latents.flatten().detach().cpu()/sigma).numpy()
        
        chunk_sizes = self.chunk_sizes_per_step[step_index]
        seed_tuple = self.seed_tuples[step_index]
        
        # TODO: decode_from_chunks is not consistent with encode, encode takes in sigma, decode assumes unit variance.
        next_sample = torch.tensor(decode_from_chunks(p, seed_tuple, chunk_sizes, int(timestep))).reshape(latents.shape).to(latents.device).to(latents.dtype) * sigma 
        if self.next_recon_step is not None and int(timestep) <= self.next_recon_step:
            self.recon_latents.append(next_sample.detach())
            self.recon_step_indices.append(step_index)
            while self.next_recon_step is not None and int(timestep) <= self.next_recon_step:
                self.next_recon_step = next(self.recon_steps_iter, None)
        return {'latents': next_sample}

def SD_decode(
        pipeline,
        prompt,
        guidance_scale,
        seed_tuples,
        chunk_sizes_per_step,
        recon_steps,
        sample_shape,
        timesteps=None,
        seed=0):
    
    generator = torch.Generator()
    generator.manual_seed(seed)
    torch.manual_seed(seed)
    starting_latent = torch.randn(sample_shape, device=torch.device('cuda'), dtype=torch.float16)

    callback = DecodingCallback(starting_latent, chunk_sizes_per_step, seed_tuples, recon_steps)

    pipeline(
        prompt=prompt,
        latents=starting_latent,
        generator=generator,
        timesteps=timesteps,
        denoising_end = .999,
        callback_on_step_end=callback,
        guidance_scale=guidance_scale,
        callback_on_step_end_tensor_inputs=["latents"],
        output_type="latent")
    return callback.recon_latents, callback.recon_step_indices
