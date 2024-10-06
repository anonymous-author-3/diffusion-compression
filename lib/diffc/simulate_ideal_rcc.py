import torch
from lib.diffc.utils.dkl import D_kl
from lib.diffc.utils.q import Q

def sample_from(mu, sigma):
    return mu + sigma * torch.randn(*mu.shape).to('cuda')

class EncodingSimulatorCallback():
    def __init__(self, starting_latent, target_latent, recon_steps):
        self.target_latent = target_latent
        self.last_latent = starting_latent
        self.DKL_per_step = []
        self.recon_steps_iter = iter(recon_steps)
        self.next_recon_step = next(self.recon_steps_iter, None)
        self.recon_latents = []
        self.recon_step_indices = []

    def __call__(self, pipe, step_index, timestep, callback_kwargs):
        if timestep == pipe.scheduler.timesteps[-1]:
            return {}
        sigma = (pipe.scheduler._get_variance(timestep) ** 0.5).item()
        sample = self.last_latent
        q_mu, q_sigma = Q(pipe.scheduler, timestep, sample, self.target_latent)

        latents = callback_kwargs['latents']

        p = (latents.flatten().detach().cpu()/sigma).numpy()
        
        q = (q_mu.flatten().detach().cpu()/sigma).numpy()
        Dkl = D_kl(torch.tensor(q).to(torch.float32), torch.tensor(p).to(torch.float32), torch.tensor(1), torch.tensor(1)) # this is in bits, not nats
        self.DKL_per_step.append(Dkl)        

        next_sample = sample_from(q_mu, q_sigma).to(sample.device).to(sample.dtype)
        self.last_latent = next_sample
        
        if self.next_recon_step is not None and int(timestep) <= self.next_recon_step:
            self.recon_latents.append(next_sample.detach())
            self.recon_step_indices.append(step_index)
            while self.next_recon_step is not None and int(timestep) <= self.next_recon_step:
                self.next_recon_step = next(self.recon_steps_iter, None)

        return {'latents': next_sample}


def simulate_ideal_coding(
        pipeline,
        prompt,
        guidance_scale,
        target_latent,
        recon_steps,
        timesteps=None,
        seed=0):
    generator = torch.Generator()
    generator.manual_seed(seed)
    torch.manual_seed(seed)

    starting_latent = torch.randn(target_latent.shape, device=torch.device('cuda'), dtype=torch.float16)
    callback = EncodingSimulatorCallback(starting_latent, target_latent, recon_steps)

    print(f"timesteps: {timesteps}")
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
    
    return callback.DKL_per_step, callback.recon_latents, callback.recon_step_indices
