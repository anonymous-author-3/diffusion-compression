import torch

def denoise(noisy_latent, start_timestep, base, refiner, prompt, guidance_scale, timesteps):
    '''Returns a PIL image.''' # TODO: better docstring

    if refiner is None:
        injected = False
        def latent_injector_callback(pipe, step_index, timestep, callback_kwargs):
            nonlocal injected
            if not injected and timestep <= start_timestep+1:
                injected = True
                return {'latents': noisy_latent}
            return {}

        filtered_timesteps = torch.tensor([start_timestep+1, start_timestep] + [t for t in timesteps if t < start_timestep])

        denoised_img = base(
            prompt=prompt,
            guidance_scale=guidance_scale,
            timesteps=filtered_timesteps,
            callback_on_step_end=latent_injector_callback).images[0]
        return denoised_img

    if start_timestep > 200:
        injected = False
        def latent_injector_callback(pipe, step_index, timestep, callback_kwargs):
            nonlocal injected
            if not injected and timestep <= start_timestep+1:
                injected = True
                return {'latents': noisy_latent}
            return {}
        
        filtered_timesteps = torch.tensor([start_timestep+1, start_timestep] + [t for t in timesteps if t < start_timestep])
        noisy_latent = base(
            prompt=prompt,
            guidance_scale=guidance_scale,
            timesteps=filtered_timesteps,
            denoising_end=0.8,
            callback_on_step_end=latent_injector_callback,
            output_type="latent").images
        start_timestep = 200
    denoising_start = (1000 - start_timestep) / 1000

    return refiner(
        prompt=prompt,
        guidance_scale=guidance_scale,
        timesteps=timesteps,
        denoising_start=denoising_start,
        image=noisy_latent, # TODO: convert to latent output instead of PIL output
    ).images[0]