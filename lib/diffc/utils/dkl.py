import torch
import numpy as np

def D_kl(mu_1, mu_2, sigma_1, sigma_2):
    """
    Calculate the KL divergence between two isotropic multivariate Gaussian distributions.
    
    :param mu_1: Mean vector of the first Gaussian distribution.
    :param mu_2: Mean vector of the second Gaussian distribution.
    :param sigma_1: Standard deviation of the first Gaussian distribution.
    :param sigma_2: Standard deviation of the second Gaussian distribution.
    
    :return: KL divergence between the two distributions.
    """
    d = len(mu_1.flatten())  # Dimensionality of the Gaussian distributions
    term1 = torch.log(sigma_2**2 / sigma_1**2)
    term2 = -d
    term3 = d * (sigma_1**2 / sigma_2**2)
    term4 = torch.sum((mu_1 - mu_2)**2) / sigma_2**2
    
    kl_div = 0.5 * (term1 + term2 + term3 + term4)
    return kl_div.item() / np.log(2) # convert nats to bits
