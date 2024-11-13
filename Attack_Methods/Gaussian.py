import numpy as np
import torch
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def gauss_noise(obs, sigma=0.2, val_min=0, val_max=1):
    mu = 0
    feature = obs[0]
    adv_X_in = feature
    noise = np.random.normal(mu, sigma, size=[40, 2]).astype(np.float32)
    spare = np.zeros((40, 6))
    noise_matrix = np.hstack((noise, spare))
    # print(noise_matrix.shape)

    adv_X_in += noise_matrix
    adv_X_in = torch.as_tensor(feature, dtype=torch.float32, device=device)
    adv_X_in[:, 0:2] = torch.clamp(adv_X_in[:, 0:2], val_min, val_max)
    feature[:, 0:2] = adv_X_in[:, 0:2].detach().numpy()

    adv_obs = (feature, obs[1], obs[2])
    return adv_obs





