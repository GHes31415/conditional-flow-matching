import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchdyn
from torchdyn.datasets import generate_moons

# Implement some helper functions


def eight_normal_sample(n, dim, scale=1, var=1):
    m = torch.distributions.multivariate_normal.MultivariateNormal(
        torch.zeros(dim), math.sqrt(var) * torch.eye(dim)
    )
    centers = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
        (1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
        (-1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
        (-1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
    ]
    centers = torch.tensor(centers) * scale
    noise = m.sample((n,))
    multi = torch.multinomial(torch.ones(8), n, replacement=True)
    data = []
    for i in range(n):
        data.append(centers[multi[i]] + noise[i])
    data = torch.stack(data)
    return data


def sample_moons(n):
    x0, _ = generate_moons(n, noise=0.2)
    return x0 * 3 - 1


def sample_8gaussians(n):
    return eight_normal_sample(n, 2, scale=5, var=0.1).float()


class torch_wrapper(torch.nn.Module):
    """Wraps model to torchdyn compatible format."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, t, x, *args, **kwargs):
        return self.model(torch.cat([x, t.repeat(x.shape[0])[:, None]], 1))


def plot_trajectories(traj):
    """Plot trajectories of some selected samples."""
    n = 2000
    plt.figure(figsize=(6, 6))
    plt.scatter(traj[0, :n, 0], traj[0, :n, 1], s=10, alpha=0.8, c="black")
    plt.scatter(traj[:, :n, 0], traj[:, :n, 1], s=0.2, alpha=0.2, c="olive")
    plt.scatter(traj[-1, :n, 0], traj[-1, :n, 1], s=4, alpha=1, c="blue")
    plt.legend(["Prior sample z(S)", "Flow", "z(0)"])
    plt.xticks([])
    plt.yticks([])
    plt.show()

def compare_trajectories(traj1, traj2):
    '''Plot trajectories of some selected samples.'''
    n = 2000
    plt.figure(figsize=(6, 6))
    plt.scatter(traj1[0, :n, 0], traj1[0, :n, 1], s=10, alpha=0.8, c='black')
    plt.scatter(traj1[:, :n, 0], traj1[:, :n, 1], s=0.2, alpha=0.2, c='olive')
    plt.scatter(traj1[-1, :n, 0], traj1[-1, :n, 1], s=4, alpha=1, c='blue')
    # plt.scatter(traj2[0, :n, 0], traj2[0, :n, 1], s=10, alpha=0.8, c='black')
    plt.scatter(traj2[:, :n, 0], traj2[:, :n, 1], s=0.2, alpha=0.2, c='olive')
    plt.scatter(traj2[-1, :n, 0], traj2[-1, :n, 1], s=4, alpha=1, c='red')
    plt.legend(['Prior sample z(S)', 'Flow', 'z(0)'])
    plt.xticks([])
    plt.yticks([])
    plt.show()


class SDE(torch.nn.Module):
    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, ode_drift, score, noise=1.0, reverse=False):
        super().__init__()
        self.drift = ode_drift
        self.score = score
        self.reverse = reverse
        self.noise = noise

    # Drift
    def f(self, t, y):
        if self.reverse:
            t = 1 - t
        if len(t.shape) == len(y.shape):
            x = torch.cat([y, t], 1)
        else:
            x = torch.cat([y, t.repeat(y.shape[0])[:, None]], 1)
        if self.reverse:
            return -self.drift(x) + self.score(x)
        return self.drift(x) + self.score(x)

    # Diffusion
    def g(self, t, y):
        return torch.ones_like(t) * torch.ones_like(y) * self.noise