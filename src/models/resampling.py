"""Module `models/resampling.py`."""
from abc import ABC, abstractmethod

import numpy as np
import torch as th



class ScheduleSampler(ABC):
    """
    A distribution over timesteps in the diffusion process, intended to reduce
    variance of the objective.

    By default, samplers perform unbiased importance sampling, in which the
    objective's mean is unchanged.
    However, subclasses may override sample() to change how the resampled
    terms are reweighted, allowing for actual changes in the objective.
    """

    @abstractmethod
    def weights(self):
        """
        Get a numpy array of weights, one per diffusion step.

        The weights needn't be normalized, but must be positive.
        """

    def sample(self, batch_size, device, start_guide_time=1000):
        """
        Importance-sample timesteps for a batch.

        :param batch_size: the number of timesteps.
        :param device: the torch device to save to.
        :return: a tuple (timesteps, weights):
                 - timesteps: a tensor of timestep indices.
                 - weights: a tensor of weights to scale the resulting losses.
        """
        w = self.weights()[:start_guide_time]
        p = w / np.sum(w)
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
        indices = th.from_numpy(indices_np).long().to(device)
        weights_np = 1 / (len(p) * p[indices_np])
        weights = th.from_numpy(weights_np).float().to(device)
        return indices, weights


class UniformSampler(ScheduleSampler):
    """Uniformsampler implementation used by the PerturbDiff pipeline."""
    def __init__(self, diffusion):
        """Special method `__init__`."""
        self.diffusion = diffusion
        self._weights = np.ones([diffusion.num_timesteps])

    def weights(self):
        """Execute `weights` and return values used by downstream logic."""
        return self._weights



class LogitNormalSampler:
    """Sample continuous flow times from a logit-normal distribution."""

    def __init__(self, flow):
        self.flow = flow

    def sample(self, batch_size, device, start_guide_time=1000):
        del start_guide_time
        logits = th.randn(int(batch_size), device=device)
        logits = logits * self.flow.time_std + self.flow.time_mean
        time = th.sigmoid(logits)
        return time, th.ones_like(time)


__all__ = ["LogitNormalSampler", "ScheduleSampler", "UniformSampler"]
