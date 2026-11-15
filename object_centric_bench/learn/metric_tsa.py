import torch as pt
import torch.nn.functional as ptnf

from .metric import Metric

class GateBimodalLossTSA(Metric):

    def compute(self, input):
        assert input.ndim in [2, 3], f"Expected (b,n) or (b,t,n), got {input.shape}"
        loss = input * (1.0 - input)
        if input.ndim == 3:
            loss = loss.mean(dim=(1, 2))
        else:
            loss = loss.mean(dim=1)
        return loss, None

class GateUsageLossTSA(Metric):

    def compute(self, input):
        assert input.ndim in [2, 3], f"Expected (b,n) or (b,t,n), got {input.shape}"
        if input.ndim == 3:
            loss = input.mean(dim=(1, 2))
        else:
            loss = input.mean(dim=1)
        return loss, None

class DynamicsLoss(Metric):

    def compute(self, input: pt.Tensor, target: pt.Tensor):
        rollout_length = input.shape[1]
        target = target[:, -rollout_length:]

        assert input.shape == target.shape
        loss = ptnf.mse_loss(input, target)[None]

        return loss, None
