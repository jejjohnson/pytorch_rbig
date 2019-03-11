"""Contains the models necessary for the nomralized flows.

* Planar Flow
* Radial Flow
* Generalized Divisive Normalization

"""
import torch
import torch.nn as nn
import torch.distributions.transforms as transform
import torch.nn.functional as F
import torch.distributions as distrib


class Flow(transform.Transform, nn.Module):
    def __init__(self):
        transform.Transform.__init__(self)
        nn.Module.__init__(self)

    # Initialize Parameters
    def init_parameters(self):
        for param in self.parameters():
            param.data.uniform_(-0.01, 0.01)

    # Hash Bypass
    def __hash__(self):
        return nn.Module.__hash__(self)


class PlanarFlow(Flow):
    def __init__(self, dim):
        super(PlanarFlow, self).__init__()

        # Register Parameters
        self.jitter = 1e-9
        self.dim = dim
        self.activation = torch.tanh
        self.dactivation = lambda x: 1 - self.activation(x) ** 2

        self.weight = nn.Parameter(torch.Tensor(1, dim))
        self.scale = nn.Parameter(torch.Tensor(1, dim))
        self.bias = nn.Parameter(torch.Tensor(1))

        # Initialize Parameters
        self.init_parameters()

    def _call(self, z):

        # Calculate h(z) = wTz + b
        f_z = F.linear(z, self.weight, self.bias)

        return z + self.scale * self.activation(f_z)

    def log_abs_det_jacobian(self, z):
        f_z = F.linear(z, self.weight, self.bias)

        psi = self.dactivation(f_z) * self.weight

        det_grad = 1 + torch.mm(psi, self.scale.t())

        return torch.log(det_grad.abs() + self.jitter)


class NormalizingFlow(nn.Module):
    def __init__(self, dim, flow_func, flow_length, base_density):
        super().__init__()

        biject = []
        if flow_func == "planar":
            for iflow in range(flow_length):
                biject.append(PlanarFlow(dim))
        else:
            raise ValueError("Unrecognized Flow Function.")

        self.transforms = transform.ComposeTransform(biject)
        self.bijectors = nn.ModuleList(biject)
        self.base_density = base_density
        self.final_density = distrib.TransformedDistribution(
            base_density, self.transforms
        )
        self.log_det = []

    def forward(self, z):
        self.log_det = []

        # Applies series of flows
        for b in range(len(self.bijectors)):
            self.log_det.append(self.bijectors[b].log_abs_det_jacobian(z))
            z = self.bijectors[b](z)

        return z, self.log_det


class FlowSequential(nn.Sequential):
    """ A sequential container for flows.
    In addition to a forward pass it implements a backward pass and
    computes log jacobians.
    """

    def forward(self, inputs, cond_inputs=None, mode="direct", logdets=None):
        """ Performs a forward or backward pass for flow modules.
        Args:
            inputs: a tuple of inputs and logdets
            mode: to run direct computation or inverse
        """
        self.num_inputs = inputs.size(-1)

        if logdets is None:
            logdets = torch.zeros(inputs.size(0), 1, device=inputs.device)

        assert mode in ["direct", "inverse"]
        if mode == "direct":
            for module in self._modules.values():
                inputs, logdet = module(inputs, cond_inputs, mode)
                logdets += logdet
        else:
            for module in reversed(self._modules.values()):
                inputs, logdet = module(inputs, cond_inputs, mode)
                logdets += logdet

        return inputs, logdets

    def log_probs(self, inputs, cond_inputs=None):
        u, log_jacob = self(inputs, cond_inputs)
        log_probs = (-0.5 * u.pow(2) - 0.5 * math.log(2 * math.pi)).sum(
            -1, keepdim=True
        )
        return (log_probs + log_jacob).sum(-1, keepdim=True)

    def sample(self, num_samples=None, noise=None, cond_inputs=None):
        if noise is None:
            noise = torch.Tensor(num_samples, self.num_inputs).normal_()
        device = next(self.parameters()).device
        noise = noise.to(device)
        if cond_inputs is not None:
            cond_inputs = cond_inputs.to(device)
        samples = self.forward(noise, cond_inputs, mode="inverse")[0]
        return samples
