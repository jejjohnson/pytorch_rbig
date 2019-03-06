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
