import torch
import torch.nn as nn
import torch.distributions.transforms as transform
from torch.distributions.transforms import Transform 
import torch.nn.functional as F
import torch.distributions as distrib
from torch.distributions import Uniform

class UnivariateNormalize(Transform):
    def __init__(self, histogram):
        super(UnivariateNormalize, self).__init__()

        # Register Parameters
        self.regularize = nn.Parameter(torch.Tensor(1, 1))

        self.build(histogram)
        # # # Initialize Parameters
        # self.init_parameters()

    # def build(self, histogram):
        
    #     pass

    # def _call(self, x):



    #     return x

    # def _fit(self, x):
    #     # Check X

    #     # Check Bounds

    #     # Fit Histogram with Bin Edges


    #     # ========================
    #     # Regularization
    #     # ========================


    #     # Convert RV to rv distribution

    #     return self
