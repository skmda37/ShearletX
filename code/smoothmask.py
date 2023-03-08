import numpy as np
import torch

import numpy as np
import torch

from torchray.attribution.extremal_perturbation import extremal_perturbation, contrastive_reward

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
softmax = torch.nn.Softmax(dim=-1)

class SmoothMask:
    def __init__(self, area, model):
        self.area = area 
        self.model = model
    def __call__(self, x, pred):
        mask, _ = extremal_perturbation(
            self.model, x, pred,
            reward_func=contrastive_reward,
            debug=False,
            areas=[self.area]
        )
        smoothmask = mask * x
        return smoothmask

 
