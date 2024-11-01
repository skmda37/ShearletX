import numpy as np
import torch

import numpy as np
import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
softmax = torch.nn.Softmax(dim=-1)


class PixelMask:
    def __init__(
            self,
            model, 
            noise_bs,
            optim_steps,
            lr,
            l1lambda,
            mask_init,
            obfuscation='uniform',
            maximize_label=False,
            tv_reg=.5,
            device=DEVICE):
        """
        args:
           model: nn.Module classifier to be explained
           noise_bs: int number of noise perturbation samples
           optim_steps: int number of optimization steps
           lr: float learning rate for mask
           l1lambda: float l1 pixel coefficient multiplier
           obfuscation: str "gaussian" or "uniform"
           maximize_label: bool - whether to maximize the label probability
           mask_init: tensor; mask on pixel coeffcients
           tv_reg: float tv spatial regularization multiplier
           device: str cpu or gpu
        """
        self.model = model
        self.noise_bs = noise_bs
        self.optim_steps = optim_steps
        self.lr = lr
        self.l1lambda = l1lambda
        self.mask_init = mask_init
        self.obfuscation = obfuscation
        self.maximize_label = maximize_label
        self.tv_reg=tv_reg
        self.device=device
    
        self.get_perturbation = None # this method will be set in method compute_obfuscation strategy
        
    
    def __call__(self, x, target):
        """
        args:
            x: torch.Tensor of shape (bs,c,h,w)
            target: torch.Tensor of shape (bs,)
        """
        assert len(x.shape)==4
        assert x.requires_grad == False
        
        # Initialize optimization loss tracking
        l1pixel_loss = []
        tv_loss = []
        distortion_loss = []
        
        # Compute obfuscation strategy
        self.compute_obfuscation_strategy(x)
        
        # Initialize pixel mask
        m = self.get_init_mask(x)
        
        # Get total number of mask entries
        with torch.no_grad():
            num_mask = m.view(m.size(0),-1).size(-1)
        
        # Initialize optimizer
        opt = torch.optim.Adam([m], lr=self.lr)
        
        # Get reference output for distortion
        if self.maximize_label:
            out_x =  torch.ones((x.size(0),),
                                requires_grad=False,
                                dtype=torch.float32,
                                device=self.device)
        else: 
            out_x = self.get_model_output(x, target)
        
       # Optimize mask 
        for i in range(self.optim_steps):
            print(f'\rIter {i}/{self.optim_steps}', end='')
            
            # Get perturbation on pixel coefficients
            p = self.get_perturbation()
            # Obfuscate pixel coefficients
            obf_x = (m.unsqueeze(1) * x.unsqueeze(1) + (1 - m.unsqueeze(1)) * p).clamp(0,1).reshape(-1, *x.shape[1:])
            # Get model output for obfuscation
            targets_copied = torch.stack(self.noise_bs*[target]).T.reshape(-1)
            out_obf = self.get_model_output(obf_x, targets_copied).reshape(x.size(0), self.noise_bs)
            
            # Compute model output distortion between x and obf_x
            distortion_batch = torch.mean((out_x.unsqueeze(1) - out_obf).pow(2), dim=-1)
            distortion = distortion_batch.sum()
            # Compute l1 norm of pixel mask
            l1pixel = m.abs().sum() / num_mask
            # Compute tv loss of pixel mask
            tv_h = ((m[:,:,1:,:] - m[:,:,:-1,:]).pow(2)).sum()
            tv_w = ((m[:,:,:,1:] - m[:,:,:,:-1]).pow(2)).sum() 
            tv = (tv_h + tv_w) / num_mask
            
            # Log losses
            distortion_loss.append(distortion_batch.detach().clone().cpu().numpy())
            l1pixel_loss.append(l1pixel.item())
            tv_loss.append(tv.item())
            
            # Compute optimization loss
            loss = distortion + self.l1lambda * l1pixel + self.tv_reg * tv
            
            # Performance optimization step
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            # Project mask into [0,1]
            with torch.no_grad():
                m.clamp_(0,1)

        assert len(m.shape)==4, m.shape
        pixelmask = m.detach() * x
        assert pixelmask.shape==x.shape, pixelmask.shape
        history = {
                'distortion': distortion_loss, 'mask': m.detach(),
                'l1pixel': l1pixel_loss, 'tv': tv_loss
        }
        return pixelmask, history
    
    
    def compute_obfuscation_strategy(self, x):
        # Get std and mean of pixel coefficients
        std = torch.std(x, dim=[1,2,3]).reshape(x.size(0),1,1,1,1)
        mean = torch.mean(x, dim=[1,2,3]).reshape(x.size(0),1,1,1,1)
        
        if self.obfuscation == 'gaussian':
            def get_perturbation():
                p = torch.randn((x.size(0), self.noise_bs, *x.shape[1:]), 
                                dtype=torch.float32,
                                device=self.device, 
                                requires_grad=False) * std + mean
                return p
            
        elif self.obfuscation == 'uniform':
            def get_perturbation():
                p = torch.rand((x.size(0), self.noise_bs, *x.shape[1:]), 
                               dtype=torch.float32, 
                               device=self.device, 
                               requires_grad=False) * 2 * std + mean - std
                return p
        elif self.obfuscation == 'zeros':
            def get_perturbation():
                p = torch.zeros((x.size(0), self.noise_bs, *x.shape[1:]), 
                               dtype=torch.float32, 
                               device=self.device, 
                               requires_grad=False) 
                return p
        else:
            raise NotImplementedError('Only uniform, gaussian, and zero perturbations were implemented.')
            
        self.get_perturbation = get_perturbation
        
    def get_init_mask(self, x):
        if self.mask_init == 'ones':
            # Get mask as all ones tensor
            m = torch.ones((x.size(0), 1, *x.shape[2:]), 
                              dtype=torch.float32,
                              device=self.device,
                              requires_grad=True)
        elif type(self.mask_init) == float or type(self.mask_init) == int:
            # Get constant mask for pixel coefficients
            m = torch.full((x.size(0), 1, *x.shape[2:]), self.mask_init,
                              dtype=torch.float32,
                              device=self.device,
                              requires_grad=True)
        
        elif self.mask_init == 'zeros':
            # Get zero mask for pixel coefficients
            m = torch.zeros((x.size(0), 1, *x.shape[2:]), 
                              dtype=torch.float32,
                              device=self.device,
                              requires_grad=True)
            
        elif type(self.mask_init) == torch.Tensor:
            # Mask is predefined as some tensor
            m = self.mask_init
        else:
            raise ValueError('Need to pass string with type of mask or entire initialization mask')
        return m
    
    def get_model_output(self, x, target):
        # Select softmax score for target label that is specified for each batch instance
        batch_idx = torch.tensor(np.arange(x.size(0)), dtype=torch.int64)
        label_idx = target
        out = softmax(self.model(x))[batch_idx, label_idx]
        return out
