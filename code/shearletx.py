import scipy.io
import numpy as np
import matplotlib.pyplot as plt


import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
import torch
import torchvision
import pyshearlab

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
softmax = torch.nn.Softmax(dim=-1)


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
dtype = torch.cuda.FloatTensor


def torchsheardec2D(torch_X,  torch_shearlets):
    """Shearlet Decomposition function."""
    coeffs = torch.zeros((torch_X.shape[0],torch_X.shape[1],torch_X.shape[2],torch_shearlets.shape[3]))
    Xfreq = torch.fft.ifftshift(torch.fft.fft2(torch.fft.fftshift(torch_X)))
    coeffs = torch.fft.ifftshift(torch.fft.ifft2(torch.fft.fftshift(torch.einsum('bij,bijk->bijk',Xfreq,torch_shearlets)),dim=(-3,2)))
    return torch.real(coeffs)


def torchshearrec2D(torch_coeffs,  torch_shearlets, dualFrameWeights, dtype):
    """Shearlet Reconstruction function."""
    torchdualFrameWeights = torch.Tensor(dualFrameWeights).type(dtype)
    torch_coeffs_freq = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(torch_coeffs)))
    Xfreq = torch.sum(torch_coeffs_freq*torch_shearlets.permute(0,3,1,2),dim=1)
    return torch.real(torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(Xfreq/torchdualFrameWeights))))






class ShearletX:
    def __init__(
            self,
            model, 
            noise_bs,
            optim_steps,
            lr,
            l1lambda,
            mask_init,
            distortion_lambda=1.,
            obfuscation='uniform',
            maximize_label=False,
            l1_reg=.8,
            device=DEVICE, 
            gamma=None):
        """
        args:
           model: nn.Module classifier to be explained
           noise_bs: int number of noise perturbation samples
           optim_steps: int number of optimization steps
           lr: float learning rate for mask
           l1lambda: float l1 shearlet coefficient multiplier
           obfuscation: str "gaussian" or "uniform"
           maximize_label: bool - whether to maximize the label probability
           mask_init: string, int or tenso describing the initial mask on shearlet coefficients
           distortion_lambda: float, multimplier for distortion loss term
           l1_reg: float l1 spatial regularization multiplier
           device: str cpu or gpu
           gamma: gamma for gamma adjustment for better visualization
        """
        self.model = model
        self.noise_bs = noise_bs
        self.optim_steps = optim_steps
        self.lr = lr
        self.l1lambda = l1lambda
        self.mask_init = mask_init
        self.distortion_lambda = distortion_lambda
        self.obfuscation = obfuscation
        self.maximize_label = maximize_label
        self.l1_reg=l1_reg
        self.device=device
        
        self.gamma = gamma
        
        self.get_perturbation = None # this method will be set in method compute_obfuscation strategy

        
    
    def __call__(self, x, target):
        """
        args:
            x: torch.Tensor of shape (bs,c,h,w)
            target: torch.Tensor of shape (bs,)
        """
        assert len(x.shape)==4
        assert x.requires_grad == False
        
        # Initialize shearlet system
        shearletSystem = pyshearlab.SLgetShearletSystem2D(0,x.size(-2), x.size(-1), 4)
        shearlets = shearletSystem['shearlets']
        dualFrameWeights = shearletSystem['dualFrameWeights']
        torch_shearlets =  torch.from_numpy(shearlets[np.newaxis]).type(dtype)

        
        
        # Initialize optimization loss tracking
        l1shearlet_loss = []
        l1spatial_loss = []
        distortion_loss = []
        
        # Get shearlet coefficients (list of shearlet coeffs per color channels)
        self.shearlet_coeffs = []
        for i in range(x.size(1)):
            coeffs = torchsheardec2D(x[:,i,:,:],  torch_shearlets).permute(0,3,1,2)
            self.shearlet_coeffs.append(coeffs)   
        # Get shearlet coefficients of gray scale image
        x_gray = x.sum(dim=1)/3
        self.shearlet_gray = torchsheardec2D(x_gray, torch_shearlets).permute(0,3,1,2)
        assert self.shearlet_gray.size(0)==x.size(0)
        assert len(self.shearlet_gray.shape)==4
        
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
        
        
        for i in range(self.optim_steps):
            print(f'\rIter {i}/{self.optim_steps}', end='')
            
            # Get perturbation on pixel coefficients
            p = self.get_perturbation()
            # Obfuscate shearlet coefficients
            obf_x = []
            for j in range(x.size(1)):
                obf_shearlet_channel_j = (m.unsqueeze(1) * self.shearlet_coeffs[j].unsqueeze(1) + (1 - m.unsqueeze(1)) * p).clamp(0,1).reshape(-1,*self.shearlet_coeffs[j].shape[1:])
                obf_x_channel_j = torchshearrec2D(obf_shearlet_channel_j,  torch_shearlets, dualFrameWeights,dtype)
                assert tuple(obf_x_channel_j.shape) == (self.noise_bs*x.size(0), *x.shape[-2:]), obf_x_channel_j.shape
                obf_x.append(obf_x_channel_j)
            obf_x = torch.stack(obf_x,dim=1)
            assert tuple(obf_x.shape)==(self.noise_bs*x.size(0),*x.shape[1:])
            
            # Get model output for obfuscation
            targets_copied = torch.stack(self.noise_bs*[target]).T.reshape(-1)
            out_obf = self.get_model_output(obf_x, targets_copied).reshape(x.size(0), self.noise_bs)
            # Get shearletx for spatial regularization
            shearletx = torchshearrec2D(m*self.shearlet_gray,  torch_shearlets, dualFrameWeights,dtype).clamp(0,1)
            
            # Compute model output distortion between x and obf_x
            distortion_batch = torch.mean((out_x.unsqueeze(1) - out_obf).pow(2), dim=-1)
            distortion = distortion_batch.sum()
            # Compute l1 norm of pixel mask
            l1shearlet = m.abs().sum() / num_mask
            # Comput l1 spatial loss for shearletX
            l1spatial = (shearletx.abs().reshape(shearletx.size(0), -1).sum(dim=-1) / (np.prod(shearletx.shape[1:]))).sum()
            
            # Log losses
            distortion_loss.append(distortion_batch.detach().clone().cpu().numpy())
            l1shearlet_loss.append(l1shearlet.item())
            l1spatial_loss.append(l1spatial.item())
            
            # Compute optimization loss
            loss = self.distortion_lambda * distortion + self.l1lambda * l1shearlet + self.l1_reg * l1spatial
            
            # Performance optimization step
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            # Project mask into [0,1]
            with torch.no_grad():
                m.clamp_(0,1)
        
        if self.gamma is not None:
            x_gamma_adjust = 255 * x  / x.max() # max value is 255
            torchvision.transforms.functional.adjust_gamma(x_gamma_adjust, self.gamma)
            x_gamma_adjust /= x_gamma_adjust.max()
            shearlet_coeffs_gamma_adjusted = []
            for i in range(x_gamma_adjust.size(1)):
                coeffs = torchsheardec2D(x[:,i,:,:],  torch_shearlets).permute(0,3,1,2)
                shearlet_coeffs_gamma_adjusted.append(coeffs) 
                
            shearletx_per_rgb = [torchshearrec2D(m.detach()*coeffs,  torch_shearlets, dualFrameWeights,dtype).clamp(0,1).unsqueeze(1) for coeffs in shearlet_coeffs_gamma_adjusted]
        else:
            shearletx_per_rgb = [torchshearrec2D(m.detach()*coeffs,  torch_shearlets, dualFrameWeights,dtype).clamp(0,1).unsqueeze(1) for coeffs in self.shearlet_coeffs]
            
        assert tuple(shearletx_per_rgb[0].shape)==(x.size(0),1,x.size(-2),x.size(-1)), tuple(shearletx_per_rgb[0].shape)
        shearletx = torch.cat(shearletx_per_rgb, dim=1).clamp(0,1)
        assert tuple(shearletx.shape)==(x.size(0),3,x.size(-2),x.size(-1)), tuple(shearletx.shape)
        
        """
        #shearletx = torchshearrec2D(m.detach()*self.shearlet_gray,  torch_shearlets, dualFrameWeights,dtype).clamp(0,1).unsqueeze(1)
        shearletx_per_rgb = [torchshearrec2D(m.detach()*coeffs,  torch_shearlets, dualFrameWeights,dtype).clamp(0,1).unsqueeze(1) for coeffs in self.shearlet_coeffs]
        assert tuple(shearletx_per_rgb[0].shape)==(x.size(0),1,x.size(-2),x.size(-1)), tuple(shearletx_per_rgb[0].shape)
        shearletx = torch.cat(shearletx_per_rgb, dim=1).clamp(0,1)
        assert tuple(shearletx.shape)==(x.size(0),3,x.size(-2),x.size(-1)), tuple(shearletx.shape)
        """
        
        assert len(tuple(shearletx.shape)) == 4, shearletx.shape # shape is bs x 1 x w x h
        
        masked_representation_rgb = [m.detach() * coeffs for coeffs in self.shearlet_coeffs]
        assert tuple(masked_representation_rgb[0].shape)==(x.size(0), masked_representation_rgb[0].size(1), x.size(-2), x.size(-1))
        l1_masked_representation = sum([b.abs().sum().item() for b in masked_representation_rgb])
        l1_img_representation = sum([a.abs().sum().item() for a in self.shearlet_coeffs])
        
        # Compute entropy in masked shearlets
        normalization = sum([b.abs().pow(2).sum().item() for b in masked_representation_rgb])
        entropy_masked_representation = -sum([( (b.abs().pow(2)/normalization) * torch.log((b.abs().pow(2)/normalization)+1e-7)).sum().item() for b in masked_representation_rgb])
        
        # Compute entropy in shearlets of original images (not masked)
        normalization = sum([a.abs().pow(2).sum().item() for a in self.shearlet_coeffs])
        entropy_img_representation = -sum([( (a.abs().pow(2)/normalization) * torch.log((a.abs().pow(2)/normalization)+1e-7)).sum().item() for a in self.shearlet_coeffs])
        
        history = {'distortion': distortion_loss,
                   'mask': m.detach(),
                   'l1shearlet': l1shearlet_loss, 
                   'l1spatial': l1spatial_loss,
                   'l1_masked_representation': np.array(l1_masked_representation),
                   'l1_img_representation': np.array(l1_img_representation), 
                   'entropy_masked_representation': np.array(entropy_masked_representation), 
                   'entropy_img_representation': np.array(entropy_img_representation)}
        return shearletx.detach(), history
    
    
    def compute_obfuscation_strategy(self, x):
        # Get std and mean of pixel coefficients
        #std = torch.std(x, dim=[1,2,3]).reshape(x.size(0),1,1,1,1)
        #mean = torch.mean(x, dim=[1,2,3]).reshape(x.size(0),1,1,1,1)
        std = torch.std(self.shearlet_gray, dim=[2,3]).reshape(x.size(0),1,-1,1,1)
        mean = torch.mean(self.shearlet_gray, dim=[2,3]).reshape(x.size(0),1,-1,1,1)
        
        if self.obfuscation == 'gaussian':
            def get_perturbation():
                p = torch.randn((x.size(0), self.noise_bs, *self.shearlet_gray.shape[1:]), 
                                dtype=torch.float32,
                                device=self.device, 
                                requires_grad=False) * std + mean
                return p
            
        elif self.obfuscation == 'uniform':
            def get_perturbation():
                p = torch.rand((x.size(0), self.noise_bs, *self.shearlet_gray.shape[1:]), 
                               dtype=torch.float32, 
                               device=self.device, 
                               requires_grad=False) * 2 * std + mean - std
                return p
        elif self.obfuscation == 'zeros':
            def get_perturbation():
                p = torch.zeros((x.size(0), self.noise_bs, *self.shearlet_gray.shape[1:]), 
                               dtype=torch.float32, 
                               device=self.device, 
                               requires_grad=False) 
                return p
        else:
            raise NotImplementedError('Only uniform, gaussian, and zero perturbations were implemented.')
            
        self.get_perturbation = get_perturbation
        
    def get_init_mask(self, x):
        if self.mask_init == 'ones':
            m = torch.ones((*self.shearlet_gray.shape), 
                              dtype=torch.float32,
                              device=self.device,
                              requires_grad=True)
        elif type(self.mask_init) == float or type(self.mask_init) == int:
            # Get mask for yl coefficients
            m = torch.full((*self.shearlet_gray.shape[2:]), self.mask_init,
                              dtype=torch.float32,
                              device=self.device,
                              requires_grad=True)
        
        elif self.mask_init == 'zeros':
            # Get mask for yl coefficients
            m = torch.zeros((*self.shearlet_gray.shape[2:]), 
                              dtype=torch.float32,
                              device=self.device,
                              requires_grad=True)
            
        elif type(self.mask_init) == torch.Tensor:
            m = self.mask_init
        else:
            raise ValueError('Need to pass string with type of mask or entire initialization mask')
        return m
    
    def get_model_output(self, x, target):
        idx_1 = torch.tensor(np.arange(x.size(0)), dtype=torch.int64)
        idx_2 = target
        out = softmax(self.model(x))[idx_1, idx_2]
        return out
