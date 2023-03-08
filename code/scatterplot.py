import os
import shutil
from functools import partial
import random
import matplotlib.pyplot as plt
from matplotlib import colors

import numpy as np
from torchvision import transforms
import torch
import torchvision.models as models
import torch.nn as nn
import kornia

from imagenet.data import Imagenet 
from imagenet.imagenet_labels import imagenet_labels_dict
from shearletx import ShearletX
from waveletx import WaveletX
from pixelmask import PixelMask
from smoothmask import SmoothMask
from shearlet_edge_detector import get_shearlet_based_edges


# Get device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    


# Seed everything for reproducibility
def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
seed_everything(42)

class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.tensor(mean, device=device, requires_grad=False)
        self.std = torch.tensor(std, device=device, requires_grad=False)

    def forward(self, x):
        x = x - self.mean.reshape(self.mean.size(0),1,1)
        x = x / self.std.reshape(self.std.size(0),1,1)
        return x


def get_model(name):
    if name == 'vgg19':
        net = models.vgg19(pretrained=True).eval().to(device)
        model = nn.Sequential(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), net)
    elif name == 'mobilenet':
        net = models.mobilenet_v3_small(pretrained=True).eval().to(device)
        model = nn.Sequential(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), net)
    elif name == 'resnet18':
        net = models.resnet18(pretrained=True).eval().to(device)
        model = nn.Sequential(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), net)
    for param in model.parameters():
        param.requires_grad = False
    return model
    

def main(HPARAMS_WAVELETX_LIST,
         HPARAMS_PIXELMASK_LIST,
         HPARAMS_SHEARLETX_LIST,
         HPARAMS_SMOOTHMASK_LIST,
         samples_per_class=10, 
         num_labels=10,
         model_name='mobilenet', 
         edge_extractor=kornia.filters.sobel):
    
    data_waveletx = [
            {
                'hparams': HPARAMS_WAVELETX,
                'hallu-score': [], 
                'l1_masked_spatial': [], 
                'l1_img_spatial': [], 
                'l1_masked_representation': [], 
                'l1_img_representation': [], 
                'fidelity': [], 
                'prob_img': [], 
                'entropy_masked_representation': [], 
                'entropy_img_representation': []
            } 
            for HPARAMS_WAVELETX in HPARAMS_WAVELETX_LIST
    ]
    data_pixelmask = [
            {
                'hparams': HPARAMS_PIXELMASK, 
                'hallu-score': [], 
                'l1_masked_spatial': [], 
                'l1_img_spatial': [], 
                'l1_masked_representation': [], 
                'l1_img_representation': [], 
                'fidelity': [], 
                'prob_img': [], 
                'entropy_masked_representation': [], 
                'entropy_img_representation': []
            } 
            for HPARAMS_PIXELMASK in HPARAMS_PIXELMASK_LIST
    ]
    data_smoothmask = [
            {
                'hparams': HPARAMS_SMOOTHMASK, 
                'hallu-score': [], 
                'l1_masked_spatial': [], 
                'l1_img_spatial': [], 
                'l1_masked_representation': [], 
                'l1_img_representation': [], 
                'fidelity': [], 
                'prob_img': [], 
                'entropy_masked_representation': [], 
                'entropy_img_representation': []
            } 
            for HPARAMS_SMOOTHMASK in HPARAMS_SMOOTHMASK_LIST
    ]
    data_shearletx = [
            {
                'hparams': HPARAMS_SHEARLETX, 
                'hallu-score': [], 
                'l1_masked_spatial': [], 
                'l1_img_spatial': [], 
                'l1_masked_representation': [], 
                'l1_img_representation': [], 
                'fidelity': [], 
                'prob_img': [], 
                'entropy_masked_representation': [], 
                'entropy_img_representation': []
            } 
            for HPARAMS_SHEARLETX in HPARAMS_SHEARLETX_LIST
    ]
    
    # Get model
    model = get_model(model_name)

    # Get dataset
    dataset = Imagenet(
            samples_per_class=samples_per_class,
            num_labels=num_labels, 
            imagenet_path='/home/groups/ai/datasets/dataset_imagenet_tmp/val'
    )
    N = len(dataset)
    idx_iter = iter(range(N))
    
    # Threshold to binarize edge extractor
    threshold, upper, lower = 0.1, 1, 0

    for n in range(N):
        print(f'Processing batch: {n}/{N}\n')
        batch_data = [dataset[next(idx_iter)]]
        batch_infos = [b[1] for b in batch_data]
        batch_images = torch.stack([b[0].squeeze(0) for b in batch_data])
        batch_images = batch_images.to(device).requires_grad_(False)
        edges_batch = edge_extractor((batch_images.sum(dim=1, keepdim=True)/3).detach().cpu())
        edges_batch = torch.where(edges_batch>threshold, upper, lower)
        dilation, kernel = kornia.morphology.dilation, torch.ones(3,3)
        assert tuple(edges_batch.shape) == (batch_images.size(0), batch_images.size(-2), batch_images.size(-1))
        edges_batch_dilated = dilation(edges_batch.unsqueeze(1), kernel).squeeze(1).detach()
        
        # Get spatial l1 of image
        l1_img_spatial = batch_images.sum(dim=[-1,-2,-3]).cpu().numpy()
        
        # Get prediction
        preds = nn.Softmax(dim=-1)(model(batch_images)).max(1)[1].detach()
        
        # Get probability for image
        prob_img = nn.Softmax(dim=-1)(model(batch_images)).max(1)[0]

        
        for k, HPARAMS_WAVELETX in enumerate(HPARAMS_WAVELETX_LIST):
            # Compute WaveletX
            print(f'Computing WaveletX...\n')
            waveletx_method = WaveletX(model=model, device=device, **HPARAMS_WAVELETX)
            waveletx, history_waveletx = waveletx_method(batch_images, preds)
            assert len(tuple(waveletx.shape))==4, waveletx.shape
            waveletx = waveletx/waveletx.max()
            assert len(tuple(waveletx.shape))==4, waveletx.shape
            edges_waveletx = edge_extractor((waveletx.sum(dim=1,keepdim=True)/3).cpu())
            edges_waveletx = torch.where(edges_waveletx>threshold, upper, lower)
            edge_intersection_waveletx = edges_waveletx - edges_batch_dilated
            assert len(edge_intersection_waveletx.shape)==3, edge_intersection_waveletx.shape
            hallu_score_waveletx = (edge_intersection_waveletx==1).sum(dim=[-1,-2])/(edges_batch==1).sum(dim=[-1,-2])
            assert len(edges_waveletx.shape)==3, edges_waveletx.shape
            
            l1_masked_spatial = (waveletx.detach().abs().sum(dim=[-1,-2,-3])).cpu().numpy()
            fidelity = nn.Softmax(dim=-1)(model(waveletx)).max(1)[0]
            l1_masked_representation = history_waveletx['l1_masked_representation']
            l1_img_representation = history_waveletx['l1_img_representation']
            entropy_masked_representation = history_waveletx['entropy_masked_representation']
            entropy_img_representation = history_waveletx['entropy_img_representation']
            
            data_waveletx[k]['fidelity'].append(fidelity.item())
            data_waveletx[k]['l1_masked_spatial'].append(l1_masked_spatial.item())
            data_waveletx[k]['l1_img_spatial'].append(l1_img_spatial.item())
            data_waveletx[k]['l1_masked_representation'].append(l1_masked_representation.item())
            data_waveletx[k]['l1_img_representation'].append(l1_img_representation.item())
            data_waveletx[k]['hallu-score'].append(hallu_score_waveletx.item())
            data_waveletx[k]['prob_img'].append(prob_img.item())
            data_waveletx[k]['entropy_masked_representation'].append(entropy_masked_representation.item())
            data_waveletx[k]['entropy_img_representation'].append(entropy_img_representation.item())
            
                
        for k, HPARAMS_SHEARLETX in enumerate(HPARAMS_SHEARLETX_LIST):
            # Compute WaveletX with l1-spatial regularization
            print(f'Computing ShearletX...\n')
            shearletx_method = ShearletX(model=model, device=device, **HPARAMS_SHEARLETX)
            shearletx, history_shearletx = shearletx_method(batch_images, preds)
            assert len(tuple(shearletx.shape))==4, shearletx.shape
            shearletx = shearletx/shearletx.max()
            assert len(tuple(shearletx.shape))==4, shearletx.shape
            edges_shearletx = edge_extractor((shearletx.sum(dim=1,keepdim=True)/3).cpu())
            edges_shearletx = torch.where(edges_shearletx>threshold, upper, lower)
            edge_intersection_shearletx = edges_shearletx - edges_batch_dilated
            assert len(edge_intersection_shearletx.shape)==3, edge_intersection_shearletx.shape
            hallu_score_shearletx = (edge_intersection_shearletx==1).sum(dim=[-1,-2])/(edges_batch==1).sum(dim=[-1,-2])
            assert len(edges_shearletx.shape)==3, edges_shearletx.shape
            
            l1_masked_spatial = (shearletx.detach().abs().sum(dim=[-1,-2,-3])).cpu().numpy()
            fidelity = nn.Softmax(dim=-1)(model(shearletx)).max(1)[0]
            l1_masked_representation = history_shearletx['l1_masked_representation']
            l1_img_representation = history_shearletx['l1_img_representation']
            entropy_masked_representation = history_shearletx['entropy_masked_representation']
            entropy_img_representation = history_shearletx['entropy_img_representation']
            
            data_shearletx[k]['fidelity'].append(fidelity.item())
            data_shearletx[k]['l1_masked_spatial'].append(l1_masked_spatial.item())
            data_shearletx[k]['l1_img_spatial'].append(l1_img_spatial.item())
            data_shearletx[k]['l1_masked_representation'].append(l1_masked_representation.item())
            data_shearletx[k]['l1_img_representation'].append(l1_img_representation.item())
            data_shearletx[k]['hallu-score'].append(hallu_score_shearletx.item())
            data_shearletx[k]['prob_img'].append(prob_img.item())
            data_shearletx[k]['entropy_masked_representation'].append(entropy_masked_representation.item())
            data_shearletx[k]['entropy_img_representation'].append(entropy_img_representation.item())
            
                
        for k, HPARAMS_SMOOTHMASK in enumerate(HPARAMS_SMOOTHMASK_LIST):
            # Compute WaveletX with l1-spatial regularization
            print(f'Computing Smooth Mask...\n')
            a = HPARAMS_SMOOTHMASK['a']
            smoothmask, _ = extremal_perturbation(
                model, batch_images, preds[0].item(),
                reward_func=contrastive_reward,
                debug=False,
                areas=[a],
            )
            smoothmasked_gray = (smoothmask*batch_images).sum(dim=1, keepdim=True)/3 # gray scale mask
            assert len(tuple(smoothmask.shape))==4, smoothmask.shape
            edges_smoothmask = edge_extractor(smoothmasked_gray.cpu())
            edges_smoothmask = torch.where(edges_smoothmask>threshold, upper, lower)
            edge_intersection_smoothmask = edges_smoothmask - edges_batch_dilated
            assert len(edge_intersection_smoothmask.shape)==3, edge_intersection_smoothmask.shape
            hallu_score_smoothmask = (edge_intersection_smoothmask==1).sum(dim=[-1,-2])/(edges_batch==1).sum(dim=[-1,-2])
            assert len(edges_smoothmask.shape)==3, edges_ssmoothmask.shape
    
            smoothmask_expl = smoothmask*batch_images
            l1_masked_spatial = (smoothmask_expl.detach().abs().sum(dim=[-1,-2,-3])).cpu().numpy() # l1 spatial is computed on mask applied to gray scale img
            fidelity = nn.Softmax(dim=-1)(model(smoothmask_expl)).max(1)[0]
            
            normalization = smoothmask_expl.detach().abs().pow(2).sum()
            entropy_masked_representation = -( (smoothmask_expl.detach().abs().pow(2)/normalization) * torch.log( (smoothmask_expl.detach().abs().pow(2)/normalization) + 1e-7)).sum().cpu().numpy()
            
            normalization = batch_images.abs().pow(2).sum()
            entropy_img_representation = -( (batch_images.abs().pow(2)/normalization) * torch.log( (batch_images.abs().pow(2)/normalization )+1e-7) ).sum().cpu().numpy()
            
            data_smoothmask[k]['fidelity'].append(fidelity.item())
            data_smoothmask[k]['l1_masked_spatial'].append(l1_masked_spatial.item())
            data_smoothmask[k]['l1_img_spatial'].append(l1_img_spatial.item())
            data_smoothmask[k]['l1_masked_representation'].append(l1_masked_spatial.item())
            data_smoothmask[k]['l1_img_representation'].append(l1_img_spatial.item())
            data_smoothmask[k]['hallu-score'].append(hallu_score_smoothmask.item())
            data_smoothmask[k]['prob_img'].append(prob_img.item())
            data_smoothmask[k]['entropy_masked_representation'].append(entropy_masked_representation.item())
            data_smoothmask[k]['entropy_img_representation'].append(entropy_img_representation.item())

        
        for k, HPARAMS_PIXELMASK in enumerate(HPARAMS_PIXELMASK_LIST):
            # Compute Pixel Mask (no smoothness constraints) 
            print(f'\nComputing Pixel Mask...\n')
            pixelmask_method = PixelMask(model=model, device=device, **HPARAMS_PIXELMASK)
            pixelmask, history_pixelmask = pixelmask_method(batch_images, preds)
            pixelmask = pixelmask / pixelmask.max()
            assert len(tuple(pixelmask.shape))==4, pixelmask.shape
            edges_pixelmask = edge_extractor((pixelmask.sum(dim=1,keepdim=True)/3).cpu())
            edges_pixelmask = torch.where(edges_pixelmask>threshold, upper, lower)
            edge_intersection_pixelmask = edges_pixelmask - edges_batch_dilated
            assert len(edge_intersection_pixelmask.shape)==3, edge_intersection_pixelmask.shape
            hallu_score_pixelmask = (edge_intersection_pixelmask==1).sum(dim=[-1,-2])/(edges_batch==1).sum(dim=[-1,-2])

            l1_masked_spatial = (pixelmask.detach().abs().sum(dim=[-1,-2,-3])).cpu().numpy() # l1 spatial is computed on mask applied to gray scale img
            fidelity = nn.Softmax(dim=-1)(model(pixelmask)).max(1)[0]
            
            normalization = pixelmask.detach().abs().pow(2).sum()
            entropy_masked_representation = -((pixelmask.detach().abs().pow(2)/normalization) * torch.log( (pixelmask.detach().abs().pow(2)/normalization)+1e-7)).sum().cpu().numpy()
            
            normalization = batch_images.detach().abs().pow(2).sum()
            entropy_img_representation = -((batch_images.abs().pow(2)/normalization) * torch.log( (batch_images.abs().pow(2)/normalization)+1e-7)).sum().cpu().numpy()
            
        
            data_pixelmask[k]['fidelity'].append(fidelity.item())
            data_pixelmask[k]['l1_masked_spatial'].append(l1_masked_spatial.item())
            data_pixelmask[k]['l1_img_spatial'].append(l1_img_spatial.item())
            data_pixelmask[k]['l1_masked_representation'].append(l1_masked_spatial.item())
            data_pixelmask[k]['l1_img_representation'].append(l1_img_spatial.item())
            data_pixelmask[k]['hallu-score'].append(hallu_score_pixelmask.item())
            data_pixelmask[k]['prob_img'].append(prob_img.item())
            data_pixelmask[k]['entropy_masked_representation'].append(entropy_masked_representation.item())
            data_pixelmask[k]['entropy_img_representation'].append(entropy_img_representation.item())
            

    np.save(f'scatter_data_{model_name}/data_waveletx.npy',data_waveletx)
    np.save(f'scatter_data_{model_name}/data_pixelmask.npy',data_pixelmask)
    np.save(f'scatter_data_{model_name}/data_shearletx.npy',data_shearletx)
    np.save(f'scatter_data_{model_name}/data_smoothmask.npy',data_smoothmask)


            
if __name__ == '__main__':
    # Model to be explained
    model_name = 'vgg19'#'resnet18'
    print(f'Explaining Model: {model_name}')
    
    
    # Save data for scatter plot
    if os.path.isdir(f'scatter_data_{model_name}'):
            shutil.rmtree(f'scatter_data_{model_name}')

    os.mkdir(f'scatter_data_{model_name}')
    
    # Set params for waveletx
    HPARAMS_WAVELETX = {"l1lambda": 2., "lr": 1e-1, 'obfuscation': 'uniform', 
                    'maximize_label': True, 
                    "optim_steps": 300,  "noise_bs": 16, 
                    'l1_reg': 10., 'mask_init': 'ones'} 
    
    # Hparams ShearletX with spatial reg
    SHEARLETX_HPARAMS = {"l1lambda": 1., "lr": 1e-1, 'obfuscation': 'uniform', 'maximize_label': True,
              "optim_steps": 300,  "noise_bs": 16, 'l1_reg': 2., 'mask_init': 'ones'} 
    
    # Set params for pixelmask
    HPARAMS_PIXELMASK = {"l1lambda": 1., "lr": 1e-1, 'obfuscation': 'uniform',  
                        'maximize_label': True, "optim_steps": 300,  
                        "noise_bs": 16, 'tv_reg': 0., 'mask_init': 'ones'} 
    
    HPARAMS_WAVELETX_LIST = [HPARAMS_WAVELETX]
    HPARAMS_SHEARLETX_LIST = [SHEARLETX_HPARAMS]
    HPARAMS_PIXELMASK_LIST = [HPARAMS_PIXELMASK]
    HPARAMS_SMOOTHMASK_LIST = [{'a': 0.05}, {'a': 0.1}, {'a': 0.2}]
    
    
            
    
    # Set edge extractor
    edge_extractor = partial(get_shearlet_based_edges, min_contrast=3)
    main(samples_per_class=5,
         num_labels=20,
         model_name=model_name, 
         HPARAMS_WAVELETX_LIST=HPARAMS_WAVELETX_LIST,
         HPARAMS_PIXELMASK_LIST=HPARAMS_PIXELMASK_LIST,
         HPARAMS_SHEARLETX_LIST=HPARAMS_SHEARLETX_LIST,
         HPARAMS_SMOOTHMASK_LIST=HPARAMS_SMOOTHMASK_LIST,
         edge_extractor=edge_extractor)

