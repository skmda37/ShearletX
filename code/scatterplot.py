import os
from functools import partial
import random
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.cm as cm
import matplotlib as mpl
import seaborn as sns
import cmocean
import pandas as pd


import numpy as np
from torchvision import transforms
import torch
import torchvision.models as models
import torch.nn as nn
import kornia

from imagenet_utils.data import Imagenet 
from imagenet_utils.imagenet_labels import imagenet_labels_dict
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

# Set plotting style
plt.style.use('seaborn-white')
tex_fonts = {
"font.family": "serif",
"axes.labelsize": 10,
"font.size": 10,
"legend.fontsize": 8,
"xtick.labelsize": 8,
"ytick.labelsize": 8
}
plt.rcParams.update(tex_fonts)


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
    

def generate_scatter_data(
         HPARAMS_WAVELETX_LIST,
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
            imagenet_path='/home/groups/ai/datasets/imagenet_dataset_tmp/val'
    )
    N = len(dataset)
    idx_iter = iter(range(N))
    
    # Threshold to binarize edge extractor
    threshold, upper, lower = 0.1, 1, 0

    for n in range(N):
        print(f'Processing Image: {n}/{N}\n')
        x_data = [dataset[next(idx_iter)]]
        x_infos = [b[1] for b in x_data]
        x = torch.stack([b[0].squeeze(0) for b in x_data])
        x = x.to(device).requires_grad_(False)
        edges_x = edge_extractor((x.sum(dim=1, keepdim=True)/3).detach().cpu())
        edges_x = torch.where(edges_x>threshold, upper, lower)
        dilation, kernel = kornia.morphology.dilation, torch.ones(3,3)
        edges_x_dilated = dilation(edges_x.unsqueeze(1), kernel).squeeze(1).detach()
        
        # Get spatial l1 of image
        l1_img_spatial = x.sum(dim=[-1,-2,-3]).cpu().numpy()
        
        # Get prediction
        preds = nn.Softmax(dim=-1)(model(x)).max(1)[1].detach()
        
        # Get probability for image
        prob_img = nn.Softmax(dim=-1)(model(x)).max(1)[0]

        
        for k, HPARAMS_WAVELETX in enumerate(HPARAMS_WAVELETX_LIST):
            # Compute WaveletX
            print(f'Computing WaveletX...\n')
            waveletx_method = WaveletX(model=model, device=device, **HPARAMS_WAVELETX)
            waveletx, history_waveletx = waveletx_method(x, preds)
            waveletx = waveletx/waveletx.max()
            edges_waveletx = edge_extractor((waveletx.sum(dim=1,keepdim=True)/3).cpu())
            edges_waveletx = torch.where(edges_waveletx>threshold, upper, lower)
            edge_intersection_waveletx = edges_waveletx - edges_x_dilated
            hallu_score_waveletx = (edge_intersection_waveletx==1).sum(dim=[-1,-2])/(edges_x==1).sum(dim=[-1,-2])
            
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
            shearletx, history_shearletx = shearletx_method(x, preds)
            shearletx = shearletx/shearletx.max()
            edges_shearletx = edge_extractor((shearletx.sum(dim=1,keepdim=True)/3).cpu())
            edges_shearletx = torch.where(edges_shearletx>threshold, upper, lower)
            edge_intersection_shearletx = edges_shearletx - edges_x_dilated
            hallu_score_shearletx = (edge_intersection_shearletx==1).sum(dim=[-1,-2])/(edges_x==1).sum(dim=[-1,-2])
            
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
            print(f'Computing Smooth Mask...\n')
            a = HPARAMS_SMOOTHMASK['a'] # area restriction
            smoothmask_method = SmoothMask(a, model)
            smoothmask = smoothmask_method(x, preds[0].item())
            assert len(smoothmask.shape)==4, f'smoothmask.shape: {smoothmask.shape}'
            smoothmasked_gray = smoothmask.sum(dim=1, keepdim=True)/3 # gray scale mask
            edges_smoothmask = edge_extractor(smoothmasked_gray.cpu())
            edges_smoothmask = torch.where(edges_smoothmask>threshold, upper, lower)
            edge_intersection_smoothmask = edges_smoothmask - edges_x_dilated
            hallu_score_smoothmask = (edge_intersection_smoothmask==1).sum(dim=[-1,-2])/(edges_x==1).sum(dim=[-1,-2])
    
            smoothmask_expl = smoothmask*x
            l1_masked_spatial = (smoothmask_expl.detach().abs().sum(dim=[-1,-2,-3])).cpu().numpy() # l1 spatial is computed on mask applied to gray scale img
            fidelity = nn.Softmax(dim=-1)(model(smoothmask_expl)).max(1)[0]
            
            normalization = smoothmask_expl.detach().abs().pow(2).sum()
            entropy_masked_representation = -( (smoothmask_expl.detach().abs().pow(2)/normalization) * torch.log( (smoothmask_expl.detach().abs().pow(2)/normalization) + 1e-7)).sum().cpu().numpy()
            
            normalization = x.abs().pow(2).sum()
            entropy_img_representation = -( (x.abs().pow(2)/normalization) * torch.log( (x.abs().pow(2)/normalization )+1e-7) ).sum().cpu().numpy()
            
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
            pixelmask, history_pixelmask = pixelmask_method(x, preds)
            pixelmask = pixelmask / pixelmask.max()
            edges_pixelmask = edge_extractor((pixelmask.sum(dim=1,keepdim=True)/3).cpu())
            edges_pixelmask = torch.where(edges_pixelmask>threshold, upper, lower)
            edge_intersection_pixelmask = edges_pixelmask - edges_x_dilated
            hallu_score_pixelmask = (edge_intersection_pixelmask==1).sum(dim=[-1,-2])/(edges_x==1).sum(dim=[-1,-2])

            l1_masked_spatial = (pixelmask.detach().abs().sum(dim=[-1,-2,-3])).cpu().numpy() # l1 spatial is computed on mask applied to gray scale img
            fidelity = nn.Softmax(dim=-1)(model(pixelmask)).max(1)[0]
            
            normalization = pixelmask.detach().abs().pow(2).sum()
            entropy_masked_representation = -((pixelmask.detach().abs().pow(2)/normalization) * torch.log( (pixelmask.detach().abs().pow(2)/normalization)+1e-7)).sum().cpu().numpy()
            
            normalization = x.detach().abs().pow(2).sum()
            entropy_img_representation = -((x.abs().pow(2)/normalization) * torch.log( (x.abs().pow(2)/normalization)+1e-7)).sum().cpu().numpy()
            
        
            data_pixelmask[k]['fidelity'].append(fidelity.item())
            data_pixelmask[k]['l1_masked_spatial'].append(l1_masked_spatial.item())
            data_pixelmask[k]['l1_img_spatial'].append(l1_img_spatial.item())
            data_pixelmask[k]['l1_masked_representation'].append(l1_masked_spatial.item())
            data_pixelmask[k]['l1_img_representation'].append(l1_img_spatial.item())
            data_pixelmask[k]['hallu-score'].append(hallu_score_pixelmask.item())
            data_pixelmask[k]['prob_img'].append(prob_img.item())
            data_pixelmask[k]['entropy_masked_representation'].append(entropy_masked_representation.item())
            data_pixelmask[k]['entropy_img_representation'].append(entropy_img_representation.item())
            

    np.save(f'scatter_data/{model_name}/data_waveletx.npy',data_waveletx)
    np.save(f'scatter_data/{model_name}/data_pixelmask.npy',data_pixelmask)
    np.save(f'scatter_data/{model_name}/data_shearletx.npy',data_shearletx)
    np.save(f'scatter_data/{model_name}/data_smoothmask.npy',data_smoothmask)


            
if __name__ == '__main__':
        
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
    
    # Make a directory to save scatter data
    if not os.path.isdir('scatter_data'):
        os.mkdir('scatter_data')
    for model_name in ['mobilenet', 'resnet18', 'vgg19']:
        # Genereate scatter plot for model_name
        print(f'Explaining Model: {model_name}')
        # Create directory to save scatter plot data for given model_name
        if not os.path.isdir(f'scatter_data/{model_name}'):
            os.mkdir(f'scatter_data/{model_name}')


        # ====================== Generate the scatter data ==================
        generate_scatter_data(
                samples_per_class=1, # 5
                num_labels=5,# 20
                model_name=model_name, 
                HPARAMS_WAVELETX_LIST=HPARAMS_WAVELETX_LIST,
                HPARAMS_PIXELMASK_LIST=HPARAMS_PIXELMASK_LIST,
                HPARAMS_SHEARLETX_LIST=HPARAMS_SHEARLETX_LIST,
                HPARAMS_SMOOTHMASK_LIST=HPARAMS_SMOOTHMASK_LIST,
                edge_extractor=edge_extractor
        )


        # ===================== Make the scatter plots ====================

        # Load the scatter data
        data_waveletx = np.load(f'scatter_data/{model_name}/data_waveletx.npy', allow_pickle=True)
        data_shearletx = np.load(f'scatter_data/{model_name}/data_shearletx.npy', allow_pickle=True)
        data_smoothmask = np.load(f'scatter_data/{model_name}/data_smoothmask.npy', allow_pickle=True)
        data_pixelmask = np.load(f'scatter_data/{model_name}/data_pixelmask.npy', allow_pickle=True)

        # Some plotting parameters
        s = 6 # fontsize 
        fontsize_for_mean = 200
        linewidth = 1
        tol = 1e-10
        alpha = 1.

        for a in [0.05, 0.1, 0.2]: # loop over different area constraints a
            if a == 0.05:
                a_idx = 0
            elif a == 0.1:
                a_idx = 1
            elif a == 0.2:
                a_idx = 2
            for information_measure in ['l1_spatial', 'l1_rep', 'entropy_rep']:
                if information_measure=='l1_spatial':
                    masked_information_measure = 'l1_masked_spatial'
                    img_information_measure = 'l1_img_spatial'
                    xlabel = 'CP-$\ell_1$ Pixel'
                elif information_measure=='l1_rep':
                    masked_information_measure = 'l1_masked_representation'
                    img_information_measure = 'l1_img_representation'
                    xlabel = 'CP-$\ell_1$'
                elif information_measure=='entropy_rep':
                    masked_information_measure = 'entropy_masked_representation'
                    img_information_measure = 'entropy_img_representation'
                    xlabel = 'CP-Entropy'
                g = sns.JointGrid()

                # Plot ShearletX
                data = data_shearletx[0]
                # CP Score  
                x = (np.array(data['fidelity']) / np.array(data['prob_img'])) / (np.array(data[masked_information_measure]) / np.array(data[img_information_measure]))
                sns.scatterplot(
                        x=x, y=tol+np.array(data['hallu-score']),
                        ax=g.ax_joint, color='tab:red',
                        label='ShearletX',
                        s=s, alpha=alpha
                )
                sns.kdeplot(
                        y=tol+np.array(data['hallu-score']),
                        linewidth=linewidth, 
                        ax=g.ax_marg_y, fill=True, 
                        color='tab:red', common_norm=False, 
                        log_scale=True
                )
                sns.kdeplot(
                        x=x, linewidth=linewidth, 
                        ax=g.ax_marg_x, fill=True, 
                        color='tab:red', common_norm=False, 
                        log_scale=True
                )


                # Plot WaveletX
                data = data_waveletx[0]
                # CP Score
                x = (np.array(data['fidelity']) / np.array(data['prob_img'])) / (np.array(data[masked_information_measure]) / np.array(data[img_information_measure]))
                sns.scatterplot(
                        x=x, y=tol+np.array(data['hallu-score']), 
                        label='WaveletX', color='tab:orange',
                        ax=g.ax_joint,  s=s, 
                        alpha=alpha
                )
                sns.kdeplot(
                        y=tol+np.array(data['hallu-score']), 
                        linewidth=linewidth, 
                        ax=g.ax_marg_y, fill=True, 
                        color='tab:orange', 
                        common_norm=False, 
                        log_scale=True
                )
                sns.kdeplot(
                        x=x, linewidth=linewidth, 
                        ax=g.ax_marg_x, fill=True, 
                        color='tab:orange', 
                        common_norm=False, 
                        log_scale=True
                )


                # Plot pixelmask
                data = data_pixelmask[0]
                # CP Score
                x = (np.array(data['fidelity']) / np.array(data['prob_img'])) / (np.array(data[masked_information_measure]) / np.array(data[img_information_measure]))
                sns.scatterplot(
                        x=x, y=tol+np.array(data['hallu-score']),
                        label='Pixel Mask (not regularized)', 
                        color='tab:green', 
                        ax=g.ax_joint, s=s, 
                        alpha=alpha
                )
                sns.kdeplot(
                        y=tol+np.array(data['hallu-score']), 
                        linewidth=linewidth, 
                        ax=g.ax_marg_y, fill=True, 
                        color='tab:green', 
                        common_norm=False, 
                        log_scale=True
                )
                sns.kdeplot(
                        x=x, linewidth=linewidth, 
                        ax=g.ax_marg_x, fill=True,
                        color='tab:green', 
                        common_norm=False, 
                        log_scale=True
                )


                # Plot Smooth mask
                data = data_smoothmask[a_idx]
                a = data['hparams']['a']
                # CP Score
                x = (np.array(data['fidelity']) / np.array(data['prob_img'])) / (np.array(data[masked_information_measure]) / np.array(data['l1_img_spatial']))
                sns.scatterplot(
                        x=x, y=np.array(data['hallu-score']), 
                        label=f'Smooth Pixel Mask', 
                        color='tab:blue', 
                        ax=g.ax_joint, 
                        s=s, alpha=alpha
                )
                sns.kdeplot(
                        x=x, linewidth=linewidth, 
                        ax=g.ax_marg_x, fill=True,
                        color='tab:blue',
                        common_norm=False,
                        log_scale=True
                )
                sns.kdeplot(
                        y=tol+np.array(data['hallu-score']),
                        linewidth=linewidth, ax=g.ax_marg_y, 
                        fill=True, color='tab:blue', 
                        common_norm=False, log_scale=True
                )


                # Plot means 
                for data, color in zip(
                        [data_smoothmask[a_idx], data_pixelmask[0], data_waveletx[0], data_shearletx[0]],
                        ['tab:blue', 'tab:green', 'tab:orange','tab:red']):
                    # CP Score
                    x = (np.array(data['fidelity']) / np.array(data['prob_img'])) / (np.array(data['l1_masked_spatial']) / np.array(data['l1_img_spatial']))
                    x_mean = x.mean()
                    y_mean = np.array(data['hallu-score']).mean()
                    sns.scatterplot(
                            x=[x_mean], y=[y_mean],
                            color=color, ax=g.ax_joint,
                            s=fontsize_for_mean, alpha=1
                    )


                ax = g.ax_joint
                ax.set_xscale('log')
                ax.set_yscale('log')
                g.ax_marg_x.set_xscale('log')
                g.ax_marg_y.set_yscale('log')

                ax.set_xlabel(xlabel)
                ax.set_ylabel('Hallucination Score')

                ax.set_xlim(0.1,2000)
                ax.plot([0.2, 1500], [0.2, 0.2], '--',color='black')

                ax.legend(markerscale=4)

                plt.legend(loc='upper right')
                plt.savefig(f'scatterplot_figures/scatterplot_{model_name}_info_as_{img_information_measure}_area_{a}.pdf',bbox_inches='tight',pad_inches = 0)
                plt.show()


