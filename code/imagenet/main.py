import os
import shutil
import sys
sys.path.append("..") 
import random
import matplotlib.pyplot as plt

import numpy as np
from torchvision import transforms
import torch
import torchvision.models as models
import torch.nn as nn
import kornia

import wandb

from data import Imagenet 
from imagenet_labels import imagenet_labels_dict
from cartoonx import CartoonX
from pixelrde import PixelRDE


#wandb_dir = '/home/groups/ai/datasets/wandb'
#if os.path.isdir(wandb_dir):
#    shutil.rmtree(wandb_dir)



wandb.init(project="cartoonx++", entity="skolek")


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
    
HPARAMS_CARTOONX = {"l1lambda": 1, "lr": 1e-1, 'obfuscation': 'uniform',
          "optim_steps": 300,  "noise_bs": 16, 'l1_reg': 1., 'mask_init': 'ones'} 
    
HPARAMS_PIXELRDE = {"l1lambda": 1., "lr": 1e-1, 'obfuscation': 'uniform',
          "optim_steps": 300,  "noise_bs": 16, 'tv_reg': 300., 'mask_init': 'ones'} 

def main(samples_per_class=10, num_labels=10, batch_size=10, model_name='mobilenet'):
    # Get model
    model = get_model(model_name)

    # Get dataset
    dataset = Imagenet(samples_per_class=samples_per_class, num_labels=num_labels)
    num_batches = len(dataset) // batch_size
    idx_iter = iter(range(num_batches*batch_size))

    for n in range(num_batches):
        print(f'Processing batch: {n}/{num_batches}\n')
        batch_data = [dataset[next(idx_iter)] for _ in range(batch_size)]
        batch_infos = [b[1] for b in batch_data]
        batch_images = torch.stack([b[0].squeeze(0) for b in batch_data])
        batch_images = batch_images.to(device).requires_grad_(False)
        _, canny_batch = kornia.filters.canny(batch_images)
        
        
        # Get prediction
        preds = nn.Softmax(dim=-1)(model(batch_images)).max(1)[1].detach()

        # Get string label for each prediction in batch
        preds_name = [imagenet_labels_dict[preds[i].item()] for i in range(preds.size(0))]

        # Compute CartoonX with l1-spatial regularization
        print(f'Computing CartoonX...\n')
        cartoonx_method = CartoonX(model=model, device=device, **HPARAMS_CARTOONX)
        cartoonx, history = cartoonx_method(batch_images, preds)
        _, canny_cartoonx = kornia.filters.canny(cartoonx)
        
        # Compute Pixelrde with tv regularization
        print(f'\nComputing Pixel Mask...\n')
        pixelrde_method = PixelRDE(model=model, device=device, **HPARAMS_PIXELRDE)
        pixelrde, history = pixelrde_method(batch_images, preds)
        pixelrde = (pixelrde * batch_images.sum(dim=1, keepdim=True)/3)
        _, canny_pixelrde = kornia.filters.canny(pixelrde)
        
        # Make matplotlib figure for each image in batch
        for i in range(batch_images.size(0)):
            fig, axs = plt.subplots(2,3, figsize=(15,15), dpi=300)
            axs[0,0].imshow(batch_images[i].permute(1,2,0).cpu().numpy())
            axs[0,0].axis('off')
            axs[0,0].set_title(f'{preds_name[i]}')
            axs[0,1].imshow(pixelrde[i].squeeze(0).cpu().numpy(),  vmin=0, vmax=1, cmap="gray")
            axs[0,1].axis('off')
            axs[0,1].set_title('Pixel Mask')
            axs[0,2].imshow(cartoonx[i].squeeze(0).cpu().numpy(), vmin=0, vmax=1, cmap="gray")
            axs[0,2].axis('off')
            axs[0,2].set_title('Cartoonx')
            
            axs[1,0].imshow(canny_batch[i].squeeze(0).cpu().numpy(), cmap='Oranges')
            axs[1,0].axis('off')
            axs[1,1].imshow(canny_pixelrde[i].squeeze(0).cpu().numpy(), cmap='Oranges')
            axs[1,1].axis('off')
            axs[1,2].imshow(canny_cartoonx[i].squeeze(0).cpu().numpy(), cmap='Oranges')
            axs[1,2].axis('off')
            
            fig.suptitle(os.path.join(batch_infos[i]['label_folder_name'], batch_infos[i]['img_file_name']))
            plt.tight_layout()
            plt.show()
            
            # Log figure to wandb
            wandb.log({model_name+'/'+batch_infos[i]['label_name']: plt})
            plt.close()
    wandb.finish()

            
if __name__ == '__main__':
    main(samples_per_class=10, num_labels=1000, batch_size=10, model_name='vgg19')
