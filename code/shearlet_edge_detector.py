import torch
import numpy as np
import kornia

from coshrem.shearletsystem import EdgeSystem
import coshrem.util
import cmocean

def CoShREM(image_gray, min_contrast):
    #image_gray = image[:,:,0] 
    sys = EdgeSystem(*image_gray.shape)
    edges, orientations_unmasked = sys.detect(image_gray, min_contrast)
    orientations = np.ma.masked_where(orientations_unmasked == -1, orientations_unmasked)
    cmap = cmocean.cm.phase
    cmap.set_bad(color='black')
    return edges,orientations_unmasked, orientations, cmap


def get_shearlet_based_edges(x, min_contrast=10):
    edges_x = []
    print('\nProcessing Edges\n')
    for i in range(x.shape[0]):
        x_i_gray = np.int8(255 * ((x[i].sum(0)/3)))
        edges_coshrem_image, _, _, _ = CoShREM(x_i_gray, min_contrast)
        edges_x.append(edges_coshrem_image)
    edges_x = torch.tensor(np.stack(edges_x), requires_grad=False)
    return edges_x
