# Code from https://github.com/facebookresearch/open_lth with few or no modifications

import torch
import typing
import numpy as np

# Added function to move mask to cuda
def mask_to_device(mask, device):
    return Mask({k: v.to(device) for k, v in mask.items()})

# Adapted from https://github.com/facebookresearch/open_lth/blob/2ce732fe48abd5a80c10a153c45d397b048e980c/pruning/pruned_model.py
def apply_mask(model, mask, device=None):
    if not device: device = next(model.parameters()).device
    cuda_mask = mask_to_device(mask, device)
    for name, param in model.named_parameters():
        if name in cuda_mask:
            param.data *= cuda_mask[name]

# from https://github.com/facebookresearch/open_lth/blob/main/models/base.py
def get_prunable_layer_names(net):
    """A list of the names of Tensors of this model that are valid for pruning.
    By default, only the weights of convolutional and linear layers are prunable.
    """

    return [name + '.weight' for name, module in net.named_modules() if
            isinstance(module, torch.nn.modules.conv.Conv2d) or
            isinstance(module, torch.nn.modules.linear.Linear)]

# from https://github.com/facebookresearch/open_lth/blob/main/pruning/mask.py
class Mask(dict):
    def __init__(self, other_dict=None):
        super(Mask, self).__init__()
        if other_dict is not None:
            for k, v in other_dict.items(): self[k] = v

    def __setitem__(self, key, value):
        if not isinstance(key, str) or len(key) == 0:
            raise ValueError('Invalid tensor name: {}'.format(key))
        if isinstance(value, np.ndarray):
            value = torch.as_tensor(value)
        if not isinstance(value, torch.Tensor):
            raise ValueError('value for key {} must be torch Tensor or numpy ndarray.'.format(key))
        if ((value != 0) & (value != 1)).any(): raise ValueError('All entries must be 0 or 1.')

        super(Mask, self).__setitem__(key, value)

    @staticmethod
    def ones_like(model, layers_to_prune=None):
        mask = Mask()
        if not layers_to_prune: layers_to_prune = model.get_prunable_layer_names()
        for name in layers_to_prune:
            mask[name] = torch.ones(list(model.state_dict()[name].shape))
        return mask

    def numpy(self):
        return {k: v.cpu().numpy() for k, v in self.items()}
        # return {k: v.to(device).numpy() # v.cpu().numpy()
        #         for k, v in self.items()}

    @property
    def sparsity(self):
        """Return the percent of weights that have been pruned as a decimal."""

        unpruned = torch.sum(torch.tensor([torch.sum(v) for v in self.values()]))
        total = torch.sum(torch.tensor([torch.sum(torch.ones_like(v)) for v in self.values()]))
        return 1 - unpruned.float() / total.float()

    @property
    def density(self):
        return 1 - self.sparsity
