import numpy as np 
import torch 
from vit_pruning.prune_utils import Mask, shuffle_state_dict, unvectorize, shuffle_tensor, vectorize

SEED = 0

# All methods are magnitude-based
# Schedules (one-shot vs. iter) is handled in experiments.py

# METHODS
# Unstructured
# Structured (row-wise)
# Structured (col-wise)
# Structured (block-wise)
# Note: Structured pruning methods expect linear layers (dim 2)

# DISTRIBUTIONS
# Class-blind (remove smallest weights across all modules)
# Class-uniform (remove smaller weights per module)
# Class-distributed (above, depending on module stdv)
# Class-varied (different pruning ratios for different modules)
# Transformer-specific (prune specific layers/attention heads, e.g. just feedforward, just attention matrix, both)

def global_unstructured(weights, current_mask, number_of_weights_to_prune, **kwargs):
    # adapted from https://github.com/facebookresearch/open_lth/blob/main/pruning/sparse_global.py
    # class-blind
    if number_of_weights_to_prune == 0:
        return current_mask

    # Create a vector of all the unpruned weights in the model.    
    current_mask = current_mask.numpy()

    weight_vector = np.concatenate([v[current_mask[k] == 1] for k, v in weights.items()])
    threshold = np.sort(np.abs(weight_vector))[number_of_weights_to_prune - 1]

    new_mask = Mask({k: np.where(np.abs(v) > threshold, current_mask[k], np.zeros_like(v))
                        for k, v in weights.items()})
    return new_mask

def per_layer_unstructured(weights, current_mask, number_of_weights_to_prune, **kwargs):
    # Prune the same fraction of weights per module

    # Create a vector of all the unpruned weights in the model.    
    current_mask = current_mask.numpy()

    # Determine how many weights to prune per layer 
    # total_weights = len(np.concatenate([current_mask[k].size for k, v in weights.items()]))
    total_weights = sum(current_mask[k].size for k in weights.keys())
    prune_frac = number_of_weights_to_prune / total_weights

    new_mask_dict = {}
    for k, v in weights.items():
        # get the module weights 
        weight_vector = v[current_mask[k]==1]
        flat_weight_vector = weight_vector.flatten()
        # determine how many weights to prune from the module (pruning evenly from each layer)
        module_weights_to_prune = int(np.ceil(prune_frac * v.size))

        if module_weights_to_prune == 0:
            new_mask_dict[k] = current_mask[k]
        else:
            threshold = np.sort(np.abs(flat_weight_vector))[module_weights_to_prune - 1]
            new_mask_dict[k] = np.where(np.abs(v) > threshold, current_mask[k], np.zeros_like(v))

    new_mask = Mask(new_mask_dict)
    return new_mask


def shuffle_np(arr, seed=1234, **kwargs):
    gen = np.random.default_rng(seed)
    gen.shuffle(arr, **kwargs)
    return arr

# Adapted random_unstructured_layerwise, random_unstructured_global, and random_unstructured_even
# from https://github.com/facebookresearch/open_lth/blob/main/lottery/branch/randomly_prune.py
def random_unstructured_layerwise(weights, current_mask, number_of_weights_to_prune, **kwargs):
    pass
    if number_of_weights_to_prune == 0:
        return current_mask

    # Create a vector of all the unpruned weights in the model.    
    current_mask = current_mask.numpy()

    total_weights = sum(current_mask[k].size for k in weights.keys())
    prune_frac = number_of_weights_to_prune / total_weights

    new_mask_dict = {}
    for k, v in weights.items():
        # get the module weights 
        weight_vector = v[current_mask[k]==1]
        flat_weight_vector = weight_vector.flatten()
        # determine how many weights to prune from the module (pruning evenly from each layer)
        module_weights_to_prune = int(np.ceil(prune_frac * v.size))

        if module_weights_to_prune == 0:
            new_mask_dict[k] = current_mask[k]
        else:
            threshold = np.sort(np.abs(flat_weight_vector))[module_weights_to_prune - 1]
            new_mask_dict[k] = np.where(np.abs(v) > threshold, current_mask[k], np.zeros_like(v))

    for k, v in current_mask.items():
        new_mask_dict[k][v==1] = shuffle_np(new_mask_dict[k][v==1])
        
    return Mask(new_mask_dict)

def random_unstructured_global(weights, current_mask, number_of_weights_to_prune, **kwargs):
    # class-blind
    if number_of_weights_to_prune == 0:
        return current_mask

    # Create a vector of all the unpruned weights in the model.    
    current_mask = current_mask.numpy()

    weight_vector = np.concatenate([v[current_mask[k] == 1] for k, v in weights.items()])
    threshold = np.sort(np.abs(weight_vector))[number_of_weights_to_prune - 1]

    new_mask = {k: np.where(np.abs(v) > threshold, current_mask[k], np.zeros_like(v))
                        for k, v in weights.items()}

    for k, v in current_mask.items():
        new_mask[k][v==1.] = shuffle_np(new_mask[k][v==1.])
    return Mask(new_mask)

def random_unstructured_even(weights, current_mask, number_of_weights_to_prune, **kwargs):
    # Note: adapted this implementation to run without errors
    # Randomize evenly across all layers.
    if current_mask is None:
        print("WARNING: current_mask is None in random_unstructured_even. Please pass a mask to shuffle.")

    sparsity = current_mask.sparsity
    for i, k in enumerate(sorted(current_mask.keys())):
        flat_mask = current_mask[k].flatten().to(device)
        layer_size = flat_mask.size()[0]
        cond = (torch.arange(layer_size) < torch.ceil(sparsity * layer_size)).to(device)
        layer_mask = torch.where(cond, torch.zeros_like(flat_mask), torch.ones_like(flat_mask))
        current_mask[k] = shuffle_tensor(layer_mask, seed=SEED+i).reshape(current_mask[k].size())
    return current_mask

def global_structured(weights, current_mask, number_of_weights_to_prune, **kwargs):
    # remove `number_of_weights_to_prune` by removing the smallest rows globally
    if number_of_weights_to_prune == 0:
        return current_mask

    n = kwargs.get('n', 1)
    dim = kwargs.get('dim', -1)
    verbose = kwargs.get('verbose', False) 
    layers_to_exclude = kwargs.get('exclude_layers', []) 

    current_mask_numpy = current_mask.numpy()

    # find the threshold for the rows to exclude
    all_norms = np.array([]) 
    for k, v in weights.items():
        if k in layers_to_exclude:
            continue 

        if len(v.shape) != 2: 
            print(f'Warning: invalid layer {k} is not dim 2')
            continue
        # get the module weights 
        weight_vector = v * current_mask_numpy[k]
        # record the norms and the number of weights included in each block 
        raw_norms = np.linalg.norm(weight_vector, ord=n, axis=dim, keepdims=True) 
        norms = raw_norms * np.ones_like(weight_vector)

        if verbose:
            print(f"    Weight shape: {weight_vector.shape}, norms shape: {raw_norms.shape}")
            print(f"    Weight shape: {weight_vector.shape}, norms shape: {raw_norms.shape} -> {norms.shape}")

        all_norms = np.append(all_norms, norms.flatten())
    print(all_norms.shape)
    threshold = np.sort(np.abs(all_norms))[number_of_weights_to_prune - 1]

    new_mask_dict = {}
    for k, v in weights.items():
        if k in layers_to_exclude:
            continue 
        # determine how many weights to prune from the module
        new_mask_dict[k] = np.where(np.linalg.norm(v * current_mask_numpy[k], ord=n, axis=dim, keepdims=True) > threshold, current_mask[k], np.zeros_like(v))
    for k in layers_to_exclude:
        new_mask_dict[k] = current_mask[k]

    new_mask = Mask(new_mask_dict)
    return new_mask

# structured pruning per module - mask out rows of the weight matrix based on magnitude
def per_layer_structured(weights, current_mask, number_of_weights_to_prune, **kwargs):
    n = kwargs.get('n', 1)
    dim = kwargs.get('dim', -1)
    verbose = kwargs.get('verbose', False)

    current_mask = current_mask.numpy()

    total_weights = sum(current_mask[k].size for k in weights.keys())
    prune_frac = number_of_weights_to_prune / total_weights

    new_mask_dict = {}
    for k, v in weights.items():
        if len(v.shape) != 2: 
            print(f'Warning: invalid layer {k} is not dim 2')
            continue
        # get the module weights 
        weight_vector = v * current_mask[k]

        # determine how many weights to prune from the module (pruning evenly from each layer)
        module_weights_to_prune = int(np.ceil(prune_frac * v.size))
        if module_weights_to_prune == 0 or module_weights_to_prune >= np.sum(current_mask[k]):
            new_mask_dict[k] = current_mask[k]
            continue 

        if verbose:
            print(k, v.shape, module_weights_to_prune)

        raw_norms = np.linalg.norm(weight_vector, ord=n, axis=dim, keepdims=True) 
        norms = raw_norms * np.ones_like(weight_vector)
        if verbose:
            print(f"    Weight shape: {weight_vector.shape}, norms shape: {raw_norms.shape} -> {norms.shape}")

        flat_norms = norms[current_mask[k]==1].flatten()
        threshold = np.sort(flat_norms)[module_weights_to_prune - 1] # remove all weights <= this threshold 
       
        new_mask_dict[k] = np.where(norms > threshold, current_mask[k], np.zeros_like(v))
        cur_module_sparsity = 1 - np.sum(new_mask_dict[k].flatten()) / len(new_mask_dict[k].flatten())
        if verbose:
            print("   ", len(flat_norms[flat_norms <= threshold])/len(flat_norms), cur_module_sparsity)

        # TODO: should we be inclusive or exclusive in terms of threshold for per-layer structured?
        # or choose the one that is closer to the desired sparsity? or choose the one that is the first greater than desired sparsity?

    new_mask = Mask(new_mask_dict)
    return new_mask

# structured pruning per module - mask out rows of the weight matrix based on magnitude
# Currently, prune from every module
def per_layer_structured_ratio(weights, current_mask, number_of_weights_to_prune, **kwargs):
    pass
    n = kwargs.get('n', 1)
    dim = kwargs.get('dim', -1)
    verbose = kwargs.get('verbose', False)

    mlp_ratio = kwargs.get('mlp_ratio', 0.5)

    current_mask = current_mask.numpy()

    total_weights = sum(current_mask[k].size for k in weights.keys())
    prune_frac = number_of_weights_to_prune / total_weights

    mlp_frac = mlp_ratio * prune_frac

    new_mask_dict = {}
    for k, v in weights.items():
        if len(v.shape) != 2: 
            print(f'Warning: invalid layer {k} is not dim 2')
            continue
        # get the module weights 
        weight_vector = v * current_mask[k]

        # determine how many weights to prune from the module (pruning evenly from each layer)
        module_weights_to_prune = int(np.ceil(prune_frac * v.size))
        if module_weights_to_prune == 0 or module_weights_to_prune >= np.sum(current_mask[k]):
            new_mask_dict[k] = current_mask[k]
            continue 

        if verbose:
            print(k, v.shape, module_weights_to_prune)

        raw_norms = np.linalg.norm(weight_vector, ord=n, axis=dim, keepdims=True) 
        norms = raw_norms * np.ones_like(weight_vector)
        if verbose:
            print(f"    Weight shape: {weight_vector.shape}, norms shape: {raw_norms.shape} -> {norms.shape}")

        flat_norms = norms[current_mask[k]==1].flatten()
        threshold = np.sort(flat_norms)[module_weights_to_prune - 1] # remove all weights <= this threshold 
       
        new_mask_dict[k] = np.where(norms > threshold, current_mask[k], np.zeros_like(v))
        cur_module_sparsity = 1 - np.sum(new_mask_dict[k].flatten()) / len(new_mask_dict[k].flatten())
        if verbose:
            print("   ", len(flat_norms[flat_norms <= threshold])/len(flat_norms), cur_module_sparsity)

    new_mask = Mask(new_mask_dict)
    return new_mask

def random_structured_layerwise(weights, current_mask, number_of_weights_to_prune, **kwargs):
    new_mask = per_layer_structured(weights, current_mask, number_of_weights_to_prune, **kwargs)
    dim = kwargs.get('dim', -1)
    new_mask = new_mask.numpy()
    current_mask = current_mask.numpy()
    for k, v in current_mask.items():
        if dim == 0: # col pruning
            shape = (v.shape[0], -1)
        elif dim == 1:
            shape = (-1, v.shape[1])
        new_mask[k][v==1] = shuffle_np(new_mask[k][v==1].reshape(shape), axis=1 - dim).reshape(*new_mask[k][v==1].shape)
        
    return Mask(new_mask)