from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from vit_pruning.model import load_model, get_optimizer
from vit_pruning.dataset import get_dataloaders
from vit_pruning.prune_utils import Mask, apply_mask

import os

TOTAL_EPOCHS = 200

def val(net, data_loader, device):
    correct_samples = 0
    total_samples = 0
    net.eval()
    with torch.no_grad():
        for (idx, (x, t)) in enumerate(data_loader):
        # for (idx, (x, t)) in tqdm(enumerate(data_loader), desc="Validation"):
            x = net(x.to(device))
            t = t.to(device)

            _, indices = torch.max(x, 1)
            correct_samples += torch.sum(indices == t)
            total_samples += t.shape[0]

    val_acc = float(correct_samples) / total_samples
    return val_acc

def prune_iter(trained_model, layers, strategy, ratio, current_mask=None, **kwargs):
    current_mask = Mask.ones_like(trained_model, layers).numpy() if current_mask is None else current_mask.numpy()

    # Determine the number of weights that need to be pruned.
    number_of_total_weights = np.sum([v.size for v in current_mask.values()])
    current_sparsity = trained_model.sparsity()
    ratio = ratio - current_sparsity
    number_of_weights_to_prune = np.ceil(
        ratio * number_of_total_weights).astype(int)
    # Get the model weights.

    weights = {k: v.clone().cpu().detach().numpy()
                for k, v in trained_model.state_dict().items()
                if k in layers}

    new_mask = strategy(weights, Mask(current_mask), number_of_weights_to_prune, **kwargs)
    

    device = next(trained_model.parameters()).device
    for k in current_mask:
        if k not in new_mask:
            new_mask[k] = current_mask[k]
    for k in new_mask:
        new_mask[k] = new_mask[k].to(device)
    

    return new_mask

def finetune(net, mask, optimizer, scheduler, data_loader, device, epochs=1, verbose=True, logger=None, initial_accuracy=None):
    if not logger: logger = defaultdict(list)
    net.train()
    for _ in range(TOTAL_EPOCHS - epochs):
        scheduler.step()
    with tqdm(total=len(data_loader) * epochs, desc="Finetuning", disable=not verbose) as pbar:
        for e in range(epochs):
            correct_samples = 0
            total_samples = 0
            epoch_loss = []
            for (idx, (x, t)) in enumerate(data_loader):
                
                apply_mask(net, mask)
                x = net(x.to(device))
                t = t.to(device)
                loss = F.cross_entropy(x, t)
                pbar.set_postfix(loss=loss.item())

                optimizer.zero_grad()
                loss.backward()
                epoch_loss.append(loss.item())

                with torch.no_grad():
                    for name, param in net.named_parameters():
                        if name in mask:
                            param.grad *= mask[name]                  

                optimizer.step()
                pbar.update()

                _, indices = torch.max(x, 1)
                correct_samples += torch.sum(indices == t)
                total_samples += t.shape[0]

            scheduler.step()

            total_loss = np.mean(epoch_loss)

            train_acc = float(correct_samples) / total_samples
            if verbose: print(f'Epoch {(e)} training acc: {(train_acc):.5f}')
            logger['loss'].append(total_loss)
            logger['train_acc'].append(train_acc)

            if initial_accuracy is not None and initial_accuracy - train_acc < 0.01:
                break

    return logger

# for iterative, directly input the ratios as [0.25, 0.5, ...., 0.975]
# for oneshot, do a float
def pruning(
    ratios, 
    strategy,  
    num_epochs, 
    device, 
    initial_mask=None,
    train_loader=None, 
    val_loader=None, 
    net=None, 
    layers=None, 
    schedule=None, 
    verbose=True, 
    random_baseline=False,
    finetuning_on=True,
    track_masks=False,
    experiment_name=None,
    **kwargs
):
    if not isinstance(ratios, list): ratios = [ratios]
    if not net: net = load_model()
    if not layers: layers = net.get_prunable_layers()
    else: net.set_prunable_layers(layers)

    default_loaders = get_dataloaders()
    if not train_loader: train_loader = default_loaders[0]
    if not val_loader: val_loader = default_loaders[1]

    prune_mask = initial_mask
    net.to(device)

    init_acc = val(net, val_loader, device)
    print("Initial val_acc: {:.4f}".format(init_acc))
    logger = defaultdict(dict)
    initial_accuracy = None 
    # uncomment to break after finetune train acc is within 1% of the previous accuracy
    # initial_accuracy = val(net, train_loader, device)

    if track_masks:
        mask_tracker = {}
    init_weights = net.state_dict()
    for r in ratios:
        print("PRUNING RATIO", r)
        prune_mask = prune_iter(net, layers, strategy, r, current_mask=prune_mask, **kwargs)

        if r == 0.75:
            for k, v in init_weights.items():
                if (v == net.state_dict()[k]).sum() < 5:
                    print(k, 'doesnt change much, only ', (v - net.state_dict()[k]).norm())

        apply_mask(net, prune_mask)
        
        val_acc = val(net, val_loader, device)
        logger[r]['val_acc/before_finetune'] = val_acc
        if verbose: print('after pruning to {:.4f}: {}'.format(net.sparsity(), val_acc))

        if finetuning_on and val_acc < (init_acc - 0.01):
            optimizer, scheduler = get_optimizer(net)
            log = finetune(net, prune_mask, optimizer, scheduler, train_loader, device, num_epochs, verbose, initial_accuracy)

            val_acc = val(net, val_loader, device)
            if verbose: print('after finetune {:.4f}: {}'.format(net.sparsity(), val_acc))
            logger[r]['val_acc/after_finetune'] = val_acc

            print(logger[r])
            print(log)
            logger[r].update(log)

        if track_masks:
            mask_tracker[r] = prune_mask 

        # model checkpointing
        state_dict = {
            'ratio': r,
            'post_pruning_val_acc':val_acc,
            'model_state_dict': net.state_dict()
        }
        if finetuning_on and val_acc < (init_acc - 0.01):
            state_dict.update({
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()
            })
        state_dict.update({"prune_mask":prune_mask})
        
        ckpt_dir = "./vit_pruning/logs/"+experiment_name
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        path = os.path.join(ckpt_dir, str(r).replace(".", "_")+".pt")
        print('Saved state dict to', path, "\n")
        torch.save(state_dict, path)

        
    if track_masks:
        return logger, mask_tracker
    return logger

