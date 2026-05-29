import torch
import torch.nn as nn
import numpy as np

from models.googlenet import Inception
import utils.common as utils
from utils.arch_modif import prune_layer,prune_l0_layer
from models.l0_layers import L0Dense,L0Conv2d
def load(model, state_dict, arch, preserved_idx=None):
    #print(arch)
    if 'vgg' in arch:
        load_vgg(model, state_dict)
    elif 'resnet_50' in arch: load_resnet50(model, state_dict)
    elif 'resnet' in arch: load_resnet(model, state_dict, preserved_idx)
    model.load_state_dict(state_dict, strict=False)

# --------------------------------------------------------------------------------

def load_vgg(model:nn.Module, state_dict: dict):
    """
    Method to load vgg.
    Args:
        - model: original model (non pruned)
        - state_dict: loaded state_dict of pruned model
    """
    prev_state_dict = model.state_dict()
    previous_pruned = False
    cnt = 0
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            cout = state_dict[name + '.weight'].size(0)
            prev_cout = prev_state_dict[name + '.weight'].size(0)
            if previous_pruned:
                # prune input channel
                prune_layer(model, name, preserved_indexes, 1)
                previous_pruned = False
            if cout != prev_cout:
                preserved_indexes = list(range(cout))
                prune_layer(model, name, preserved_indexes, 0)
                previous_pruned = True
            cnt+=1
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)) and previous_pruned:
            prune_layer(model, name, preserved_indexes)

# --------------------------------------------------------------------------------

def load_resnet(model:nn.Module, state_dict: dict, preserved_idx):
    """
    Method to load resnet.
    Args:
        - model: original model (non pruned)
        - state_dict: loaded state_dict of pruned model
        - preserved_idx: index of the preserved filters (after pruning)
    """
    prev_state_dict = model.state_dict()
    previous_pruned = False
    cnt = 0
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            cout = state_dict[name + '.weight'].size(0)
            prev_cout = prev_state_dict[name + '.weight'].size(0)
            if previous_pruned:
                if cnt%2 == 0:
                    # prune input channel
                    prune_layer(model, name, preserved_indexes, 1)
                previous_pruned = False
            if cout != prev_cout:
                preserved_indexes = list(range(cout))
                prune_layer(model, name, preserved_indexes, 0)
                previous_pruned = True
                if cnt%2 == 0:
                    m = model
                    #print(name)
                    for m_name in name.split(".")[:-1]:
                        m = m[int(m_name)] if m_name.isdigit() else getattr(m, m_name)
                    setattr(m, 'out_index', torch.tensor(preserved_idx[cnt]))
            cnt+=1
        elif (isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d)) and previous_pruned:
            prune_layer(model, name, preserved_indexes)

# --------------------------------------------------------------------------------

def load_resnet50(model:nn.Module, state_dict: dict):
    """
    Method to load resnet50.
    Args:
        - model: original model (non pruned)
        - state_dict: loaded state_dict of pruned model
    """
    prev_state_dict = model.state_dict()
    previous_pruned_residual = False
    cnt = 0

    for name_block, block in model.named_children():
        if isinstance(block, nn.Conv2d):
            cout = state_dict[name_block + '.weight'].size(0)
            prev_cout = prev_state_dict[name_block + '.weight'].size(0)
            if cout != prev_cout:
                preserved_indexes_residual = list(range(cout))
                prune_layer(model, name_block, preserved_indexes_residual, 0)
                previous_pruned_residual = True
            cnt+=1
        if isinstance(block, nn.BatchNorm2d):
            prune_layer(model, name_block, preserved_indexes_residual)
        # === LAYER ===
        if isinstance(block, nn.ModuleList):
            # ==== BOTTLENECK IN LAYER ===
            for name_bottleneck, bottleneck in block.named_children():
                idx_in_bottleneck = 1
                preserved_indexes_local = preserved_indexes_residual
                previous_pruned_local = previous_pruned_residual
                downsample_input = preserved_indexes_residual if previous_pruned_residual else None
                # ==== MODULES IN BOTTLENECK ===
                for name_module, module in bottleneck.named_modules():
                    name = f"{name_block}.{name_bottleneck}.{name_module}"
                    print(name)
                    if isinstance(module, nn.Conv2d):
                        cout = state_dict[name + '.weight'].size(0)
                        prev_cout = prev_state_dict[name + '.weight'].size(0)
                        
                        if idx_in_bottleneck in [1,2,3]:
                            if previous_pruned_local:
                                prune_layer(model, name, preserved_indexes_local, 1)
                                previous_pruned_local = False
                            preserved_indexes_local = list(range(cout))
                        if idx_in_bottleneck == 3:
                            if cnt in [3,13,26,45]:
                                previous_pruned_residual = True if cout != prev_cout else False
                                preserved_indexes_residual = preserved_indexes_local
                            else:
                                preserved_indexes_local = preserved_indexes_residual
                        if idx_in_bottleneck in [1,2,3]:
                            if cout != prev_cout:
                                prune_layer(model, name, preserved_indexes_local, 0)
                                previous_pruned_local = True
                        if idx_in_bottleneck == 4:
                            if downsample_input is not None:
                                prune_layer(model, name, downsample_input, 1)
                            if previous_pruned_residual:
                                prune_layer(model, name, preserved_indexes_residual, 0)
                        idx_in_bottleneck += 1
                        cnt += 1
                    elif isinstance(module, nn.BatchNorm2d):
                        prune_layer(model, name, preserved_indexes_local)
        if isinstance(block, nn.Linear) and previous_pruned_residual: prune_layer(model, name_block, preserved_indexes_residual, 1)
