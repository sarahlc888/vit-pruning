# Adapted from https://github.com/omihub777/ViT-CIFAR

import torch
import torch.nn as nn
import torchsummary
from argparse import Namespace
import torch.nn.functional as F

import warmup_scheduler

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, feats:int, head:int=8, dropout:float=0.):
        super(MultiHeadSelfAttention, self).__init__()
        self.head = head
        self.feats = feats
        self.sqrt_d = self.feats**0.5

        self.q = nn.Linear(feats, feats)
        self.k = nn.Linear(feats, feats)
        self.v = nn.Linear(feats, feats)

        self.o = nn.Linear(feats, feats)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, n, f = x.size()
        q = self.q(x).view(b, n, self.head, self.feats//self.head).transpose(1,2)
        k = self.k(x).view(b, n, self.head, self.feats//self.head).transpose(1,2)
        v = self.v(x).view(b, n, self.head, self.feats//self.head).transpose(1,2)

        score = F.softmax(torch.einsum("bhif, bhjf->bhij", q, k)/self.sqrt_d, dim=-1) #(b,h,n,n)
        attn = torch.einsum("bhij, bhjf->bihf", score, v) #(b,n,h,f//h)
        o = self.dropout(self.o(attn.flatten(2)))
        return o

class TransformerEncoder(nn.Module):
    def __init__(self, feats:int, mlp_hidden:int, head:int=8, dropout:float=0.):
        super(TransformerEncoder, self).__init__()
        self.la1 = nn.LayerNorm(feats)
        self.msa = MultiHeadSelfAttention(feats, head=head, dropout=dropout)
        self.la2 = nn.LayerNorm(feats)
        self.mlp = nn.Sequential(
            nn.Linear(feats, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, feats),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        out = self.msa(self.la1(x)) + x
        out = self.mlp(self.la2(out)) + out
        return out

class ViT(nn.Module):
    def __init__(self, in_c:int=3, num_classes:int=10, img_size:int=32,
                 patch:int=8, dropout:float=0.0, num_layers:int=7,
                 hidden:int=384, mlp_hidden:int=384, head:int=12,
                 is_cls_token:bool=True):

        super(ViT, self).__init__()
        # hidden=384

        self.patch = patch # number of patches in one row(or col)
        self.is_cls_token = is_cls_token
        self.patch_size = img_size//self.patch
        f = (img_size//self.patch)**2*3 # 48 # patch vec length
        num_tokens = (self.patch**2)+1 if self.is_cls_token else (self.patch**2)

        self.emb = nn.Linear(f, hidden) # (b, n, f)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden)) if is_cls_token else None
        self.pos_emb = nn.Parameter(torch.randn(1,num_tokens, hidden))
        enc_list = [TransformerEncoder(hidden,mlp_hidden=mlp_hidden, dropout=dropout, head=head) for _ in range(num_layers)]
        self.enc = nn.Sequential(*enc_list)
        self.fc = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, num_classes) # for cls_token
        )

        self.prunable_layers = [
            name + '.weight' for name, module in self.named_modules() if
            isinstance(module, torch.nn.modules.linear.Linear)
            ]

    def forward(self, x):
        out = self._to_words(x)
        out = self.emb(out)
        if self.is_cls_token:
            out = torch.cat([self.cls_token.repeat(out.size(0),1,1), out],dim=1)
        out = out + self.pos_emb
        out = self.enc(out)
        if self.is_cls_token:
            out = out[:,0]
        else:
            out = out.mean(1)
        out = self.fc(out)
        return out

    def _to_words(self, x):
        out = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size).permute(0,2,3,4,5,1)
        out = out.reshape(x.size(0), self.patch**2 ,-1)
        return out
    
    def get_prunable_layers(self):
        return self.prunable_layers

    def get_prunable_layer_names(self):
        return self.prunable_layers

    def set_prunable_layers(self, layers):
        self.prunable_layers = layers


    def sparsity(self):
        pruned_weights = 0
        total_weights = 0
        for name, param in self.named_parameters():
            if name in self.get_prunable_layer_names():
                pruned_weights += torch.sum(param.data == 0)
                total_weights += param.data.nelement()
        sparsity = (float(pruned_weights) / float(total_weights))
        if sparsity < 1e-6:
            sparsity = 0.
        return sparsity

def get_model():
    return ViT()

def get_model_summary(model):
    torchsummary.summary(model, (3, 32, 32))

def load_model():
    path = "vit_pruning/vit.ckpt"
    if torch.cuda.is_available():
        checkpoint = torch.load(path)
    else:
        checkpoint = torch.load(path, map_location='cpu')
    net = ViT(Namespace(**checkpoint['hyper_parameters']))
    if next(iter(checkpoint['state_dict'].keys()))[:6] == 'model.':
        d = {}
        for k, v in checkpoint['state_dict'].items():
            d[k.replace('model.', '')] = v 
        checkpoint['state_dict'] = d
    net.load_state_dict(checkpoint['state_dict'])
    return net

def get_optimizer(model, max_epochs=200):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=5e-5)
    base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-5)
    scheduler = warmup_scheduler.GradualWarmupScheduler(optimizer, multiplier=1.0, total_epoch=5, after_scheduler=base_scheduler)

    return optimizer, scheduler



