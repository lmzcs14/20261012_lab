"""Backbones (Sapiens v1 / Sapiens2 standalone), LoRA and the MedSapiens heatmap head.

LoRA follows mmpretrain LoRAModel as configured by MedSapiens:
  attention q/k/v: rank 4, alpha 4; attention output proj: rank 8, alpha 8; dropout 0.1.
Sapiens v1 has one fused `qkv` Linear; Sapiens2 has separate wq/wk/wv (each gets its own rank-4 adapter).
Everything in the backbone except LoRA is frozen; the head is fully trained.
"""
import functools
import math
import os
import re
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'third_party', 'sapiens2', 'sapiens', 'backbones', 'standalone'))
from sapiens import Sapiens  # noqa: E402
from sapiens2 import Sapiens2  # noqa: E402

BACKBONES = {
    'sapiens1_0.3b': dict(ckpt='checkpoints/pretrained/sapiens_0.3b_epoch_1600_clean.pth',
                          embed_dim=1024, num_layers=24,
                          lora_targets=[(r'layers\.\d+\.attn\.qkv$', 4, 4), (r'layers\.\d+\.attn\.proj$', 8, 8)]),
    'sapiens2_0.4b': dict(ckpt='checkpoints/pretrained/sapiens2_0.4b_pretrain.safetensors',
                          embed_dim=1024, num_layers=24,
                          lora_targets=[(r'blocks\.\d+\.attn\.w[qkv]$', 4, 4), (r'blocks\.\d+\.attn\.proj$', 8, 8)]),
}


class LoRALinear(nn.Module):
    """mmpretrain LoRALinear: y = W x + B(A(dropout(x))) * alpha / rank, B initialised to zero."""

    def __init__(self, base: nn.Linear, rank, alpha, drop_rate=0.1):
        super().__init__()
        self.base = base
        self.lora_down = nn.Linear(base.in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, base.out_features, bias=False)
        self.lora_dropout = nn.Dropout(drop_rate)
        self.scaling = alpha / rank
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x):
        return self.base(x) + self.lora_up(self.lora_down(self.lora_dropout(x))) * self.scaling


def inject_lora(model, targets, drop_rate=0.1):
    hits = 0
    for name, module in list(model.named_modules()):
        for pattern, rank, alpha in targets:
            if isinstance(module, nn.Linear) and re.search(pattern, name):
                parent_name, attr = name.rsplit('.', 1)
                setattr(model.get_submodule(parent_name), attr, LoRALinear(module, rank, alpha, drop_rate))
                hits += 1
                break
    assert hits > 0, 'no LoRA target matched'
    return hits


def build_backbone(name, img_size_hw, grad_checkpoint=True):
    cfg = BACKBONES[name]
    ckpt = os.path.join(ROOT, cfg['ckpt'])
    if name.startswith('sapiens1'):
        # same settings as MedSapiens' mmpretrain VisionTransformer config
        bb = Sapiens(arch='sapiens_0.3b', img_size=img_size_hw, patch_size=16, qkv_bias=True, final_norm=True,
                     out_type='featmap', with_cls_token=False, patch_cfg=dict(padding=2))
        sd = torch.load(ckpt, map_location='cpu', weights_only=False)
        sd = {k: v for k, v in sd.get('state_dict', sd).items() if k != 'cls_token'}
        blocks = bb.layers
    else:
        from safetensors.torch import load_file
        bb = Sapiens2(arch='sapiens2_0.4b', img_size=img_size_hw, patch_size=16, out_type='featmap')
        sd = load_file(ckpt)
        blocks = bb.blocks
    bb.load_state_dict(sd, strict=True)
    for p in bb.parameters():
        p.requires_grad = False
    inject_lora(bb, cfg['lora_targets'])
    if grad_checkpoint:
        for blk in blocks:
            blk.forward = functools.partial(_checkpointed, blk.forward)
    return bb


def _checkpointed(fn, *args, **kwargs):
    if torch.is_grad_enabled():
        return checkpoint(fn, *args, use_reentrant=False, **kwargs)
    return fn(*args, **kwargs)


class HeatmapHead(nn.Module):
    """mmpose HeatmapHead as in MedSapiens: 2x (deconv4x4 s2 768 + BN + ReLU), 2x (conv1x1 768 + BN + ReLU),
    final 1x1 conv -> K heatmaps at 1/4 input resolution."""

    def __init__(self, in_channels, num_keypoints, channels=768):
        super().__init__()
        layers = []
        c = in_channels
        for _ in range(2):
            layers += [nn.ConvTranspose2d(c, channels, 4, stride=2, padding=1, output_padding=0, bias=False),
                       nn.BatchNorm2d(channels), nn.ReLU(inplace=True)]
            c = channels
        for _ in range(2):
            layers += [nn.Conv2d(c, channels, 1), nn.BatchNorm2d(channels), nn.ReLU(inplace=True)]
        self.body = nn.Sequential(*layers)
        self.final_layer = nn.Conv2d(channels, num_keypoints, 1)

    def forward(self, x):
        return self.final_layer(self.body(x))


class PoseModel(nn.Module):
    def __init__(self, backbone_name, num_keypoints, img_size_hw, grad_checkpoint=True):
        super().__init__()
        self.backbone_name = backbone_name
        self.backbone = build_backbone(backbone_name, img_size_hw, grad_checkpoint)
        self.head = HeatmapHead(BACKBONES[backbone_name]['embed_dim'], num_keypoints)

    def forward(self, x):
        return self.head(self.backbone(x)[0])

    def trainable_state_dict(self):
        """Only LoRA + head weights (and BN buffers); the frozen backbone comes from the pretrained file."""
        keep = {n for n, p in self.named_parameters() if p.requires_grad}
        keep |= {n for n, _ in self.head.named_buffers(prefix='head')}
        return {k: v for k, v in self.state_dict().items() if k in keep}

    def load_trainable(self, sd, skip_final_layer=False):
        """Load LoRA + head; with skip_final_layer the K-specific last conv is re-initialised
        (MedSapiens stage 2: shape-mismatched head.final_layer is skipped by mmengine)."""
        if skip_final_layer:
            sd = {k: v for k, v in sd.items() if not k.startswith('head.final_layer.')}
        own = self.state_dict()
        bad = [k for k in sd if k not in own or own[k].shape != sd[k].shape]
        assert not bad, f'unexpected / mismatched keys: {bad[:5]}'
        self.load_state_dict(sd, strict=False)
        return len(sd)


def param_groups(model, lr, weight_decay, layer_decay):
    """mmpose LayerDecayOptimWrapperConstructor: lr * decay^(num_layers + 1 - layer_id).

    layer_id: block i -> i + 1; head -> num_layers + 1. No weight decay for biases and norm parameters.
    """
    num_layers = BACKBONES[model.backbone_name]['num_layers']
    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        m = re.search(r'(?:layers|blocks)\.(\d+)\.', name)
        layer_id = int(m.group(1)) + 1 if m and name.startswith('backbone.') else num_layers + 1
        no_decay = p.ndim == 1 or name.endswith('.bias')
        key = (layer_id, no_decay)
        if key not in groups:
            groups[key] = {'params': [], 'lr': lr * layer_decay ** (num_layers + 1 - layer_id),
                           'weight_decay': 0.0 if no_decay else weight_decay}
        groups[key]['params'].append(p)
    for g in groups.values():
        g['initial_lr'] = g['lr']
    return list(groups.values())


def heatmap_loss(pred, target, weights):
    """mmpose KeypointMSELoss(use_target_weight=True)."""
    w = weights[:, :, None, None]
    return F.mse_loss(pred * w, target * w)
