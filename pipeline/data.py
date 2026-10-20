"""COCO keypoint dataset with MedSapiens' top-down train/val pipelines (one instance per image).

Train: RandomFlip(h) -> RandomBBoxTransform -> TopdownAffine(UDP) -> PhotometricDistortion
       -> Blur(p=.1) / MedianBlur(p=.1) / CoarseDropout(1 hole, 20-40%, p=1) -> UDP heatmap target.
Val/test: TopdownAffine(UDP) only.

Unlike MedSapiens (swap='' everywhere), horizontal flip swaps left/right keypoints given by `flip_pairs`.
"""
import json
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .codec import (UDPHeatmap, apply_affine, bbox_xywh_to_center_scale, fix_aspect_ratio,
                    get_udp_warp_matrix)

MEAN = np.array([123.675, 116.28, 103.53], np.float32)
STD = np.array([58.395, 57.12, 57.375], np.float32)


def flip_index(num_keypoints, flip_pairs):
    idx = list(range(num_keypoints))
    for a, b in flip_pairs:
        idx[a], idx[b] = b, a
    return idx


def _truncnorm(rng, size):
    """Standard normal truncated to [-1, 1] (mmpose RandomBBoxTransform._truncnorm)."""
    out = rng.standard_normal(size)
    bad = np.abs(out) > 1
    while bad.any():
        out[bad] = rng.standard_normal(bad.sum())
        bad = np.abs(out) > 1
    return out


class KeypointDataset(Dataset):
    def __init__(self, ann_file, img_dir, codec: UDPHeatmap, train, flip_pairs=()):
        with open(ann_file) as f:
            coco = json.load(f)
        anns = {a['image_id']: a for a in coco['annotations']}
        assert len(anns) == len(coco['images']), 'expected exactly one annotation per image'
        self.samples = [(im, anns[im['id']]) for im in coco['images']]
        self.keypoint_names = coco['categories'][0]['keypoints']
        self.num_keypoints = len(self.keypoint_names)
        self.flip_idx = flip_index(self.num_keypoints, flip_pairs)
        self.img_dir = img_dir
        self.codec = codec
        self.input_size = codec.input_size.astype(int)  # (w, h)
        self.train = train

    def __len__(self):
        return len(self.samples)

    def _rng(self, i):
        # per-worker randomness; torch seeds each worker differently every epoch
        return np.random.default_rng(torch.randint(0, 2 ** 31, (1,)).item() if self.train else i)

    def __getitem__(self, i):
        im_info, ann = self.samples[i]
        img = cv2.imread(os.path.join(self.img_dir, im_info['file_name']), cv2.IMREAD_COLOR)
        assert img is not None, im_info['file_name']
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        kps = np.array(ann['keypoints'], np.float32).reshape(-1, 3)
        xy, vis = kps[:, :2].copy(), kps[:, 2] > 0
        center, scale = bbox_xywh_to_center_scale(ann['bbox'])
        rot = 0.0

        if self.train:
            rng = self._rng(i)
            if rng.random() < 0.5:  # RandomFlip(direction='horizontal')
                w = img.shape[1]
                img = img[:, ::-1].copy()
                xy[:, 0] = w - 1 - xy[:, 0]
                xy, vis = xy[self.flip_idx], vis[self.flip_idx]
                center[0] = w - 1 - center[0]
            # RandomBBoxTransform defaults: shift .16 (p .3), scale [.5, 1.5] (p 1), rotate 80 deg (p .6)
            if rng.random() < 0.3:
                center = center + _truncnorm(rng, 2).astype(np.float32) * 0.16 * scale
            scale = scale * (_truncnorm(rng, 1)[0] * 0.5 + 1.0)
            if rng.random() < 0.6:
                rot = float(_truncnorm(rng, 1)[0] * 80.0)

        scale = fix_aspect_ratio(scale, self.input_size[0] / self.input_size[1])
        warp = get_udp_warp_matrix(center, scale, rot, self.input_size)
        img = cv2.warpAffine(img, warp, tuple(int(v) for v in self.input_size), flags=cv2.INTER_LINEAR)
        xy_in = apply_affine(xy, warp)

        if self.train:
            img = self._photometric(img, rng)
            img = self._blur_and_dropout(img, rng)

        x = torch.from_numpy(((img.astype(np.float32) - MEAN) / STD).transpose(2, 0, 1).copy())
        out = {'img': x, 'index': i, 'warp': warp}
        if self.train:
            heatmaps, weights = self.codec.encode(xy_in, vis)
            out.update(heatmaps=torch.from_numpy(heatmaps), weights=torch.from_numpy(weights))
        else:
            out.update(gt=torch.from_numpy(xy), visible=torch.from_numpy(vis))
        return out

    @staticmethod
    def _photometric(img, rng):
        """mmpose PhotometricDistortion: brightness +-32, contrast .5-1.5, each p=.5.

        Saturation/hue jitter are no-ops on our grayscale radiographs, so they are omitted.
        """
        img = img.astype(np.float32)
        if rng.random() < 0.5:
            img += rng.uniform(-32, 32)
        if rng.random() < 0.5:
            img *= rng.uniform(0.5, 1.5)
        return np.clip(img, 0, 255).astype(np.uint8)

    @staticmethod
    def _blur_and_dropout(img, rng):
        """albumentations Blur(p=.1), MedianBlur(p=.1), CoarseDropout(1 hole, 20-40% side, fill 0, p=1)."""
        if rng.random() < 0.1:
            k = int(rng.choice([3, 5, 7]))
            img = cv2.blur(img, (k, k))
        if rng.random() < 0.1:
            img = cv2.medianBlur(img, int(rng.choice([3, 5, 7])))
        h, w = img.shape[:2]
        hh, ww = int(h * rng.uniform(0.2, 0.4)), int(w * rng.uniform(0.2, 0.4))
        y0, x0 = rng.integers(0, h - hh + 1), rng.integers(0, w - ww + 1)
        img = img.copy()
        img[y0:y0 + hh, x0:x0 + ww] = 0
        return img


def collate(batch):
    out = {}
    for k in batch[0]:
        v = [b[k] for b in batch]
        out[k] = torch.stack(v) if isinstance(v[0], torch.Tensor) else v
    return out
