"""Top-down geometry and UDP heatmap codec, ported from mmpose 1.x (as used by MedSapiens).

Coordinates are (x, y) in pixels. input_size / heatmap_size are (w, h).
"""
import math

import cv2
import numpy as np


def bbox_xywh_to_center_scale(bbox, padding=1.25):
    x, y, w, h = bbox
    return np.array([x + w * 0.5, y + h * 0.5], np.float32), np.array([w, h], np.float32) * padding


def fix_aspect_ratio(scale, aspect_ratio):
    """Grow the shorter side of `scale` (w, h) so that w / h == aspect_ratio (mmpose TopdownAffine)."""
    w, h = scale
    if w > h * aspect_ratio:
        return np.array([w, w / aspect_ratio], np.float32)
    return np.array([h * aspect_ratio, h], np.float32)


def get_udp_warp_matrix(center, scale, rot, output_size):
    """mmpose.structures.bbox.get_udp_warp_matrix: maps image coords -> output (input_size) coords."""
    input_size = center * 2
    rot_rad = np.deg2rad(rot)
    warp = np.zeros((2, 3), dtype=np.float32)
    scale_x = (output_size[0] - 1) / scale[0]
    scale_y = (output_size[1] - 1) / scale[1]
    warp[0, 0] = math.cos(rot_rad) * scale_x
    warp[0, 1] = -math.sin(rot_rad) * scale_x
    warp[0, 2] = scale_x * (-0.5 * input_size[0] * math.cos(rot_rad) +
                            0.5 * input_size[1] * math.sin(rot_rad) + 0.5 * scale[0])
    warp[1, 0] = math.sin(rot_rad) * scale_y
    warp[1, 1] = math.cos(rot_rad) * scale_y
    warp[1, 2] = scale_y * (-0.5 * input_size[0] * math.sin(rot_rad) -
                            0.5 * input_size[1] * math.cos(rot_rad) + 0.5 * scale[1])
    return warp


def apply_affine(pts, warp):
    pts = np.asarray(pts, np.float32)
    return pts @ warp[:, :2].T + warp[:, 2]


class UDPHeatmap:
    """Gaussian UDP heatmaps (mmpose UDPHeatmap, heatmap_type='gaussian') with DARK-UDP decoding."""

    def __init__(self, input_size, heatmap_size, sigma, blur_kernel_size=11):
        self.input_size = np.array(input_size, np.float32)
        self.heatmap_size = np.array(heatmap_size, np.float32)
        self.sigma = sigma
        self.blur_kernel_size = blur_kernel_size
        self.scale_factor = (self.input_size - 1) / (self.heatmap_size - 1)

    def encode(self, keypoints, visible):
        """keypoints (K, 2) in input-image coords -> heatmaps (K, H, W), weights (K,)."""
        W, H = self.heatmap_size.astype(int)
        K = len(keypoints)
        heatmaps = np.zeros((K, H, W), np.float32)
        weights = visible.astype(np.float32).copy()
        radius = self.sigma * 3
        xs = np.arange(W, dtype=np.float32)
        ys = np.arange(H, dtype=np.float32)[:, None]
        for k in range(K):
            if weights[k] <= 0:
                continue
            mu = keypoints[k] / self.scale_factor
            # mmpose drops keypoints whose 3-sigma box lies fully outside the heatmap
            left, top = mu - radius
            right, bottom = mu + radius + 1
            if left >= W or top >= H or right < 0 or bottom < 0:
                weights[k] = 0
                continue
            heatmaps[k] = np.exp(-((xs - mu[0]) ** 2 + (ys - mu[1]) ** 2) / (2 * self.sigma ** 2))
        return heatmaps, weights

    def decode(self, heatmaps):
        """heatmaps (K, H, W) -> keypoints (K, 2) in input-image coords, scores (K,)."""
        K, H, W = heatmaps.shape
        flat = heatmaps.reshape(K, -1)
        idx = flat.argmax(1)
        scores = flat.max(1)
        kpts = np.stack([idx % W, idx // W], 1).astype(np.float32)
        kpts = self._refine_dark_udp(kpts, heatmaps.copy())
        return kpts * self.scale_factor, scores

    def _refine_dark_udp(self, kpts, heatmaps):
        """mmpose refine_keypoints_dark_udp for a single instance."""
        K, H, W = heatmaps.shape
        ks = self.blur_kernel_size
        for k in range(K):
            origin_max = heatmaps[k].max()
            dr = np.zeros((H + 2, W + 2), np.float32)
            dr[1:-1, 1:-1] = heatmaps[k]
            dr = cv2.GaussianBlur(dr, (ks, ks), 0)
            heatmaps[k] = dr[1:-1, 1:-1]
            heatmaps[k] *= origin_max / max(heatmaps[k].max(), 1e-12)
        np.clip(heatmaps, 1e-3, 50.0, heatmaps)
        np.log(heatmaps, heatmaps)
        hp = np.pad(heatmaps, ((0, 0), (1, 1), (1, 1)), mode='edge').flatten()
        x, y = kpts[:, 0] + 1, kpts[:, 1] + 1
        index = (x + y * (W + 2)).astype(int) + np.arange(K) * (W + 2) * (H + 2)
        i_ = hp[index]
        ix1, iy1 = hp[index + 1], hp[index + W + 2]
        ix1y1 = hp[index + W + 3]
        ix1_y1_ = hp[index - W - 3]
        ix1_, iy1_ = hp[index - 1], hp[index - 2 - W]
        dx = 0.5 * (ix1 - ix1_)
        dy = 0.5 * (iy1 - iy1_)
        derivative = np.stack([dx, dy], 1).reshape(K, 2, 1)
        dxx = ix1 - 2 * i_ + ix1_
        dyy = iy1 - 2 * i_ + iy1_
        dxy = 0.5 * (ix1y1 - ix1 - iy1 + i_ + i_ - ix1_ - iy1_ + ix1_y1_)
        hessian = np.stack([dxx, dxy, dxy, dyy], 1).reshape(K, 2, 2)
        hessian = np.linalg.inv(hessian + np.finfo(np.float32).eps * np.eye(2))
        return kpts - np.einsum('kij,kjl->kil', hessian, derivative).squeeze(-1)
