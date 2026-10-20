"""Prediction (with heatmap flip test, as MedSapiens) and dental metrics in original-resolution pixels."""
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import collate, flip_index

SDR_THRESHOLDS_PX = (10, 20, 30, 40)  # original-resolution pixels; no mm spacing available


@torch.no_grad()
def predict(model, ds, dev, flip_pairs, batch_size=8):
    """Returns predicted keypoints (N, K, 2) in each sample's image coords and mean EPE over visible kps."""
    model.eval()
    fidx = flip_index(ds.num_keypoints, flip_pairs)
    preds, errs = [], []
    for batch in DataLoader(ds, batch_size, shuffle=False, num_workers=4, collate_fn=collate):
        x = batch['img'].to(dev)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            hm = model(x).float()
            hm_flip = model(x.flip(-1)).float().flip(-1)[:, fidx]
        hm = ((hm + hm_flip) * 0.5).cpu().numpy()
        for j in range(len(hm)):
            kp_in, _ = ds.codec.decode(hm[j])
            inv = cv2.invertAffineTransform(batch['warp'][j])
            kp = kp_in @ inv[:, :2].T + inv[:, 2]
            preds.append(kp)
            vis = batch['visible'][j].numpy()
            errs.append(np.linalg.norm(kp - batch['gt'][j].numpy(), axis=-1)[vis])
    return {'pred': np.stack(preds), 'epe': float(np.concatenate(errs).mean())}


def dental_metrics(ds, pred):
    """Errors in original-radiograph pixels.

    crop coords -> stored-image coords (+ crop offset) -> original resolution (x sz_origin / stored size).
    Also reports the inter-annotator reference: each annotator vs the mean of the other two, and the model
    vs the same leave-one-out means, so both are compared against an equally noisy two-reader reference.
    """
    names = ds.keypoint_names
    err, err_loo_model, err_loo_human, rows = [], [], [], []
    for (im, ann), p in zip(ds.samples, pred):
        s = np.array(im['sz_origin_wh'], float) / np.array(im['full_size_wh'], float)
        off = np.array(im['crop_box_xyxy'][:2], float)
        g = np.array(ann['keypoints'], float).reshape(-1, 3)[:, :2]
        A = np.array(ann['keypoints_annotators'], float)  # (3, K, 2)
        e = np.linalg.norm((p - g) * s, axis=-1)
        err.append(e)
        for a in range(len(A)):
            ref = np.delete(A, a, 0).mean(0)
            err_loo_human.append(np.linalg.norm((A[a] - ref) * s, axis=-1))
            err_loo_model.append(np.linalg.norm((p - ref) * s, axis=-1))
        rows.append({'stem': im['stem'], 'tooth_index': im['tooth_index'],
                     'pred_orig_px': ((p + off) * s).tolist(), 'gt_orig_px': ((g + off) * s).tolist(),
                     'err_orig_px': e.tolist()})
    err = np.stack(err)  # (N, K)
    m = {'num_teeth': len(err), 'mre_px': float(err.mean()), 'mre_std_px': float(err.mean(1).std()),
         'median_px': float(np.median(err)),
         'per_keypoint_mre_px': {n: float(err[:, k].mean()) for k, n in enumerate(names)},
         'sdr': {f'{t}px': float((err <= t).mean()) for t in SDR_THRESHOLDS_PX},
         'reference_leave_one_annotator_out': {
             'model_mre_px': float(np.mean(err_loo_model)),
             'inter_annotator_mre_px': float(np.mean(err_loo_human))}}
    by_type = {}
    for k, n in enumerate(names):
        by_type.setdefault(n.split('_')[-1], []).append(err[:, k])
    m['per_type_mre_px'] = {t: float(np.mean(v)) for t, v in by_type.items()}
    return m, rows


def tooth_val_errors(ds, pred):
    """Per tooth-keypoint MRE (original-resolution px) over the dental samples of a val set, or None.

    Works for stage-2 files (6 keypoints) and stage-1 files (dental slots inside the 94-keypoint layout).
    """
    kidx = [k for k, n in enumerate(ds.keypoint_names) if n.startswith('Tooth_')]
    errs = []
    for (im, ann), p in zip(ds.samples, pred):
        if im.get('source') != 'dental':
            continue
        s = np.array(im['sz_origin_wh'], float) / np.array(im['full_size_wh'], float)
        g = np.array(ann['keypoints'], float).reshape(-1, 3)[kidx, :2]
        errs.append(np.linalg.norm((p[kidx] - g) * s, axis=-1))
    if not errs:
        return None
    errs = np.stack(errs)
    names = [ds.keypoint_names[k][len('Tooth_'):] for k in kidx]  # L_C, L_D, L_A, R_C, R_D, R_A
    return {'mre_px': float(errs.mean()), 'per_keypoint_mre_px': {n: float(errs[:, j].mean()) for j, n in enumerate(names)}}
