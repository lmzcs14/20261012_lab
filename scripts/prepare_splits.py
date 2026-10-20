"""Build leakage-free COCO keypoint splits for the two stage-1 setups.

Setup "public":      stage-1 trains on the 4 MedSapiens public datasets only (one run, reused by all folds).
Setup "public_ours": stage-1 trains on public + our dental TRAIN folds (one run per fold).
Stage 2 (both setups) trains on our dental train split of the fold and is tested on the held-out fold.

For fold k: test = official fold k; the other 3 folds are split patient-wise into train / val.
Test images never appear in any stage-1 or stage-2 train/val file (checked by check_leakage.py).

Usage: conda run -n tooth python scripts/prepare_splits.py
"""
import glob
import hashlib
import json
import os
import pickle
import random
import re

import cv2
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKL_DIR = os.path.join(ROOT, 'pkl20260405-002')
FOLD_XLSX = os.path.join(PKL_DIR, 'KHdental 根尖片影像 20260406', 'PA_cross_data_316x.xlsx')
PUBLIC_DIR = os.path.join(ROOT, 'data', 'med_sapien')
OUT = os.path.join(ROOT, 'data', 'splits')
DENTAL_IMG_DIR = os.path.join(ROOT, 'data', 'dental', 'Images')
ALL_IMG_DIR = os.path.join(ROOT, 'data', 'all_images')

VAL_FRAC = 0.12
SEED = 0
CROP_PAD = 0.25  # margin around the tooth-mask box, as a fraction of its size (99% of teeth need <= 0.12)

# KP row order from the dataset readme: left C, D, A, right C, D, A
TOOTH_KPTS = ['Tooth_L_C', 'Tooth_L_D', 'Tooth_L_A', 'Tooth_R_C', 'Tooth_R_D', 'Tooth_R_A']
TOOTH_FLIP_PAIRS = [[0, 3], [1, 4], [2, 5]]
OUTSIDE_ANNOTATOR_PTS = []


def patient_id(stem):
    """Patient key from file name; 'teethNN' carries no patient info, so each is its own patient."""
    if re.fullmatch(r'\d{8}_\d{2}', stem):
        return 'A' + stem[:8]
    m = re.fullmatch(r'(\d{3})\.?-\d{2}(jpg)?', stem)  # also '073.-11jpg'
    if m:
        return 'B' + m.group(1)
    return 'T' + stem


def load_folds(stems):
    def norm(s):
        return re.sub(r'[^0-9a-z]', '', str(s).lower().replace('jpg', ''))

    by_norm = {norm(s): s for s in stems}
    sheet = pd.read_excel(FOLD_XLSX, header=None)  # first row is data, not a header
    fold = {}
    for col in sheet.columns:
        for v in sheet[col].dropna():
            s = str(v).strip()
            s = s if s in stems else by_norm[norm(s)]
            assert s not in fold, f'{s} listed twice'
            fold[s] = int(col)
    assert set(fold) == set(stems), 'fold file does not cover every pkl'
    return fold


def match_mask(masks, pts, w):
    """Index of the annotation mask belonging to a tooth: most keypoints within a slightly dilated mask,
    tie-broken by distance from keypoint centre to mask centroid (KP and mask indices do not correspond)."""
    r = max(3, int(0.02 * w))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    xi = np.clip(pts[:, 0].round().astype(int), 0, masks[0].shape[1] - 1)
    yi = np.clip(pts[:, 1].round().astype(int), 0, masks[0].shape[0] - 1)
    scores = []
    for i, m in enumerate(masks):
        hits = cv2.dilate(m.astype(np.uint8), ker)[yi, xi].sum()
        dist = np.linalg.norm(np.argwhere(m)[:, ::-1].mean(0) - pts.mean(0))
        scores.append((hits, -dist, i))
    return max(scores)[2]


def build_dental(stems):
    """Crop every annotated tooth into its own image (one tooth = one sample, like MedSapiens).

    The crop is the tooth's segmentation-mask box plus CROP_PAD margin, NOT derived from the keypoints,
    so keypoint positions inside the crop carry no shortcut. Keypoints are shifted into crop coordinates.
    Returns {stem: [crop record, ...]}.
    """
    os.makedirs(DENTAL_IMG_DIR, exist_ok=True)
    records = {}
    for stem in stems:
        with open(os.path.join(PKL_DIR, stem + '.pkl'), 'rb') as f:
            d = pickle.load(f)
        im = d['im']
        h, w = im.shape
        masks = [m['mask'] for m in d['annotation_masks']]
        per_ann = np.stack([np.asarray(a, float) for a in d['KP']])  # (3 annotators, n_teeth, 6, 2)
        crops, used = [], set()
        for j in range(per_ann.shape[1]):
            pts = per_ann[:, j]  # (3, 6, 2)
            mi = match_mask(masks, pts.mean(0), w)
            assert mi not in used, f'{stem}: two teeth matched mask {mi}'
            used.add(mi)
            ys, xs = np.nonzero(masks[mi])
            bw, bh = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
            x0 = max(0, int(xs.min() - CROP_PAD * bw))
            y0 = max(0, int(ys.min() - CROP_PAD * bh))
            x1 = min(w, int(np.ceil(xs.max() + 1 + CROP_PAD * bw)))
            y1 = min(h, int(np.ceil(ys.max() + 1 + CROP_PAD * bh)))
            local = pts - [x0, y0]
            inside = lambda q: (q >= 0).all(-1) & (q[..., 0] < x1 - x0) & (q[..., 1] < y1 - y0)
            assert inside(local.mean(0)).all(), f'{stem} tooth {j}: ground-truth keypoint outside crop'
            # single-annotator points may fall just outside; they are kept as-is (metadata only)
            OUTSIDE_ANNOTATOR_PTS.append(int((~inside(local)).sum()))
            fname = f'Tooth_{stem}_t{j}.png'
            cv2.imwrite(os.path.join(DENTAL_IMG_DIR, fname), im[y0:y1, x0:x1])
            crops.append({'file_name': fname, 'height': y1 - y0, 'width': x1 - x0, 'tooth_index': j,
                          'crop_box_xyxy': [x0, y0, x1, y1], 'sz_origin': list(d['sz_origin']),
                          'full_size_wh': [w, h],
                          'mean': local.mean(0), 'annotators': local})
        records[stem] = crops
    return records


def dedup_public(public):
    """Drop public train/val images that are pixel-identical to a public test (or, for val, train) image.

    The MedSapiens release has such duplicates (e.g. Head_038 == Head_283); we keep the test side untouched.
    """
    def md5(fname):
        with open(os.path.join(PUBLIC_DIR, 'Images', fname), 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()

    hashes = {s: {im['id']: md5(im['file_name']) for im in c['images']} for s, c in public.items()}
    banned = {'train': set(hashes['test'].values()),
              'val': set(hashes['test'].values()) | set(hashes['train'].values())}
    dropped = []
    for s in ('train', 'val'):
        bad = {i for i, hsh in hashes[s].items() if hsh in banned[s]}
        c = public[s]
        dropped += [f'{s}:{im["file_name"]}' for im in c['images'] if im['id'] in bad]
        c['images'] = [im for im in c['images'] if im['id'] not in bad]
        c['annotations'] = [a for a in c['annotations'] if a['image_id'] not in bad]
    return dropped


def kp_list(xy, n_total, offset):
    kps = [0.0] * (3 * n_total)
    for i, (x, y) in enumerate(xy):
        kps[3 * (offset + i):3 * (offset + i) + 3] = [float(x), float(y), 2]
    return kps


def dental_coco(stems, records, n_total, offset, category):
    """One image + one full-image-box annotation per cropped tooth."""
    images, anns = [], []
    for stem in sorted(stems):
        for c in records[stem]:
            img_id = len(images) + 1
            images.append({'id': img_id, 'file_name': c['file_name'], 'height': c['height'],
                           'width': c['width'], 'source': 'dental', 'stem': stem, 'patient': patient_id(stem),
                           'tooth_index': c['tooth_index'], 'crop_box_xyxy': c['crop_box_xyxy'],
                           'full_size_wh': c['full_size_wh'],
                           # (W, H) of the original radiograph; pkl images are downscaled by 1-2x
                           'sz_origin_wh': c['sz_origin']})
            anns.append({'id': 0, 'image_id': img_id, 'category_id': 1,
                         'keypoints': kp_list(c['mean'], n_total, offset), 'num_keypoints': 6,
                         'bbox': [0, 0, c['width'], c['height']], 'area': c['width'] * c['height'],
                         'iscrowd': 0,
                         # raw per-annotator 6x2 points (A, B, C) in crop coords, for inter-observer analysis
                         'keypoints_annotators': c['annotators'].tolist()})
    return {'images': images, 'annotations': anns, 'categories': [category]}


def pad_public(coco, n_total):
    """Append zero (invisible) dental keypoint slots to every public annotation."""
    out = json.loads(json.dumps(coco))
    for a in out['annotations']:
        a['keypoints'] = a['keypoints'] + [0.0] * (3 * (n_total - len(a['keypoints']) // 3))
    for im in out['images']:
        im['source'] = 'public'
    out['categories'][0]['keypoints'] = out['categories'][0]['keypoints'] + TOOTH_KPTS
    out['categories'][0]['name'] = out['categories'][0]['supercategory'] = 'Combined X-ray + Tooth'
    return out


def merge(a, b):
    """Concatenate two COCO dicts, renumbering image and annotation ids."""
    out = {'images': [], 'annotations': [], 'categories': a['categories']}
    for src in (a, b):
        remap = {}
        for im in src['images']:
            remap[im['id']] = len(out['images']) + 1
            out['images'].append(dict(im, id=remap[im['id']]))
        for an in src['annotations']:
            out['annotations'].append(dict(an, image_id=remap[an['image_id']]))
    return out


def renumber(coco):
    for i, a in enumerate(coco['annotations'], start=1):
        a['id'] = i
    return coco


def dump(coco, *path):
    p = os.path.join(OUT, *path)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'w') as f:
        json.dump(renumber(coco), f)
    return len(coco['images']), len(coco['annotations'])


def link_images():
    os.makedirs(ALL_IMG_DIR, exist_ok=True)
    for src_dir in (os.path.join(PUBLIC_DIR, 'Images'), DENTAL_IMG_DIR):
        for f in os.listdir(src_dir):
            dst = os.path.join(ALL_IMG_DIR, f)
            if not os.path.lexists(dst):
                os.symlink(os.path.relpath(os.path.join(src_dir, f), ALL_IMG_DIR), dst)


def main():
    stems = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(PKL_DIR, '*.pkl')))
    fold = load_folds(stems)
    records = build_dental(stems)
    link_images()

    public = {s: json.load(open(os.path.join(PUBLIC_DIR, f'med_sapien_coco_{s}_annotations.json')))
              for s in ('train', 'val', 'test')}
    dropped = dedup_public(public)
    n_public = len(public['train']['categories'][0]['keypoints'])
    n_total = n_public + len(TOOTH_KPTS)
    combined_cat = pad_public(public['train'], n_total)['categories'][0]
    tooth_cat = {'id': 1, 'name': 'Tooth', 'supercategory': 'Tooth', 'keypoints': TOOTH_KPTS,
                 'skeleton': [], 'flip_pairs': TOOTH_FLIP_PAIRS}
    combined_cat['flip_pairs_tooth'] = [[a + n_public, b + n_public] for a, b in TOOTH_FLIP_PAIRS]

    log = [f'tooth crops: {sum(map(len, records.values()))}; single-annotator points outside their crop: '
           f'{sum(OUTSIDE_ANNOTATOR_PTS)} in {sum(x > 0 for x in OUTSIDE_ANNOTATOR_PTS)} teeth',
           f'dropped public duplicates of held-out images: {dropped}']
    # Setup "public": identical for every fold, contains no dental images at all.
    for s in ('train', 'val'):
        n = dump(dict(pad_public(public[s], n_total), categories=[combined_cat]), 'public', f'stage1_{s}.json')
        log.append(f'public/stage1_{s}.json: {n[0]} images, {n[1]} annotations')

    rng = random.Random(SEED)
    for k in range(4):
        test = [s for s in stems if fold[s] == k]
        rest = [s for s in stems if fold[s] != k]
        # patient-wise val split inside the training folds
        patients = sorted({patient_id(s) for s in rest})
        rng.shuffle(patients)
        val_p = set(patients[:round(VAL_FRAC * len(patients))])
        val = [s for s in rest if patient_id(s) in val_p]
        train = [s for s in rest if patient_id(s) not in val_p]

        fd = f'fold{k}'
        for name, sub in (('train', train), ('val', val), ('test', test)):
            n = dump(dental_coco(sub, records, 6, 0, tooth_cat), fd, f'stage2_dental_{name}.json')
            log.append(f'{fd}/stage2_dental_{name}.json: {n[0]} tooth crops from {len(sub)} radiographs')
        for name, sub in (('train', train), ('val', val)):
            dent = dental_coco(sub, records, n_total, n_public, combined_cat)
            pub = pad_public(public[name], n_total)
            n = dump(dict(merge(pub, dent), categories=[combined_cat]), fd, f'stage1_public_ours_{name}.json')
            log.append(f'{fd}/stage1_public_ours_{name}.json: {n[0]} images, {n[1]} annotations')

    with open(os.path.join(OUT, 'fold_assignment.json'), 'w') as f:
        json.dump({s: {'fold': fold[s], 'patient': patient_id(s)} for s in stems}, f, indent=1)
    print('\n'.join(log))


if __name__ == '__main__':
    main()
