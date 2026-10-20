"""Verify the splits written by prepare_splits.py contain no test leakage.

Dental samples are per-tooth crops; leakage is checked at the radiograph level (all crops of a radiograph
must be in the same split), plus by file name and pixel hash.

For every fold k, checks (by image file name AND by pixel hash, so renamed duplicates are caught):
  - fold-k test images appear in no stage-1 or stage-2 train/val file used for fold k
  - stage-2 val images are not in stage-2 train, nor in stage-1 train
  - public test images are in no training file
  - every dental image is in exactly one of train/val/test per fold
Also reports patients whose images sit in both train/val and test (official folds are not patient-wise).

Usage: conda run -n tooth python scripts/check_leakage.py
"""
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S = os.path.join(ROOT, 'data', 'splits')
IMG = os.path.join(ROOT, 'data', 'all_images')
PUBLIC_TEST = os.path.join(ROOT, 'data', 'med_sapien', 'med_sapien_coco_test_annotations.json')

_hash = {}


def h(fname):
    if fname not in _hash:
        with open(os.path.join(IMG, fname), 'rb') as f:
            _hash[fname] = hashlib.md5(f.read()).hexdigest()
    return _hash[fname]


def load(*p):
    with open(os.path.join(S, *p)) as f:
        return json.load(f)


def names(coco):
    return {im['file_name'] for im in coco['images']}


def hashes(coco):
    return {h(n) for n in names(coco)}


def main():
    errors = []

    def expect_disjoint(a_name, a, b_name, b):
        for kind, x, y in (('file name', names(a), names(b)), ('pixel hash', hashes(a), hashes(b))):
            if x & y:
                errors.append(f'{a_name} ∩ {b_name}: {len(x & y)} shared by {kind}')

    public_test = load_abs(PUBLIC_TEST)
    pub_train, pub_val = load('public', 'stage1_train.json'), load('public', 'stage1_val.json')
    if any(im['source'] != 'public' for im in pub_train['images'] + pub_val['images']):
        errors.append('public/stage1 contains dental images')
    expect_disjoint('public/stage1_train', pub_train, 'public test', public_test)
    expect_disjoint('public/stage1_val', pub_val, 'public test', public_test)
    expect_disjoint('public/stage1_val', pub_val, 'public/stage1_train', pub_train)

    fold_assign = load('fold_assignment.json')
    stems = lambda c: {im['stem'] for im in c['images'] if im.get('source') == 'dental'}
    for k in range(4):
        fd = f'fold{k}'
        tr, va, te = (load(fd, f'stage2_dental_{n}.json') for n in ('train', 'val', 'test'))
        s1_tr, s1_va = load(fd, 'stage1_public_ours_train.json'), load(fd, 'stage1_public_ours_val.json')
        expected_test = {s for s, v in fold_assign.items() if v['fold'] == k}
        if stems(te) != expected_test:
            errors.append(f'{fd}: test set differs from official fold {k}')
        # partition by radiograph: every crop of one radiograph must be in exactly one split
        if stems(tr) | stems(va) | stems(te) != set(fold_assign) or \
                len(stems(tr)) + len(stems(va)) + len(stems(te)) != len(fold_assign):
            errors.append(f'{fd}: dental train/val/test is not an exact partition of radiographs')
        for split_name, split in (('stage1 train', s1_tr), ('stage1 val', s1_va)):
            if stems(split) & expected_test:
                errors.append(f'{fd}: {split_name} contains crops of test radiographs')
        for split_name, split in (('stage2 train', tr), ('stage2 val', va), ('stage1 train', s1_tr),
                                  ('stage1 val', s1_va), ('public stage1 train', pub_train),
                                  ('public stage1 val', pub_val)):
            expect_disjoint(f'{fd} test', te, f'{fd} {split_name}', split)
        expect_disjoint(f'{fd} stage2 val', va, f'{fd} stage2 train', tr)
        expect_disjoint(f'{fd} stage2 val', va, f'{fd} stage1 train', s1_tr)
        expect_disjoint(f'{fd} stage1 train', s1_tr, 'public test', public_test)
        dental_in_s1 = {im['file_name'] for im in s1_tr['images'] if im['source'] == 'dental'}
        if dental_in_s1 != names(tr):
            errors.append(f'{fd}: dental part of stage1 train != stage2 train')

        pat = lambda c: {im['patient'] for im in c['images']}
        shared = (pat(tr) | pat(va)) & pat(te)
        print(f'{fd}: crops train {len(tr["images"])} / val {len(va["images"])} / test {len(te["images"])}; '
              f'patients in both train/val and test: {len(shared)} {sorted(shared)}')

    if errors:
        print('LEAKAGE FOUND:\n  ' + '\n  '.join(errors))
        sys.exit(1)
    print('OK: no test image (by name or pixel hash) is in any training or validation file.')


def load_abs(p):
    with open(p) as f:
        return json.load(f)


if __name__ == '__main__':
    main()
