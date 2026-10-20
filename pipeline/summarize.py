"""Collect stage-2 test results into one table: mean +- std over folds, per backbone x setup.

Usage: conda run -n tooth python -m pipeline.summarize [--runs-dir runs]
"""
import argparse
import glob
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs-dir', default=os.path.join(ROOT, 'runs'))
    args = p.parse_args()
    groups = {}
    for f in sorted(glob.glob(os.path.join(args.runs_dir, '*', 's2_from_*', 'fold*', 'done.json'))):
        bb, setup = f.split(os.sep)[-4], f.split(os.sep)[-3][len('s2_from_'):]
        groups.setdefault((bb, setup), []).append(json.load(open(f))['test'])
    if not groups:
        print('no finished stage-2 runs')
        return
    fmt = lambda v: f'{np.mean(v):.1f} ± {np.std(v):.1f}'
    first = next(iter(groups.values()))[0]
    sdr_keys, types = list(first['sdr']), list(first['per_type_mre_px'])
    head = ['backbone', 'stage 1', 'folds', 'MRE px'] + [f'{t} MRE' for t in types] + \
           [f'SDR {k}' for k in sdr_keys] + ['MRE vs 2-reader ref', 'inter-annotator']
    lines = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    for (bb, setup), ms in sorted(groups.items()):
        row = [bb, setup, str(len(ms)), fmt([m['mre_px'] for m in ms])]
        row += [fmt([m['per_type_mre_px'][t] for m in ms]) for t in types]
        row += [f"{100 * np.mean([m['sdr'][k] for m in ms]):.1f}%" for k in sdr_keys]
        ref = [m['reference_leave_one_annotator_out'] for m in ms]
        row += [fmt([r['model_mre_px'] for r in ref]), fmt([r['inter_annotator_mre_px'] for r in ref])]
        lines.append('| ' + ' | '.join(row) + ' |')
    print('Errors in original-resolution pixels; mean ± std over folds.\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
