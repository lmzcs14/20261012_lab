"""Run the experiment grid in dependency order; finished runs (done.json) are skipped, so it can be resumed.

Backbones: sapiens1_0.3b (MedSapiens' backbone), sapiens2_0.4b.
Stage-2 init ("setup"):
  none         stage 2 directly from the pretrained backbone (does stage 1 help at all?)
  public       stage 1 on the 4 public datasets only        -> runs/<bb>/s1_public            (one run)
  public_ours  stage 1 on public + this fold's dental train -> runs/<bb>/s1_public_ours/fold<k> (one per fold)
Stage 2 always: dental train/val of fold k, test on official fold k -> runs/<bb>/s2_from_<setup>/fold<k>

Defaults: fold 0 only; every run uses the plateau schedule (LR x0.1 twice on val stalls, then early stop),
capped at MedSapiens' 210 epochs.

Examples:
  conda run -n tooth python -m pipeline.run_grid --dry-run
  conda run -n tooth python -m pipeline.run_grid --folds 0 1 2 3          # expand to full CV
  conda run -n tooth python -m pipeline.run_grid --input-size 768 1024    # MedSapiens' image size -> runs_768x1024/
  conda run -n tooth python -m pipeline.run_grid --backbones sapiens2_0.4b --setups public --folds 0
  conda run -n tooth python -m pipeline.run_grid -- --epochs 20      # args after -- go to every train run
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLITS = os.path.join(ROOT, 'data', 'splits')
RUNS = os.path.join(ROOT, 'runs')  # overridden by --runs-dir / --input-size
DEFAULT_INPUT_SIZE = [384, 512]  # W H, same default as pipeline.train
INPUT_SIZE = []  # extra train args set from --input-size


def schedule(max_epochs):
    """Plateau schedule with early stopping (see pipeline.train); max_epochs is only a cap."""
    return ['--epochs', str(max_epochs), '--schedule', 'plateau']


S1_SCHEDULE = schedule(210)  # overridden by --s1-epochs
S2_SCHEDULE = schedule(210)  # overridden by --s2-epochs


def stage1(bb, setup, fold):
    if setup == 'public':
        out, d, prefix = os.path.join(RUNS, bb, 's1_public'), 'public', 'stage1'
    else:
        out, d, prefix = os.path.join(RUNS, bb, 's1_public_ours', f'fold{fold}'), f'fold{fold}', 'stage1_public_ours'
    args = ['--backbone', bb, '--out', out, '--init', 'pretrained',
            '--train-ann', os.path.join(SPLITS, d, f'{prefix}_train.json'),
            '--val-ann', os.path.join(SPLITS, d, f'{prefix}_val.json')] + S1_SCHEDULE + INPUT_SIZE
    return out, args


def stage2(bb, setup, fold):
    d = os.path.join(SPLITS, f'fold{fold}')
    out = os.path.join(RUNS, bb, f's2_from_{setup}', f'fold{fold}')
    args = ['--backbone', bb, '--out', out,
            '--train-ann', os.path.join(d, 'stage2_dental_train.json'),
            '--val-ann', os.path.join(d, 'stage2_dental_val.json'),
            '--test-ann', os.path.join(d, 'stage2_dental_test.json')] + S2_SCHEDULE + INPUT_SIZE
    if setup == 'none':
        args += ['--init', 'pretrained']
    else:
        s1_out, _ = stage1(bb, setup, fold)
        args += ['--init', os.path.join(s1_out, 'best.pt'), '--skip-final-layer']
    return out, args


def plan(backbones, setups, folds):
    jobs, seen = [], set()
    for bb in backbones:
        for setup in setups:
            for k in folds:
                if setup != 'none':
                    out, a = stage1(bb, setup, k)
                    if out not in seen:
                        seen.add(out)
                        jobs.append((out, a))
                jobs.append(stage2(bb, setup, k))
    return jobs


def main():
    argv = sys.argv[1:]
    extra = argv[argv.index('--') + 1:] if '--' in argv else []
    argv = argv[:argv.index('--')] if '--' in argv else argv
    p = argparse.ArgumentParser()
    p.add_argument('--backbones', nargs='+', default=['sapiens2_0.4b', 'sapiens1_0.3b'])
    p.add_argument('--setups', nargs='+', default=['none', 'public', 'public_ours'],
                   choices=['none', 'public', 'public_ours'])
    p.add_argument('--folds', nargs='+', type=int, default=[0], help='fold 0 only until a setup shows a gain')
    p.add_argument('--s1-epochs', type=int, default=210, help='cap; early stopping usually ends sooner')
    p.add_argument('--s2-epochs', type=int, default=210, help='cap; early stopping usually ends sooner')
    p.add_argument('--input-size', type=int, nargs=2, default=DEFAULT_INPUT_SIZE, metavar=('W', 'H'),
                   help='network input size for every run; MedSapiens used 768 1024')
    p.add_argument('--runs-dir', default=None,
                   help='default: runs/ at the default size, runs_<W>x<H>/ otherwise (keeps sizes apart)')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args(argv)
    w, h = args.input_size
    default_dir = RUNS if args.input_size == DEFAULT_INPUT_SIZE else f'{RUNS}_{w}x{h}'
    globals()['RUNS'] = os.path.abspath(args.runs_dir or default_dir)
    globals()['INPUT_SIZE'] = ['--input-size', str(w), str(h)]
    globals()['S1_SCHEDULE'] = schedule(args.s1_epochs)
    globals()['S2_SCHEDULE'] = schedule(args.s2_epochs)

    for out, a in plan(args.backbones, args.setups, args.folds):
        status = 'done' if os.path.exists(os.path.join(out, 'done.json')) else 'todo'
        print(f'[{status}] {os.path.relpath(out, ROOT)}', flush=True)
        if args.dry_run or status == 'done':
            continue
        subprocess.run([sys.executable, '-m', 'pipeline.train', *a, *extra], cwd=ROOT, check=True)
        # refresh per-run and comparison plots after every finished run
        subprocess.run([sys.executable, '-m', 'pipeline.plot', '--runs-dir', RUNS, '--fold', str(args.folds[0])],
                       cwd=ROOT, check=False)


if __name__ == '__main__':
    main()
