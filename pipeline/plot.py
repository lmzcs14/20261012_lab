"""Training curves from log.jsonl files.

Per run   -> <run>/plots/training.png: train loss, LR, val error, and val MRE of each tooth keypoint
             (original-resolution px), with LR drops / early stop marked.
Across runs (stage 2, one fold) -> <runs-dir>/plots/:
  keypoints_val.png     one panel per tooth keypoint, val MRE across training, one line per run
  keypoint_<name>.png   the same, one larger figure per keypoint
  keypoints_test.png    final test MRE per keypoint per run (best-on-val checkpoint)

Usage: conda run -n tooth python -m pipeline.plot [--runs-dir runs] [--fold 0]
"""
import argparse
import glob
import json
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# validated categorical slots (dataviz default palette, light mode); identity never by color alone:
# keypoint type -> color + direct label, side -> line style; setup -> color, backbone -> line style
SLOT = ['#2a78d6', '#eb6834', '#1baf7a']
INK, MUTED, GRID = '#1f1f1e', '#6b6a64', '#e6e5e0'
KP_ORDER = ['L_C', 'L_D', 'L_A', 'R_C', 'R_D', 'R_A']
TYPE_NAME = {'C': 'C', 'D': 'D', 'A': 'A (apex)'}
SETUP_ORDER = ['none', 'public', 'public_ours']
SETUP_LABEL = {'none': 'no stage 1', 'public': 'stage 1: public', 'public_ours': 'stage 1: public + ours'}
BACKBONE_STYLE = {'sapiens2_0.4b': ('-', 'Sapiens2'), 'sapiens1_0.3b': ('--', 'Sapiens v1')}

plt.rcParams.update({
    'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb', 'axes.edgecolor': GRID,
    'axes.labelcolor': MUTED, 'axes.titlecolor': INK, 'axes.titlesize': 11, 'axes.labelsize': 9,
    'xtick.color': MUTED, 'ytick.color': MUTED, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8, 'axes.spines.top': False,
    'axes.spines.right': False, 'legend.frameon': False, 'legend.fontsize': 8, 'lines.linewidth': 2,
})


def read_log(path):
    train, val = [], []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            (val if 'val_epe_px' in r else train).append(r)
    if not val:
        return None
    ipe = val[0]['iter'] / (val[0]['epoch'] + 1)  # iterations per epoch
    for r in train:
        r['x'] = r['iter'] / ipe
    for r in val:
        r['x'] = r['epoch'] + 1
    return train, val


def mark_events(ax, val):
    for r in val:
        if 'lr_drop' in r:
            ax.axvline(r['x'], color=MUTED, linestyle=':', linewidth=1)
        if r.get('early_stop'):
            ax.axvline(r['x'], color=INK, linestyle=':', linewidth=1)


def plot_run(run_dir):
    parsed = read_log(os.path.join(run_dir, 'log.jsonl'))
    if parsed is None:
        return False
    train, val = parsed
    has_kp = any('val_tooth_kp_mre_px' in r for r in val)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    (ax_loss, ax_lr), (ax_val, ax_kp) = axes

    ax_loss.plot([r['x'] for r in train], [r['loss'] for r in train], color=SLOT[0])
    ax_loss.set(yscale='log', title='Train heatmap loss', xlabel='epoch')
    ax_lr.plot([r['x'] for r in train], [r['lr'] for r in train], color=SLOT[0])
    ax_lr.set(yscale='log', title='Learning rate (base; blocks scaled by layer decay)', xlabel='epoch')

    xs, ys = [r['x'] for r in val], [r['val_epe_px'] for r in val]
    ax_val.plot(xs, ys, color=SLOT[0])
    best = min(val, key=lambda r: r['val_epe_px'])
    ax_val.plot([best['x']], [best['val_epe_px']], 'o', color=SLOT[0], markersize=8,
                markeredgecolor='#fcfcfb', markeredgewidth=2)
    ax_val.annotate(f"best {best['val_epe_px']:.1f} px @ ep {best['x']}", (best['x'], best['val_epe_px']),
                    xytext=(6, 10), textcoords='offset points', fontsize=8, color=INK)
    ax_val.set(title='Val error, all keypoints (px of each val image)', xlabel='epoch', ylabel='px')
    ax_val.set_ylim(0, min(max(ys), 5 * best['val_epe_px']) * 1.05)

    if has_kp:
        kv = [r for r in val if 'val_tooth_kp_mre_px' in r]
        kx = [r['x'] for r in kv]
        top = 0
        for name in KP_ORDER:
            side, t = name.split('_')
            color = SLOT['CDA'.index(t)]
            ky = [r['val_tooth_kp_mre_px'][name] for r in kv]
            top = max(top, sorted(ky)[len(ky) // 2] * 3)
            ax_kp.plot(kx, ky, color=color, linestyle='-' if side == 'L' else '--',
                       label=f"{'left' if side == 'L' else 'right'} {TYPE_NAME[t]}")
        ax_kp.set_ylim(0, top)
        ax_kp.legend(ncol=2, loc='upper right')
        ax_kp.set(title='Val MRE per tooth keypoint (original-resolution px)', xlabel='epoch', ylabel='px')
    else:
        ax_kp.set_axis_off()
        ax_kp.text(0.5, 0.5, 'no dental teeth in this val set', ha='center', va='center', color=MUTED,
                   transform=ax_kp.transAxes)
    for ax in (ax_loss, ax_lr, ax_val, ax_kp):
        mark_events(ax, val)
    name = '/'.join(os.path.normpath(run_dir).split(os.sep)[-3:])
    fig.suptitle(f'{name}\ngray dotted = LR drop, black dotted = early stop', fontsize=10, color=MUTED)
    os.makedirs(os.path.join(run_dir, 'plots'), exist_ok=True)
    fig.savefig(os.path.join(run_dir, 'plots', 'training.png'), dpi=130)
    plt.close(fig)
    return True


def stage2_runs(runs_dir, fold):
    out = []
    for bb in BACKBONE_STYLE:
        for setup in SETUP_ORDER:
            d = os.path.join(runs_dir, bb, f's2_from_{setup}', f'fold{fold}')
            if os.path.exists(os.path.join(d, 'log.jsonl')):
                parsed = read_log(os.path.join(d, 'log.jsonl'))
                if parsed:
                    out.append(dict(dir=d, bb=bb, setup=setup, val=parsed[1]))
    return out


def run_style(r):
    ls, bb_name = BACKBONE_STYLE[r['bb']]
    return dict(color=SLOT[SETUP_ORDER.index(r['setup'])], linestyle=ls,
                label=f"{bb_name} · {SETUP_LABEL[r['setup']]}")


def plot_keypoint(ax, runs, name):
    top = 0
    for r in runs:
        kv = [v for v in r['val'] if 'val_tooth_kp_mre_px' in v]
        if not kv:
            continue
        ys = [v['val_tooth_kp_mre_px'][name] for v in kv]
        top = max(top, sorted(ys)[len(ys) // 2] * 2.5)
        ax.plot([v['x'] for v in kv], ys, **run_style(r))
    ax.set_ylim(0, top or 1)
    side, t = name.split('_')
    ax.set(title=f"{'Left' if side == 'L' else 'Right'} {TYPE_NAME[t]} ({name})", xlabel='epoch',
           ylabel='val MRE (orig px)')


def plot_comparison(runs_dir, fold):
    runs = stage2_runs(runs_dir, fold)
    if not runs:
        return []
    out_dir = os.path.join(runs_dir, 'plots')
    os.makedirs(out_dir, exist_ok=True)
    written = []

    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), constrained_layout=True, sharex=True)
    for ax, name in zip(axes.ravel(), KP_ORDER):
        plot_keypoint(ax, runs, name)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=3)
    fig.suptitle(f'Stage 2, fold {fold}: val MRE per tooth keypoint across training', fontsize=11, color=INK)
    written.append(os.path.join(out_dir, 'keypoints_val.png'))
    fig.savefig(written[-1], dpi=130)
    plt.close(fig)

    for name in KP_ORDER:
        fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
        plot_keypoint(ax, runs, name)
        ax.legend(loc='upper right')
        written.append(os.path.join(out_dir, f'keypoint_{name}.png'))
        fig.savefig(written[-1], dpi=130)
        plt.close(fig)

    done = [(r, json.load(open(os.path.join(r['dir'], 'done.json')))['test'])
            for r in runs if os.path.exists(os.path.join(r['dir'], 'done.json'))]
    if done:
        fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
        width = 0.8 / len(done)
        for i, (r, t) in enumerate(done):
            st = run_style(r)
            xs = [k + (i - (len(done) - 1) / 2) * width for k in range(len(KP_ORDER))]
            ys = [t['per_keypoint_mre_px'][f'Tooth_{n}'] for n in KP_ORDER]
            ax.bar(xs, ys, width * 0.9, color=st['color'], label=st['label'],
                   hatch='//' if st['linestyle'] == '--' else None, edgecolor='#fcfcfb', linewidth=0)
        ax.set_xticks(range(len(KP_ORDER)), KP_ORDER)
        ax.set(title=f'Test MRE per tooth keypoint, fold {fold} (best-on-val checkpoint; hatched = Sapiens v1)',
               ylabel='orig px')
        fig.legend(loc='outside lower center', ncol=3)
        written.append(os.path.join(out_dir, 'keypoints_test.png'))
        fig.savefig(written[-1], dpi=130)
        plt.close(fig)
    return written


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs-dir', default=os.path.join(ROOT, 'runs'))
    p.add_argument('--fold', type=int, default=0)
    args = p.parse_args()
    n = sum(plot_run(os.path.dirname(f))
            for f in sorted(glob.glob(os.path.join(args.runs_dir, '**', 'log.jsonl'), recursive=True)))
    print(f'per-run plots: {n} (in <run>/plots/training.png)')
    for f in plot_comparison(args.runs_dir, args.fold):
        print(f)


if __name__ == '__main__':
    main()
