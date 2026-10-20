"""Train one stage (LoRA + heatmap head), select the best checkpoint on val, then evaluate test once.

Stage 1 (generalist):  --init pretrained, train/val = stage-1 COCO files (94 keypoints).
Stage 2 (specialist):  --init <stage-1 run>/best.pt --skip-final-layer, train/val/test = dental files (6 kps);
                       or --init pretrained for the no-stage-1 baseline.

Example:
  conda run -n tooth python -m pipeline.train --backbone sapiens2_0.4b \
      --train-ann data/splits/fold0/stage2_dental_train.json --val-ann data/splits/fold0/stage2_dental_val.json \
      --test-ann data/splits/fold0/stage2_dental_test.json --init pretrained --out runs/debug
"""
import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .codec import UDPHeatmap
from .data import KeypointDataset, collate
from .evaluate import dental_metrics, predict, tooth_val_errors
from .models import PoseModel, heatmap_loss, param_groups

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--backbone', required=True, choices=['sapiens1_0.3b', 'sapiens2_0.4b'])
    p.add_argument('--train-ann', required=True)
    p.add_argument('--val-ann', required=True)
    p.add_argument('--test-ann', default=None, help='evaluated once, with the best-on-val checkpoint')
    p.add_argument('--img-dir', default=os.path.join(ROOT, 'data', 'all_images'))
    p.add_argument('--init', default='pretrained', help="'pretrained' or path to a trainable checkpoint (.pt)")
    p.add_argument('--skip-final-layer', action='store_true', help='re-init head.final_layer (stage 2)')
    p.add_argument('--out', required=True)
    # MedSapiens recipe; image size reduced from 768x1024 (8 GB GPU)
    p.add_argument('--input-size', type=int, nargs=2, default=[384, 512], metavar=('W', 'H'))
    p.add_argument('--sigma', type=float, default=2.0, help='heatmap sigma at 1/4 input res')
    p.add_argument('--epochs', type=int, default=210, help='maximum epochs (plateau schedule usually stops earlier)')
    p.add_argument('--schedule', choices=['plateau', 'step'], default='plateau',
                   help='plateau: LR x0.1 when val stalls, stop after max-lr-drops; step: MedSapiens milestones')
    p.add_argument('--milestones', type=int, nargs='*', default=[170, 200], help='step schedule only')
    p.add_argument('--patience', type=int, default=10, help='val evaluations without improvement before LR drop/stop')
    p.add_argument('--min-delta', type=float, default=0.01, help='relative val improvement that resets patience')
    p.add_argument('--max-lr-drops', type=int, default=2, help='MedSapiens drops LR twice (x0.1 each)')
    p.add_argument('--batch-size', type=int, default=8, help='MedSapiens: 4 GPUs x 2')
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--weight-decay', type=float, default=0.1)
    p.add_argument('--layer-decay', type=float, default=0.85)
    p.add_argument('--warmup-epochs', type=float, default=3,
                   help='linear warmup length; MedSapiens used 500 iters, i.e. ~15 epochs on our 270 teeth')
    p.add_argument('--clip-grad', type=float, default=1.0)
    p.add_argument('--val-every', type=int, default=1, help='epochs between val evaluations')
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--no-grad-checkpoint', action='store_true')
    p.add_argument('--max-iters', type=int, default=0, help='debug: stop after this many iterations')
    return p.parse_args(argv)


def flip_pairs_of(ann_file):
    with open(ann_file) as f:
        cat = json.load(f)['categories'][0]
    return cat.get('flip_pairs') or cat.get('flip_pairs_tooth') or [], len(cat['keypoints'])


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    if os.path.exists(os.path.join(args.out, 'done.json')):
        print(f'{args.out} already finished, skipping')
        return
    with open(os.path.join(args.out, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    dev = torch.device('cuda')

    flip_pairs, num_kp = flip_pairs_of(args.train_ann)
    w, h = args.input_size
    codec = UDPHeatmap((w, h), (w // 4, h // 4), args.sigma)
    train_ds = KeypointDataset(args.train_ann, args.img_dir, codec, train=True, flip_pairs=flip_pairs)
    val_ds = KeypointDataset(args.val_ann, args.img_dir, codec, train=False, flip_pairs=flip_pairs)
    assert val_ds.num_keypoints == num_kp
    loader = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=args.workers, collate_fn=collate,
                        persistent_workers=args.workers > 0, pin_memory=True)

    model = PoseModel(args.backbone, num_kp, (h, w), grad_checkpoint=not args.no_grad_checkpoint)
    if args.init != 'pretrained':
        n = model.load_trainable(torch.load(args.init, map_location='cpu'), skip_final_layer=args.skip_final_layer)
        print(f'loaded {n} tensors from {args.init}')
    model.to(dev)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'trainable params: {n_train / 1e6:.2f}M; train {len(train_ds)} / val {len(val_ds)} samples')

    opt = torch.optim.AdamW(param_groups(model, args.lr, args.weight_decay, args.layer_decay),
                            lr=args.lr, betas=(0.9, 0.999))

    warmup_iters = max(1, round(args.warmup_epochs * len(loader)))
    lr_scale = 1.0  # plateau schedule: multiplied by 0.1 on each drop

    def lr_factor(it, epoch):
        warm = min(1.0, 0.001 + (1 - 0.001) * it / warmup_iters)
        if args.schedule == 'step':
            return warm * 0.1 ** sum(epoch >= m for m in args.milestones)
        return warm * lr_scale

    log = open(os.path.join(args.out, 'log.jsonl'), 'w')  # restarts begin from scratch
    best = math.inf
    stale, drops, stop = 0, 0, False  # plateau bookkeeping
    it = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        for batch in loader:
            f = lr_factor(it, epoch)
            for g in opt.param_groups:
                g['lr'] = g['initial_lr'] * f
            with torch.autocast('cuda', dtype=torch.bfloat16):
                pred = model(batch['img'].to(dev, non_blocking=True))
            loss = heatmap_loss(pred.float(), batch['heatmaps'].to(dev), batch['weights'].to(dev))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.clip_grad)
            opt.step()
            it += 1
            if it % 50 == 0:
                rec = dict(epoch=epoch, iter=it, loss=loss.item(), lr=args.lr * f,
                           sec=round(time.time() - t0), mem_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2))
                print(json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + '\n')
            if args.max_iters and it >= args.max_iters:
                break
        last = epoch == args.epochs - 1 or (args.max_iters and it >= args.max_iters)
        if (epoch + 1) % args.val_every == 0 or last:
            val = predict(model, val_ds, dev, flip_pairs)
            epe = val['epe']
            rec = dict(epoch=epoch, iter=it, val_epe_px=epe)
            tooth = tooth_val_errors(val_ds, val['pred'])  # per-keypoint curves for plotting
            if tooth:
                rec['val_tooth_mre_px'] = tooth['mre_px']
                rec['val_tooth_kp_mre_px'] = tooth['per_keypoint_mre_px']
            improved = epe < best * (1 - args.min_delta)
            if epe < best:
                best = epe
                torch.save(model.trainable_state_dict(), os.path.join(args.out, 'best.pt'))
                rec['best'] = True
            if args.schedule == 'plateau' and it >= warmup_iters:  # warmup is noisy, don't count it
                stale = 0 if improved else stale + 1
                if stale >= args.patience:
                    stale = 0
                    if drops < args.max_lr_drops:
                        drops += 1
                        lr_scale *= 0.1
                        rec['lr_drop'] = drops
                    else:
                        stop = True
                        rec['early_stop'] = True
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + '\n')
            log.flush()
        if last or stop:
            break

    # final evaluation with the best-on-val checkpoint; test is touched only here
    model.load_trainable(torch.load(os.path.join(args.out, 'best.pt'), map_location='cpu'))
    result = {'best_val_epe_px': best, 'epochs_run': epoch + 1, 'iters': it}
    if args.test_ann:
        test_ds = KeypointDataset(args.test_ann, args.img_dir, codec, train=False, flip_pairs=flip_pairs)
        out = predict(model, test_ds, dev, flip_pairs)
        result['test_epe_px'] = out['epe']
        if all(im.get('source') == 'dental' for im, _ in test_ds.samples):
            metrics, rows = dental_metrics(test_ds, out['pred'])
            result['test'] = metrics
            with open(os.path.join(args.out, 'test_predictions.json'), 'w') as f:
                json.dump(rows, f)
    with open(os.path.join(args.out, 'done.json'), 'w') as f:
        json.dump(result, f, indent=1)
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
