# Dental keypoints with Sapiens / Sapiens2 + MedSapiens-style LoRA

Goal: 6 keypoints per tooth (left/right C, D, A) on KH periapical radiographs (`pkl20260405-002/`, 316 images,
409 annotated teeth, 3 annotators, 4 official CV folds).

Method (MedSapiens, `MedSapiens Taking a Pose...pdf`): supervised LoRA on a Sapiens ViT in two stages.
- **Stage 1 (generalist):** heatmap regression on 4 public X-ray landmark datasets (Head, Hand, Chest, Leg),
  optionally plus our dental *training* teeth.
- **Stage 2 (specialist):** continue LoRA on our dental training teeth; pick checkpoint on val; test once.

## Grid (`pipeline/run_grid.py`)

| | stage 1 = none | stage 1 = public | stage 1 = public + our train fold |
|---|---|---|---|
| Sapiens v1 0.3B (MedSapiens backbone) | `s2_from_none` | `s2_from_public` | `s2_from_public_ours` |
| Sapiens2 0.4B | `s2_from_none` | `s2_from_public` | `s2_from_public_ours` |

Currently run on fold 0 only (expand to folds 0-3 once a setup shows a gain). Stage-1 "public" is trained once per backbone; "public_ours" once per fold.

## Leakage rules (`scripts/check_leakage.py` verifies them)
- Test = official fold k. Never used for training or checkpoint selection, in either stage.
- Val (~12% of the other folds, split by patient) is used only for checkpoint selection.
- All crops of one radiograph are in the same split. Public duplicates of held-out images are dropped.
- Known, kept: the official folds are not patient-wise (3-8 patients per fold appear on both sides,
  different teeth). Patient IDs exist only in file names; `teethNN` files carry none.

## Data (`scripts/prepare_splits.py`)
- Each annotated tooth is cropped from its segmentation-mask box + 25% margin (not from keypoints, to avoid a
  position shortcut) and treated as one image with a full-image box, exactly as MedSapiens treats its images.
- Ground truth = mean of the 3 annotators; raw annotator points are kept in `keypoints_annotators`.
- Stage-1 files use 94 keypoints (MedSapiens' 88 + our 6); stage-2 files use our 6.

## Deviations from MedSapiens
- Own PyTorch port (`pipeline/`) instead of their mmpose code (torch 2.4 / mmcv do not support the RTX 5060 Ti).
  Same recipe: LoRA r4 on q/k/v, r8 on attn proj, deconv heatmap head, UDP heatmaps + DARK decoding, MSE,
  AdamW 5e-4, layer decay 0.85, batch 8, flip test.
- Input 512x384 instead of 1024x768 (8 GB GPU; tooth crops are only ~250-400 px tall). Heatmap sigma 2.
- Schedule: instead of fixed LR steps at epochs 170/200 of 210, LR x0.1 when val error stalls for 10 epochs
  (<1% gain), at most twice (same two drops as MedSapiens), then early stop; 210 epochs is only a cap.
  Warmup is 3 epochs instead of 500 iterations (500 iters = ~15 epochs on our ~270 teeth).
- Horizontal flip swaps left/right tooth keypoints (MedSapiens' configs never swap).
- Sapiens2 has separate wq/wk/wv, so it gets three rank-4 adapters where v1 has one on fused qkv.

## Metrics
Errors in original-resolution pixels (`sz_origin`; no mm spacing available): MRE, per keypoint and per type
(C/D/A), SDR at 10/20/30/40 px. Reference: model and each annotator vs the mean of the other two annotators
(inter-annotator MRE is ~24 px).

## Logs and plots
Per run (`runs/<backbone>/<stage>/...`): `args.json`, `log.jsonl` (train loss/LR every 50 iters; every epoch:
val error, val MRE per tooth keypoint in original px, LR drops, early stop), `best.pt`, `done.json`
(final + test metrics), `test_predictions.json`, `plots/training.png`.
Across stage-2 runs (`runs/plots/`): `keypoints_val.png` (val MRE per keypoint across training, one line per
run), `keypoint_<name>.png` (one keypoint, larger), `keypoints_test.png` (final test MRE per keypoint).

New machine: see `SETUP.md` (environment, downloads, where to put the private data). Run order: `RUN_COMMANDS.md`.

## Commands (conda env `tooth`)
```bash
conda run -n tooth python scripts/prepare_splits.py && conda run -n tooth python scripts/check_leakage.py
conda run -n tooth python -m pipeline.run_grid --dry-run          # list runs and their status
conda run -n tooth python -m pipeline.run_grid                    # run everything (resumable)
conda run -n tooth python -m pipeline.run_grid --backbones sapiens2_0.4b --setups public --folds 0
conda run -n tooth python -m pipeline.summarize                   # results table
conda run -n tooth python -m pipeline.plot                        # curves (also refreshed after each grid run)
```
