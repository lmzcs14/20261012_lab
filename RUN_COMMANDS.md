# Run commands (fold 0)

Run the steps one at a time, in order; each one waits for the previous to finish.
Times are estimates. Early stopping decides the real length, and stage 1 is the most uncertain.

## 0. Setup (once per terminal)
```bash
cd /home/lmz/lab/20261019_sapiens2
conda activate tooth
mkdir -p runs
tmux new -s tooth        # detach: Ctrl+b then d; reattach: tmux attach -t tooth
```

## Sapiens2

### 1. Baseline: no stage 1, stage 2 only (~35 min)
```bash
python -m pipeline.run_grid --backbones sapiens2_0.4b --setups none 2>&1 | tee runs/1_s2_baseline.log
```

### 2. Mode C: stage 1 on public, then stage 2 (~4 h)
```bash
python -m pipeline.run_grid --backbones sapiens2_0.4b --setups public 2>&1 | tee runs/2_s2_C_public.log
```

### 3. Mode D: stage 1 on public + our training teeth, then stage 2 (~5 h)
```bash
python -m pipeline.run_grid --backbones sapiens2_0.4b --setups public_ours 2>&1 | tee runs/3_s2_D_public_ours.log
```

## Sapiens v1 (MedSapiens' backbone)

### 4. Baseline: no stage 1, stage 2 only (~25 min)
```bash
python -m pipeline.run_grid --backbones sapiens1_0.3b --setups none 2>&1 | tee runs/4_s1_baseline.log
```

### 5. Mode A: stage 1 on public, then stage 2 (~3 h)
```bash
python -m pipeline.run_grid --backbones sapiens1_0.3b --setups public 2>&1 | tee runs/5_s1_A_public.log
```

### 6. Mode B: stage 1 on public + our training teeth, then stage 2 (~3.5 h)
```bash
python -m pipeline.run_grid --backbones sapiens1_0.3b --setups public_ours 2>&1 | tee runs/6_s1_B_public_ours.log
```

## Training at MedSapiens' image size (768x1024)
Add `--input-size 768 1024` to any command; results go to `runs_768x1024/` (kept apart from `runs/`), e.g. mode C:
```bash
mkdir -p runs_768x1024
python -m pipeline.run_grid --input-size 768 1024 --backbones sapiens2_0.4b --setups public 2>&1 | tee runs_768x1024/C.log
```
Plots / table for these: add `--runs-dir runs_768x1024` to `pipeline.plot` / `pipeline.summarize`.
Each epoch costs roughly 4-5x more. Batch 8 at this size needs ~10 GB (estimate); on the 8 GB card add
`-- --batch-size 2 --accum-steps 4` (same effective batch of 8).

## Checking progress and results (any time, from another terminal)
```bash
python -m pipeline.run_grid --dry-run       # which runs are done / todo
python -m pipeline.summarize                # test results table of finished runs
python -m pipeline.plot                     # redraw plots (also done automatically after each run)
tail -f runs/sapiens2_0.4b/s1_public/log.jsonl   # follow a running job live
```
Plots: `runs/<backbone>/<stage>/plots/training.png` per run; comparisons in `runs/plots/`.

## If something stops halfway
Re-run the same command. Finished runs are skipped. The interrupted run starts again from the beginning.
