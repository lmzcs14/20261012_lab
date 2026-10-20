# Setting up on a new machine

Tested on Linux / WSL2 with an RTX 5060 Ti (8 GB). Any NVIDIA GPU that CUDA 12.8 PyTorch supports
(RTX 30/40/50 series) should work.

## 1. Code
```bash
git clone <this repo> 20261019_sapiens2
cd 20261019_sapiens2
```

## 2. Python environment (conda env `tooth`)
```bash
conda create -y -n tooth python=3.11
conda activate tooth
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"   # True, <your GPU>
```

## 3. Public data, backbone code and pretrained weights
```bash
bash scripts/setup.sh
```
This downloads (about 8 GB, plus 2.8 GB of weights):

| What | Source | Goes to |
|---|---|---|
| Sapiens2 repo at a pinned commit (we use its standalone `sapiens.py` / `sapiens2.py` backbones) | github.com/facebookresearch/sapiens2 | `third_party/sapiens2/` |
| Sapiens v1 0.3B and Sapiens2 0.4B pretrained weights | Hugging Face `facebook/sapiens-pretrain-0.3b`, `facebook/sapiens2-pretrain-0.4b` | `checkpoints/pretrained/` |
| MedSapiens public landmark datasets: Head (ISBI 2015 cephalometric), Hand (Digital Hand Atlas), Chest (Shenzhen), Leg (BMPLE), as COCO keypoints with MedSapiens' train/val/test splits | Google Drive link in the [MedSapiens README](https://github.com/xmed-lab/MedSapiens) (`med_sapien.zip`) | `data/med_sapien/` |

If the Google Drive download fails (quota limits are common), open
`https://drive.google.com/uc?id=1G_3Gir_MJ2Hbm4A2Oqwcy579Mgo2hpYQ` in a browser, save the file as
`data/raw_download/med_sapien.zip`, and re-run `bash scripts/setup.sh`; it will only unzip.
Check the original datasets' licenses before publishing results on them.

## 4. Our dental data (private, not in git)
Copy the folder `pkl20260405-002/` (316 `.pkl` files plus `KHdental 根尖片影像 20260406/` with the fold
spreadsheet) into the repo root, so that `pkl20260405-002/teeth08.pkl` exists. It is in `.gitignore`;
never commit it.

## 5. Build the splits and check for leakage
```bash
python scripts/prepare_splits.py     # tooth crops -> data/dental/, COCO files -> data/splits/
python scripts/check_leakage.py      # must end with "OK: no test image ... is in any training or validation file."
```
The splits are deterministic (seed 0), so every machine gets identical files.

## 6. Train
See `RUN_COMMANDS.md`. With two GPUs in one machine, pick one per terminal, e.g.
`CUDA_VISIBLE_DEVICES=1 python -m pipeline.run_grid ...`. Use different `--backbones` / `--setups` in each
terminal so they don't train the same run.
