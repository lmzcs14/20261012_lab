#!/usr/bin/env bash
# Download everything the pipeline needs except our private dental data (see SETUP.md).
# Safe to re-run: finished steps are skipped.
#   1. Sapiens2 repo (standalone Sapiens v1 / Sapiens2 backbone code), pinned commit -> third_party/sapiens2
#   2. Pretrained backbones (Hugging Face, public)                                   -> checkpoints/pretrained
#   3. MedSapiens public landmark datasets (Google Drive, ~5.1 GB zip, ~8.7 GB unzipped) -> data/med_sapien
# Run from the repo root inside the `tooth` env: bash scripts/setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."

SAPIENS2_COMMIT=7e5bae88456ac418ff0e58e74106c9fe192055d4
MEDSAPIENS_DATA_GDRIVE_ID=1G_3Gir_MJ2Hbm4A2Oqwcy579Mgo2hpYQ  # from github.com/xmed-lab/MedSapiens README

echo "== 1/3 Sapiens2 code"
if [ ! -f third_party/sapiens2/sapiens/backbones/standalone/sapiens2.py ]; then
    mkdir -p third_party/sapiens2
    git -C third_party/sapiens2 init -q
    git -C third_party/sapiens2 remote add origin https://github.com/facebookresearch/sapiens2.git 2>/dev/null || true
    git -C third_party/sapiens2 fetch -q --depth 1 origin "$SAPIENS2_COMMIT"
    git -C third_party/sapiens2 checkout -q FETCH_HEAD
fi

echo "== 2/3 pretrained weights"
mkdir -p checkpoints/pretrained
fetch() {  # url, output file
    if [ ! -s "$2" ]; then curl -fL --retry 3 -o "$2.part" "$1" && mv "$2.part" "$2"; fi
}
fetch https://huggingface.co/facebook/sapiens2-pretrain-0.4b/resolve/main/sapiens2_0.4b_pretrain.safetensors \
    checkpoints/pretrained/sapiens2_0.4b_pretrain.safetensors
fetch https://huggingface.co/facebook/sapiens-pretrain-0.3b/resolve/main/sapiens_0.3b_epoch_1600_clean.pth \
    checkpoints/pretrained/sapiens_0.3b_epoch_1600_clean.pth

echo "== 3/3 MedSapiens public datasets (Head, Hand, Chest, Leg)"
if [ ! -f data/med_sapien/med_sapien_coco_train_annotations.json ]; then
    mkdir -p data/raw_download
    [ -s data/raw_download/med_sapien.zip ] || gdown "$MEDSAPIENS_DATA_GDRIVE_ID" -O data/raw_download/med_sapien.zip
    unzip -q data/raw_download/med_sapien.zip -d data/
    rm -rf data/__MACOSX data/med_sapien/.DS_Store
fi

echo "done. Next: put pkl20260405-002/ in the repo root, then run scripts/prepare_splits.py (see SETUP.md)."
