# TSA: Temporal Slot Activation for Persistent Object-Centric Video Representation

## How to Use

Take TSA on YTVIS-HQ as an example.

**(1) Environment**

Python 3.11.15, PyTorch 2.13.0, torchvision 0.28.0, and `requirements.txt`. Create that environment step by step, or build the Docker image and use it directly.

Local environment:

```shell
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install --index-url https://download.pytorch.org/whl/cu130 \
    torch==2.13.0 torchvision==0.28.0
python -m pip install -r requirements.txt
python -m pip install pillow scikit-image
```

Docker:

```shell
docker compose build
```

**(2) Data and weights**

Download MOVi-C, MOVi-E, and YTVIS-HQ from the [RandSF.Q releases](https://github.com/Genera1Z/RandSF.Q/releases).

| Dataset | Release |
|---|---|
| MOVi-C | https://github.com/Genera1Z/RandSF.Q/releases/tag/dataset-movi_c |
| MOVi-E | https://github.com/Genera1Z/RandSF.Q/releases/tag/dataset-movi_e |
| YTVIS-HQ | https://github.com/Genera1Z/RandSF.Q/releases/tag/dataset-ytvis_hq |
| OVIS | https://github.com/anonymoususer951357/tsa/releases/tag/data-ovis |

Put the extracted LMDB files in the matching folder. Each split is one file:

```shell
reproduce/data/
  movi_c/train.lmdb
  movi_c/val.lmdb
  movi_e/train.lmdb
  movi_e/val.lmdb
  ytvis_hq/train.lmdb
  ytvis_hq/val.lmdb
  ovis/train.lmdb
  ovis/val.lmdb
```

Download the checkpoints and put them into `reproduce/weights`:

| Method | Checkpoint |
|---|---|
| TSA | [weights-tsa.zip](https://github.com/anonymoususer951357/tsa/releases/download/checkpoints/weights-tsa.zip) |
| SmoothSA | [weights-smoothsa.zip](https://github.com/anonymoususer951357/tsa/releases/download/checkpoints/weights-smoothsa.zip) |
| SlotContrast | [weights-slotcontrast.zip](https://github.com/anonymoususer951357/tsa/releases/download/checkpoints/weights-slotcontrast.zip) |
| VideoSAUR | [weights-videosaur.zip](https://github.com/anonymoususer951357/tsa/releases/download/checkpoints/weights-videosaur.zip) |
| RandSF.Q (SSC) | [weights-randsfq_ssc.zip](https://github.com/anonymoususer951357/tsa/releases/download/checkpoints/weights-randsfq_ssc.zip) |
| RandSF.Q (TSIM) | [weights-randsfq_tsim.zip](https://github.com/anonymoususer951357/tsa/releases/download/checkpoints/weights-randsfq_tsim.zip) |

There are three checkpoints per method and dataset, named `weight1.pth`, `weight2.pth`, and `weight3.pth`:

```shell
reproduce/weights/
  tsa/{movi_c,movi_e,ytvis_hq,ovis}/weight{1,2,3}.pth
  smoothsa/{movi_c,movi_e,ytvis_hq,ovis}/weight{1,2,3}.pth
  slotcontrast/{movi_c,movi_e,ytvis_hq,ovis}/weight{1,2,3}.pth
  videosaur/{movi_c,movi_e,ytvis_hq,ovis}/weight{1,2,3}.pth
  randsfq_ssc/{movi_c,movi_e,ytvis_hq,ovis}/weight{1,2,3}.pth
  randsfq_tsim/{movi_c,movi_e,ytvis_hq,ovis}/weight{1,2,3}.pth
```

**(3) Train**

Local:

```shell
CUDA_VISIBLE_DEVICES=0 python train.py \
    --seed 42 \
    --cfg_file config-tsa/tsa-ytvis_hq.py \
    --data_dir reproduce/data \
    --save_dir path/to/save/dir
```

Docker:

```shell
docker compose run --rm tsa \
    python train.py \
    --seed 42 \
    --cfg_file config-tsa/tsa-ytvis_hq.py \
    --data_dir reproduce/data \
    --save_dir save
```

Train every TSA config and seed:

```shell
bash scripts/train_tsa_all.sh
```

```shell
docker compose run --rm tsa bash scripts/train_tsa_all.sh
```

**(4) Evaluate**

Local:

```shell
CUDA_VISIBLE_DEVICES=0 python eval.py \
    --cfg_file config-tsa/tsa-ytvis_hq.py \
    --data_dir reproduce/data \
    --ckpt_file reproduce/weights/tsa/ytvis_hq/weight1.pth
```

Docker:

```shell
docker compose run --rm tsa \
    python eval.py \
    --cfg_file config-tsa/tsa-ytvis_hq.py \
    --data_dir reproduce/data \
    --ckpt_file reproduce/weights/tsa/ytvis_hq/weight1.pth
```

Evaluate every checkpoint. Results are written to `reproduce/main_table_results.txt`.

```shell
bash scripts/run_main_eval.sh
```

```shell
docker compose run --rm tsa bash scripts/run_main_eval.sh
```

TSA configs:

| Config | Task |
|---|---|
| `config-tsa/tsa-movi_c.py` | discovery on MOVi-C |
| `config-tsa/tsa-movi_e.py` | discovery on MOVi-E |
| `config-tsa/tsa-ytvis_hq.py` | discovery on YTVIS-HQ |
| `config-tsa/tsa-ovis.py` | discovery on OVIS |
| `config-tsa/tsa_recogn-ytvis_hq.py` | object recognition downstream |
| `config-tsa/tsa_dynamic-ytvis_hq.py` | dynamics prediction downstream |

## Acknowledgement

This code builds on the public [RandSF.Q](https://github.com/Genera1Z/RandSF.Q) codebase.
