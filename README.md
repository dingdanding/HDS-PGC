# HDS-PGC: depth-supervision regulation for DSFN

DSFN-based research implementation of pixel reliability weighting, sample-level
information gating, and asymmetric grouped gradient coordination (internal variant L).
The geometric network retains the DSFN CCL / homography / TPS pipeline. These
additions act during Warp training; inference uses RGB pairs and the existing Fusion network.

## Status and attribution

This folder is a portable source package, not a dataset or trained-weight bundle.
It is not the official DSFN repository. The upstream implementation is
[DLUT-YRH/DSFN](https://github.com/DLUT-YRH/DSFN).
The DSFN repository contact has confirmed by email that we may modify and
publicly release this implementation with appropriate citation. We retain
attribution to the original project and its authors. This permission does not
establish an MIT or Apache license for the inherited code. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for source attribution and
third-party terms.

## Installation

Commands below are run from this repository root in Anaconda Prompt.

```bat
conda create -n hds_pgc python=3.10 pip
conda activate hds_pgc
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

The versions above describe the observed local Windows environment, not a claim
that every historical server run used these versions. Select a matching PyTorch /
torchvision CUDA build appropriate for the target machine if necessary; see the
[official installation selector](https://pytorch.org/get-started/locally/).
CUDA GPU execution is the tested workflow; full CPU training is not validated.

ResNet50 initialization uses torchvision's original ImageNet V1 weights
(`pretrained=True`, retained for behavior compatibility). First use may download
the weights. Offline users can set `TORCH_HOME` to their own populated cache.

## Data and weights

Obtain UDIS-D / IVSD and original checkpoints from the upstream README.
No dataset, pretrained model, private path, training archive or generated result
is included here. Training expects paired, matching `*.jpg` filenames:

```text
training/
  input1/000001.jpg
  input2/000001.jpg
  depthInput1/000001.jpg
  depthInput2/000001.jpg
testing/
  input1/000001.jpg
  input2/000001.jpg
```

Training uses offline relative-depth maps generated with Depth Anything; these
must correspond to the RGB images. No depth prediction model is bundled.
See [weights/README.md](weights/README.md) for checkpoint provenance and hashes.

## Train the L method

Replace the example relative paths with your own paths. Start from the shared
epoch100 Warp checkpoint, or resume a compatible L checkpoint. The original
training script validates its epoch and method metadata; the filename alone is
not evidence of the checkpoint epoch.

```bat
python run.py train --data data/training --resume weights/epoch001_model_warp.pth --output outputs/L_seed2026 --max-epoch 140 --seed 2026 --batch-size 4 --workers 4 --gpu 0
```

This preserves the L training defaults, including depth loss coefficient 0.3,
gradient projection, pixel reliability and sample information gate. The gate
thresholds are the frozen training-calibrated constants in `Warp/newCodes/train.py`.
Checkpoints are saved every five epochs. Use an explicit L checkpoint and a new
output directory to resume. The launcher refuses existing output directories.

## Warp then Fusion

```bat
python run.py stitch --data data/testing --warp-checkpoint weights/epoch140_model_.pth --fusion-checkpoint weights/epoch002_model_fusion.pth --output outputs/L140_test --gpu 0
```

Final images: `outputs/L140_test/fusion/composition`. Warp results and masks:
`outputs/L140_test/warp`. Inputs are never overwritten. Saved names follow the
original sequential six-digit numbering, in sorted JPEG input order; use the
standard dataset filenames. Variable-size inference uses batch size one.
Only load trusted checkpoints: loading training metadata uses Python pickle.

## Alignment evaluation

```bat
python -m pip install -r requirements-eval.txt
python evaluation/evaluate_alignment.py --warp-root outputs/L140_test/warp --output-dir outputs/metrics --name L140
```

This is the existing project PSNR/SSIM/LPIPS protocol on valid Warp overlap,
not an invented reference image and not a score of the Fusion canvas.
`--no-lpips` runs PSNR/SSIM only. LPIPS requires its pretrained AlexNet resources.
SIQE is not bundled: use its author-distributed MATLAB implementation and state
the evaluation settings separately. No paper scores are embedded as computed outputs.

## Checks and provenance

`test_l_information.py` contains the L gate/loss tests; `preflight_l.py` performs
a real-data optimizer-step check when explicitly supplied with data.
`docs/environment_observed.json` records local versions. Source file hashes and
portability changes are documented in `docs/PORTABILITY.md` and the source manifest.
No full retraining is performed merely to package the code.

## Cite the baseline

Jiang Z, Yan R, Zhang Z, Zhang B, Liu J. Depth-Supervised Fusion Network for
Seamless-Free Image Stitching. NeurIPS, 2025.
https://arxiv.org/abs/2510.21396

The citation for the present method should be added after its publication details
are finalized; no DOI or acceptance status is asserted here.
