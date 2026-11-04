# Third-party sources and authorization status

DSFN upstream: https://github.com/DLUT-YRH/DSFN
Paper: Depth-Supervised Fusion Network for Seamless-Free Image Stitching,
Zhiying Jiang, Ruhao Yan, Zengxi Zhang, Bowei Zhang, Jinyuan Liu, NeurIPS 2025.

Warp geometry (ResNet features, CCL, homography, TPS), Fusion, associated losses,
and dataset handling derive from DSFN and its included components. Preserve all
existing author and copyright notices. Local development adds reliable-depth
supervision, sample information gates and grouped gradient coordination; it does
not claim authorship of the DSFN backbone or the general gradient-projection idea.
The latter is related to Yu et al., Gradient Surgery for Multi-Task Learning,
NeurIPS 2020. A paper citation is not a software redistribution license.

The inspected local original snapshot and upstream root listing did not supply
a top-level LICENSE. This package therefore does NOT assign MIT, Apache or any
other license to inherited code. Authorization and any nested third-party terms
must be resolved with the rights holders before public distribution. This notice
is a provenance record, not legal permission or a public-release clearance.

Original code ZIP recorded SHA256:
060932915605bdea441dd25987fb4f1f1f8f861737b3d0c02753856c575b5084
This identifies the recorded downloaded archive, not a verified upstream commit.

PyTorch, torchvision, NumPy, OpenCV, Pillow, SciPy, scikit-image, imageio,
TensorBoard and optional LPIPS retain their own licenses and pretrained-weight
terms; install from their distributions. No third-party package or pretrained
weight binaries are vendored here. Dataset redistribution rights are separate.
