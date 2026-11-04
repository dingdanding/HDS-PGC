# Packaging validation (2026-11-04)

Observed on Windows with the environment recorded in `environment_observed.json`:

* `pip check`: no broken requirements found.
* Root launcher, Warp training and alignment evaluation help/import checks: passed.
* L information-gate regression and random-tensor forward/backward checks: passed.
* Real pair 000031: full Warp and Fusion inference completed using the frozen
  seed2026 epoch140 Warp and inherited Fusion files; both strict-loads succeeded.
* The above real image pair was used only as a packaging smoke test; no new
  paper evaluation or claimed improvement was calculated.
* No full retraining, full-dataset evaluation, clean-machine installation,
  CPU-only full inference, or optional LPIPS installation test is claimed.
* torchvision's deprecated `pretrained=True` warning is expected; its original
  ImageNet V1 initialization behavior was deliberately retained.

Run `python check_package.py` to verify the delivered file hashes and syntax.
This is technical package validation, not an authorization for public redistribution.
