# Checkpoints (not bundled)

`manifest.json` records SHA256 and size of the exact local files identified for
the L experiment. It contains no private machine paths.

* `warp_initial_epoch100`: official DSFN Warp checkpoint, distributed locally
  under `epoch001_model_warp.pth`. The training code checks internal epoch100 metadata.
* `warp_L_seed2026_epoch140`: local L experiment trained from that shared checkpoint,
  seed2026, epoch140. It is one checkpoint, not a multi-seed average checkpoint.
* `fusion_used_for_L140`: inherited Fusion checkpoint, not a newly trained L Fusion
  model. Its SHA256 matches the recorded official baseline Fusion file.

Original model download links are in https://github.com/DLUT-YRH/DSFN .
No public download URL for the local L checkpoint has been supplied; do not
invent one. Provide the L file separately to authorized collaborators and add a
real download link only after redistribution permissions have been resolved.

Use checkpoints only from trusted sources. The manifest is a file identity check,
not a guarantee about licensing, accuracy, provenance of training data or safety.
