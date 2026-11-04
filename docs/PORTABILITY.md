# Packaging changes

Source: frozen local DSFN-L method (variant L). Original experiment files are not
edited. Core loss, gradient coordinator, data handling and network calculations
are preserved. Changes in the release copy:

1. Remove the global TLS-verification bypass before torchvision initialization.
2. In supported Warp/Fusion inference scripts, require explicit checkpoint and
   output directory. Strict-load weights, fail on missing checkpoint and require
   batch size one. Keep original JPEG writer, color conventions and geometry.
3. Remove an obsolete private Linux-path comment. Exclude ad-hoc legacy test
   entry points, IDE/cache directories, server launchers, logs, data and weights.
4. Add a root launcher using interpreter-relative subprocesses and explicit paths,
   matching-filename checks, GPU selection and overwrite protection.
5. Record actual installed dependencies instead of copying the old Linux-wide
   Conda export. Optional evaluation is the existing alignment script, not a new
   metric definition.

Original inference auto-picked the last file in a model directory; this release
does not. Training still accepts compatible L resumes or the shared epoch100
checkpoint; it does not silently treat arbitrary files as valid starting weights.
