# COBE: Public JointMAMBA Architecture Release

This repository is the public architecture release for the JointMAMBA research
track in one-dimensional crystal-truncation-rod (CTR) phase retrieval.

It contains the reusable neural relation modules and their mathematical
regression tests.  In particular, the relation block combines local
convolution, relative-position bias, content-dependent nonlocal attention,
low-order Chebyshev cross terms, and gated residual fusion.

> **Terminology.** “MAMBA” here means a two-branch 1D U-Net CTR
> phase-retrieval architecture; it is not a state-space Mamba model.

## Included

```text
stage1_architectures.py      Baseline, dual-branch, and relation U-Nets
stage1_relation_layers.py    Relative-position and polynomial relation layers
tests/                       Algebra, invariance, gradient, and shape tests
docs/                        Scope and reproducibility boundaries
```

## Not included

The experimental reciprocal-to-real-space reconstruction engine, material
data interfaces, calibration logic, data-generation pipeline, and trained
artifacts are proprietary internal research components.  This public release
does not contain those implementations, experimental data, checkpoints, or
machine-specific configuration.

## Installation and test

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
python -m pytest tests
```

## Scientific boundary

The included tests establish implementation-level properties of the neural
relation layer.  They do not demonstrate physical reconstruction accuracy,
cross-material generalization, or performance on experimental samples.

See [docs/RESEARCH_SCOPE.md](docs/RESEARCH_SCOPE.md) for the claim boundary.
