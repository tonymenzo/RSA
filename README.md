# Rejection Sampling with Autodifferentiation (RSA) [arXiv:2411.02194](https://arxiv.org/abs/2411.02194v1)

Combining Monte Carlo accept-reject sampling, statistical reweighting, and modern differential programming libraries into a single algorithmic framework facilitates the _efficient_ and _differentiable_ exploration of single- or multi-dimensional parameter spaces within an arbitrary probabilistic model. To showcase this framework, termed __*Rejection Sampling with Autodifferentiation*__ (RSA), we consider the Lund string model of hadronization and perform an automated two-parameter fit using hierarchical (pseudo-)datasets generated from hadronizing quark-antiquark strings. We also emphasize the compatibility and efficacy of the tuning framework with binned, unbinned, and _*machine-learning-based*_ observables. For more details on the method and the presented example, see the documentation as well as the accompanying preprint ["Rejection Sampling with Autodifferentiation - Case Study: Fitting a Hadronization Model"](https://arxiv.org/abs/2411.02194v1). 

The repo is partitioned into two directories:

- ### [`data/`](https://github.com/tonymenzo/RSA/tree/main/data)
    This [`data/`](https://github.com/tonymenzo/RSA/tree/main/data) directory includes all of the necessary scripts for altering [`pythia8312`](https://gitlab.com/Pythia8/releases/-/tags/pythia8312)[^1] and generating hadronization datasets compatible with post-hoc reweighting. For more details see [`./data/README.md`](./data/README.md).
    \
    For immediate use with `src/RSA_tuner.py`, `src/lund_weight.py`, etc, a pre-processed dataset containing 10,000 events can be downloaded from [https://zenodo.org/records/14289503](https://zenodo.org/records/14289503).

- ### [`src/`](https://github.com/tonymenzo/RSA/tree/main/src)
    The [`src/`](https://github.com/tonymenzo/RSA/tree/main/src) directory contains the main elements for RSA-based parameter estimation. See [`./src/README.md`](./data/README.md) for more details.

[^1]: [https://pythia.org/](https://pythia.org/)