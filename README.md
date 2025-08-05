# Improving Adversarial Transferability via Semantic-Style Joint Expectation Perturbations

This repository contains the official code for our paper (**Improving Adversarial Transferability via Semantic-Style Joint Expectation Perturbations**). The project is based on and extends the [TransferAttack](https://github.com/Trustworthy-AI-Group/TransferAttack) framework. The main contributions and experimental code are located in the `transferattack/myexp/` directory.

## Overview

This project proposes and implements new transfer-based adversarial attack methods for image classification models. All novel algorithms and experiments are implemented in the `transferattack/myexp/` directory (e.g., `mi_gram.py`, `pgn_gram.py`).

## Key Files

- `transferattack/myexp/mi_gram.py`: Implementation of the $SSEP_{MI}$ attack method.
- `transferattack/myexp/pgn_gram.py`: Implementation of the $SSEP_{PGN}$ attack method.
- Additional attacks, transformations, and model-related code can be found in other submodules under `transferattack/`.

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Getting Started

You can use `main.py` as the entry point, or refer to the scripts and instructions in the `transferattack/myexp/` directory to run your experiments.

## Citation

If you use this code for your research, please cite our paper (citation information will be updated after publication).

## Acknowledgements

This project is developed based on [TransferAttack](https://github.com/Trustworthy-AI-Group/TransferAttack). We thank the original authors for their contributions.
