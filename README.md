# TransferKAN: Ballistic Wound Classification via Transfer Learning and Kolmogorov-Arnold Networks

This repository contains the source code, experimental notebooks, and academic manuscripts for a Capstone Project (TCC) focused on gunshot and entry/exit wound classification using Deep Learning architectures (ResNet152 and KAN).

---

## Project Structure

The project is structured into the following directories:

* **`notebooks/`** Contains the machine learning code. The current experiments run under a single, leak-free protocol: `protocolo_comum.py` (training and evaluation module), `configs_mlsd.py` (shooting-distance task and the reproduction of the 2024 hold-out split) and notebooks `00`–`05b` (data and grouped folds, L4 calibration, head ablation, baselines, MLSD, leakage test and analyses). The eight original notebooks (ConvKAN, PureKAN, TransferKAN and ResNet152 hybrids) and their extracted logs are kept unchanged as a historical record.

* **`resultados/`** Shared grouped folds and the analysis outputs of the single protocol (tables in CSV/LaTeX, figures, decisions and the transcription of Tables 4 and 5 of Lira et al.). Images, caches and checkpoints are never stored here.

* **`docs/planos/`** Experiment plans: the pre-registered protocol (P1), multitask (P2), Wav-KAN heads (P3) and the dissertation rewrite (P4).

* **`references/`** Stores relevant academic literature, reference papers, and text documentation used to support the research.

* **`sibgrapi2026/`** Contains the LaTeX source files, bibliography, and Makefile for the short paper manuscript targeting the SIBGRAPI 2026 conference (formatted using the IEEEtran class).

* **`tcc_dissertation/`** Contains the complete graduation thesis (monograph) codebase formatted to meet the standards of the Department of Computer Science at the University of Brasília (UnB). This includes individual chapter files, image assets, and institutional style templates.

---

## Getting Started

### Machine Learning Environment
The protocol notebooks are meant for Google Colab with an NVIDIA L4 GPU and install their own pinned dependencies (PyTorch, torchvision, Albumentations, `efficient-kan`). The dataset (FDCPUnBGunshotDB, Lira et al., 2025, IJLM) is cloned at a fixed commit inside each session and is not redistributed in this repository.


### LaTeX Compilation

To compile the academic documents inside `sibgrapi2026/` or `tcc_dissertation/`, ensure you have a standard TeX distribution (such as TeX Live or MiKTeX) along with make utilities installed. Simply navigate into the respective folder and run:

```bash
make
```


---

## License

This project is licensed under the terms specified in the LICENSE.md file found in the root directory.
