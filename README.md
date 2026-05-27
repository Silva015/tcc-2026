# TransferKAN: Ballistic Wound Classification via Transfer Learning and Kolmogorov-Arnold Networks

This repository contains the source code, experimental notebooks, and academic manuscripts for a Capstone Project (TCC) focused on gunshot and entry/exit wound classification using Deep Learning architectures (ResNet152 and KAN).

---

## Project Structure

The project is structured into four main directories:

* **`notebooks/`** Contains all Jupyter Notebooks with the machine learning experiments, model training code (ConvKAN, PureKAN, TransferKAN, and ResNet152 hybrids), and raw data extraction text logs.

* **`references/`** Stores relevant academic literature, reference papers, and text documentation used to support the research.

* **`sibgrapi2026/`** Contains the LaTeX source files, bibliography, and Makefile for the short paper manuscript targeting the SIBGRAPI 2026 conference (formatted using the IEEEtran class).

* **`tcc_dissertation/`** Contains the complete graduation thesis (monograph) codebase formatted to meet the standards of the Department of Computer Science at the University of Brasília (UnB). This includes individual chapter files, image assets, and institutional style templates.

---

## Getting Started

### Machine Learning Environment
To run the project notebooks, ensure you have a Python environment with PyTorch (GPU support recommended) and the KAN library installed:
```bash
pip install torch torchvision jupyterlab pykan
```


### LaTeX Compilation

To compile the academic documents inside `sibgrapi2026/` or `tcc_dissertation/`, ensure you have a standard TeX distribution (such as TeX Live or MiKTeX) along with make utilities installed. Simply navigate into the respective folder and run:

```bash
make
```


---

## License

This project is licensed under the terms specified in the LICENSE.md file found in the root directory.
