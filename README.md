# Provable Imbalanced Point Clustering

Official implementation of **Provable Imbalanced Point Clustering**, published at CSCML 2024. The paper presents provable approximation methods for clustering imbalanced point sets using coresets, together with experiments on images, synthetic data, and real-world datasets.

- [Conference paper — CSCML 2024](https://doi.org/10.1007/978-3-031-76934-4_5)
- [Full manuscript with appendix](https://arxiv.org/abs/2408.14225)
- [Journal article — Cryptography and Communications, 2026](https://doi.org/10.1007/s12095-026-00910-1)

## Repository contents

The repository contains algorithm implementations, experiment scripts, images, plots, and saved results. Section references below follow the full conference manuscript, including its appendix.

| File or directory | Description |
| --- | --- |
| `algorithms.py` | Implementations of the clustering algorithms used in the experiments. |
| `motivation.py` | Introductory examples from Section 1. Experiment settings are configured in the script. |
| `image_test.py` | Drawing-image experiment from Section 3.1. |
| `new_motivation.py` | Scikit-learn-inspired image experiment from Section 3.2. |
| `choice_toy.py` | Cat-and-boat experiment from Section 3.4. |
| `synthetic_test.py` | Synthetic-data experiments from Appendix E.1. |
| `real_test.py` | Real-world-data experiments from Appendix E.1. |
| `divisive.py` | Experiments from Appendix E.2. |
| `choice/` | Images used in Section 3.4 and the cropping script. |
| `imgs/` | Original images used in the paper. |
| `motivation/` | Images used in Sections 1, 3.1, and 3.3. |
| `plots/` | Plots from Appendix E. |
| `res/` | Saved results from Appendix E. |
| `requirements.txt` | Python dependencies. |

## Running the experiments

Install the dependencies listed in `requirements.txt` before running the experiment scripts. For `synthetic_test.py`, check that your CUDA setup is compatible with the installed dependencies.

Experiment settings are configured directly in the scripts. Review the relevant script before changing a dataset or experiment configuration.

## Contact

For questions, suggestions, or collaboration, contact [David Denisov](mailto:DavidDenisov14@gmail.com).

## License

This work is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License](https://creativecommons.org/licenses/by-nc/4.0/).

See [LICENSE.md](LICENSE.md) and the [plain-text license](License) for details.
