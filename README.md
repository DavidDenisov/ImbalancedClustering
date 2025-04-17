# Smart-init of neural networks
The official code of the paper **Smart-init of neural networks** published (in press) in ICCAI 2025,
which suggested an initialization method that has improved the results of the experiments done in the paper.

*Fell free to contact [David-Denisov](mailto:DavidDenisov14@gmail.com) for suggestions, collaborations, e.t.c.*

This repository contains the code and results for all the algorithms and examples that appeared in the paper.

### CONSTRUCTION
- **paper.pdf** the current version of the paper.
- **algorithms.py** All the algorithms used.
- **motivation.py** Run all the tests in **Section 1**, to change between the tests change the hard-coded values.
- **image_test.py** Run the test for the drawing image at **Section 3.1**.
- **motivation_new.py** Run the test for the sklearn inspired test image at **Section 3.2**.
- **choice_toy.py** Run the cat and boat test from **Section 3.4**.
- **synthetic_test.py** Run the synthetic test from **Section E.1**.
- **real_test.py** Run the real-world test from **Section E.1**.
- **divisive.py** Run all the tests from **Section E.2**.
- **choice/** has all the images for **Section 3.4**, and the script to crop them.
- **imgs/** contains all the original images used throughout the paper.
- **motivation/** contains the *motivation* images used in **Section 1, 3.1, and 3.3**
- **plots/** all the images used in **Section E**.
- **res/** contains all the results for the tests in **Section E**.

### For initialization, install the requirements; check the CUDA version for synthetic_test.

## License
Shield: [![CC BY-NC 4.0][cc-by-nc-shield]][cc-by-nc]

This work is licensed under a
[Creative Commons Attribution-NonCommercial 4.0 International License][cc-by-nc].

[![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]

[cc-by-nc]: https://creativecommons.org/licenses/by-nc/4.0/
[cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png
[cc-by-nc-shield]: https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg

See [License](License.md) and an unformatted version at [Un-formated-License](License).
