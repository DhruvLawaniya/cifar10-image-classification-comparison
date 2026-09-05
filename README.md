# CIFAR-10 Image Classification: Comparing PCA+SVM, MLP Ensembles, and CNNs

Comparative study of classical machine learning and deep learning approaches for image
classification on CIFAR-10, built for COMP4318/5318 (University of Sydney).

## Sample Data
![CIFAR-10 sample classes](figs/sample_classes.png)

## Overview
This project evaluates three families of models on 32×32 RGB images across 10 classes
(airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck):

- **Baseline / classical ML:** PCA + Linear SVM, tuned via GridSearchCV over PCA
  variance retained and SVM regularisation strength.
- **Fully connected neural network (MLP):** multiple architectures tuned across
  activation function, layer width/depth, optimiser, batch size, and epochs, tested
  on RGB and HSL colour spaces at full (32×32) and pooled (16×16) resolution, then
  combined into a weighted ensemble.
- **Convolutional neural network (CNN):** a 3-block Conv2D architecture tuned via
  random search over L2 regularisation, dropout, filter count, batch size, and
  learning rate.

Each model is evaluated with validation/test accuracy, confusion matrices,
per-class precision/recall/F1, and (for the CNN) macro-averaged ROC-AUC.


## Data
The dataset is a CIFAR-10 subset provided as `.npy` files, expected at:
```
./Assignment2Data/X_train.npy
./Assignment2Data/y_train.npy
./Assignment2Data/X_test.npy
./Assignment2Data/y_test.npy
```
This data is coursework-provided and is **not included** in this repository.
If you have access to the original CIFAR-10 dataset, you can regenerate equivalent
`.npy` files with `numpy.save`.

## Setup
```bash
python3 -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Requires Python 3.12.

## Usage
1. Place the four `.npy` data files in `Assignment2Data/`.
2. Launch Jupyter and run the notebook top to bottom:
   ```bash
   jupyter notebook notebook.ipynb
   ```
3. Section 4 ("Final models") trains each model independently with its best
   hyperparameters and does not depend on the tuning cells having been run —
   this is the fastest path to reproducing final results.

## Methods summary

| Model | Key techniques |
|---|---|
| PCA + Linear SVM | GridSearchCV over PCA variance (0.80–0.99) and SVM `C` (0.05–10) |
| MLP ensemble | 7 architectures × 2 activations × batch size, RGB/HSL × full/pooled resolution, weighted ensembling |
| CNN | 3× Conv2D + MaxPooling blocks, dense head, random search over L2/dropout/filters/LR/batch size |



*(Fill in your final headline numbers, e.g.)*
- PCA+SVM test accuracy: **XX.X%**
- MLP weighted ensemble test accuracy: **XX.X%**
- CNN test accuracy: **XX.X%**, macro ROC-AUC: **0.XX**


## Acknowledgements
CNN architecture adapted from Ajala, S. (2021), *Convolutional Neural Network
Implementation for Image Classification using CIFAR-10 Dataset*, ResearchGate.
