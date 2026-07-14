# BBB Permeability Prediction using 3D Graph Neural Networks and Ensemble Learning

Prediction of Blood–Brain Barrier (BBB) permeability (logBB) using a hybrid machine learning framework that combines **3D Graph Neural Networks (SchNet)** and **ExtraTrees** regression.

This repository contains the source code developed for my Master's thesis in Biomedical Engineering at the **University of Palermo**, focused on molecular property prediction for drug discovery.

---

![Python](https://img.shields.io/badge/Python-3.12-blue)

![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)

![PyG](https://img.shields.io/badge/PyTorch_Geometric-2.x-orange)

![License](https://img.shields.io/badge/License-Academic-lightgrey)

---

## Methodological Pipeline

<img width="440" height="339" alt="image" src="https://github.com/user-attachments/assets/0dd4da89-3b21-4d62-80ab-5c5519a36212" />


The proposed workflow combines complementary molecular representations to improve continuous logBB prediction.

The pipeline consists of:

- Molecular preprocessing and descriptor extraction with **RDKit**
- Training of a **SchNet** model on 3D molecular graphs
- Training of an **ExtraTrees** model on selected molecular descriptors
- Multi-seed and cross-validation ensemble generation
- Weighted hybrid ensemble using Out-of-Fold predictions
- Final evaluation on an independent hold-out test set

---

# Overview

The Blood–Brain Barrier (BBB) represents one of the major challenges in central nervous system drug development.

Experimental determination of BBB permeability is expensive, time-consuming and often unavailable during the early stages of drug discovery. Machine learning models can therefore provide valuable support by estimating BBB permeability directly from molecular structure.

This project investigates the prediction of **continuous logBB values** through the integration of graph neural networks and descriptor-based machine learning models.

---

# Key Features

- 3D molecular graph representation
- SchNet architecture with engineered node features
- ExtraTrees regression with feature selection
- Multi-seed ensemble learning
- Out-of-Fold weighted hybrid ensemble
- Nested cross-validation
- Data leakage prevention
- Fully reproducible training pipeline

---  

# Dataset

The project combines publicly available BBB permeability datasets into a single regression dataset containing:

- **1303 molecules**
- Experimental **logBB** values
- 3D molecular conformations
- RDKit molecular descriptors

The processed dataset is **not included** in this repository due to licensing and academic restrictions.

---

# Methodology

## SchNet

The graph neural network was developed using **PyTorch Geometric**.

The model exploits:

- 3D atomic coordinates
- Atomic number embeddings
- 15 engineered node features
- 9 selected global molecular descriptors
- Batch Normalization
- PReLU activation
- Dropout regularization
- Early stopping
- 5-fold cross-validation
- Multi-seed training (3 random seeds)

---

## ExtraTrees

The descriptor-based model was developed using **Scikit-learn**.

Main steps include:

- RDKit descriptor extraction
- Sequential Forward Feature Selection (SFS)
- Standardization
- Hyperparameter optimization using GridSearchCV
- Ensemble averaging across folds

---

## Hybrid Ensemble

The final prediction combines SchNet and ExtraTrees outputs through a weighted linear ensemble:

<p align="center">

**ŷ = α · ŷ<sub>SchNet</sub> + (1 − α) · ŷ<sub>ExtraTrees</sub>**

</p>

The ensemble coefficient (α) is selected **only using Out-of-Fold predictions**, preventing information leakage from the hold-out test set.

Nested cross-validation is then performed to evaluate the robustness of the hybrid strategy.

---

# Results

The models were evaluated on the same independent hold-out test set (196 molecules), which was never used during feature selection, hyperparameter optimization, or model training.

Since the curated molecular dataset used in this thesis cannot be publicly distributed, the results reported below are provided for methodological transparency only and are not intended as directly reproducible benchmarks from this repository alone.

| Model | Test R² |
|:-----------------------------|---------:|
| ExtraTrees 5-fold Ensemble | 0.4383 |
| SchNet Multi-Seed Ensemble | 0.5146 |
| **Weighted Hybrid Ensemble** | **0.5161** |

The final model combines the predictions of the SchNet multi-seed ensemble and the ExtraTrees 5-fold ensemble using a weighted average, where the ensemble weight is selected exclusively from out-of-fold predictions to avoid information leakage.


<p align="center">
  <strong>
    ŷ<sub>hybrid</sub> =
    α · ŷ<sub>SchNet</sub> +
    (1 − α) · ŷ<sub>ExtraTrees</sub>
  </strong>
</p>

The ensemble coefficient was selected exclusively from out-of-fold predictions.
The optimal value was:

<p align="center">
  <strong>α = 0.65</strong>
</p>

Therefore, the final prediction assigned a weight of 0.65 to SchNet and 0.35
to ExtraTrees.

A nested five-fold procedure was also applied specifically to the selection
of the ensemble weight, obtaining:

<p align="center">
  <strong>Cross-validation R² = 0.562 ± 0.057</strong>
</p>

The weighted hybrid ensemble achieved the best overall hold-out performance,
with a test R² of 0.5161, RMSE of 0.6974, and MAE of 0.4614. The improvement
over the SchNet ensemble was limited but consistent, reflecting the relatively
high correlation between the out-of-fold residuals of SchNet and ExtraTrees
(r = 0.782).

---

### Experimental versus Predicted logBB

<img width="465" height="166" alt="image" src="https://github.com/user-attachments/assets/d94a9788-72c6-4996-bbc2-b50ad82eeccf" />


Comparison between experimental and predicted logBB values for the SchNet
ensemble, ExtraTrees ensemble, and weighted hybrid model on the hold-out test set.

### Ensemble Weight Selection

<img width="361" height="246" alt="image" src="https://github.com/user-attachments/assets/5505e2d7-f1a7-46c5-a18b-e4deccfff016" />


The ensemble weight was selected using only out-of-fold predictions.
The optimal value was α = 0.65.

### Nested Cross-Validation Stability

<img width="353" height="241" alt="image" src="https://github.com/user-attachments/assets/c3d80f54-4ca8-405a-b19e-f38c378adcc7" />


The nested weight-selection procedure produced a mean R² of 0.562 with a
standard deviation of 0.057 across the five folds.

### Hybrid Model Residual Analysis

<img width="440" height="180" alt="image" src="https://github.com/user-attachments/assets/728ad285-8710-48b7-8817-1380fc41c4ef" />


The residuals were approximately centered around zero and did not show an
evident systematic relationship with the predicted logBB values.

---

# Repository Structure

```text
.
├── data/
│   └── README.md
│
├── docs/
│   ├── pipeline.png
│   ├── scatter_hybrid.png
│   ├── alpha_curve.png
│   └── training_curve.png
│
├── models/
│
├── results/
│
├── src/
│   ├── feature_selection/
│   │   └── extratrees_feature_selection.py
│   │
│   ├── models/
│   │   ├── schnet.py
│   │   ├── schnet_multiseed_ensemble.py
│   │   ├── extratrees_ensemble.py
│   │   ├── hybrid_extratrees_schnet.py
│   │   └── schnet_common.py
│   │
│   └── config.py
│
├── README.md
├── requirements.txt
└── RUNNING.md
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/marcgambi/bbb-permeability-prediction.git

cd bbb-permeability-prediction
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

# Running the Pipeline

The scripts are intended to be executed sequentially.
Each stage depends on the outputs generated by the previous one.
Execute the scripts in the following order:

```bash
python -m src.feature_selection.extratrees_feature_selection

python -m src.models.extratrees_ensemble

python -m src.models.schnet

python -m src.models.schnet_multiseed_ensemble

python -m src.models.hybrid_extratrees_schnet
```

---

# Technologies

- Python
- PyTorch
- PyTorch Geometric
- RDKit
- Scikit-learn
- NumPy
- Pandas
- Matplotlib

---

# Future Improvements

Possible future developments include:

- evaluation on larger molecular datasets
- comparison with additional Graph Neural Network architectures
- uncertainty estimation
- explainable AI methods
- web deployment for virtual screening applications
- transformer-based molecular foundation models

---

# Author

**Marco Gambino**

M.Sc. Biomedical Engineering  
University of Palermo

- LinkedIn: https://www.linkedin.com/in/marco-gambino-7a3071138/
- GitHub: https://github.com/marcgambi

---

## Disclaimer

This repository is intended for research and educational purposes only.

The dataset used during the thesis is not redistributed.
