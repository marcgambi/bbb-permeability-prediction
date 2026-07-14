# Blood-Brain Barrier Permeability Prediction

Machine learning and deep learning models for predicting blood-brain barrier permeability (logBB) from molecular data.

This repository contains the best-performing models developed during my MSc thesis in Biomedical Engineering at the University of Palermo.

---

## Project Overview

Predicting whether a molecule can cross the blood-brain barrier (BBB) is a key challenge in central nervous system (CNS) drug discovery.

The objective of this project is to estimate the logarithmic blood-brain barrier partition coefficient (logBB) using both traditional machine learning and graph neural network approaches.

The implemented pipeline combines:

- Molecular graph representations
- Physicochemical descriptors
- Feature selection techniques
- Ensemble learning
- Deep learning on molecular graphs

---

## Repository Structure

```
src/
│
├── feature_selection/
│   └── extratrees_feature_selection.py
│
├── models/
│   ├── extratrees_ensemble.py
│   ├── schnet.py
│   ├── schnet_multiseed_ensemble.py
│   └── hybrid_extratrees_schnet.py
```

---

## Implemented Models

### Extra Trees with Feature Selection

Feature importance estimation using Extra Trees Regressor for descriptor selection before model training.

---

### Extra Trees Ensemble

Ensemble regression model trained on optimized molecular descriptors.

---

### SchNet

Graph neural network based on continuous-filter convolutions for learning molecular representations directly from atomic information.

---

### Multi-Seed SchNet Ensemble

Multiple SchNet models trained using different random seeds to improve robustness and reduce prediction variance.

---

### Hybrid Extra Trees + SchNet

Hybrid architecture combining:

- SchNet latent molecular embeddings
- Selected RDKit molecular descriptors
- Extra Trees regression

This model achieved the best predictive performance among all tested approaches.

---

## Dataset

The project combines publicly available blood-brain barrier permeability datasets to create a larger regression dataset for logBB prediction.

The final dataset contains over 1300 molecules after preprocessing and descriptor extraction.

*The original datasets are not redistributed in this repository.*

---

## Technologies

- Python
- PyTorch
- PyTorch Geometric
- RDKit
- Scikit-learn
- NumPy
- Pandas
- Matplotlib

---

## Evaluation Metrics

Model performance was evaluated using:

- R² Score
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)

---

## Main Research Topics

- Blood-Brain Barrier permeability prediction
- Molecular machine learning
- Graph Neural Networks
- SchNet
- Ensemble Learning
- Feature Selection
- Drug Discovery
- Cheminformatics

---

## Author

**Marco Gambino**

MSc Biomedical Engineering  
University of Palermo

LinkedIn:

https://www.linkedin.com/in/marco-gambino

---

## License

This project is released under the MIT License.
