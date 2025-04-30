# 🔋 Battery End-of-Life Prediction with Uncertainty Estimation

This repository contains the implementation of several machine learning methods for predicting the Remaining Useful Life (RuL) of lithium-ion batteries, along with estimation of prediction uncertainty. This project was developed as part of a semester thesis at the Technical University of Munich (TUM).

## 📌 Objective

The goal is to compare different ML-based methods for battery lifetime prediction **with uncertainty estimation**, addressing both **aleatoric** and **epistemic** uncertainty. The MIT open-source battery cycling dataset is used.

Implemented models include:

- Gaussian Process Regression (GPR)
- Random Forest Regression (RFR)
- Monte Carlo Dropout (MCD)
- Bayes by Backprop (BBP)

## 📊 Evaluation Metrics

The following metrics are used to evaluate model performance:

- **MAPE** (Mean Absolute Percentage Error)
- **RMSCE** (Root Mean Square Calibration Error)
- **CRPS** (Continuous Ranked Probability Score)
- **Sharpness**

## 🔍 Key Features

- Uses feature-based approach inspired by Severson and Fei et al.
- Focuses on calibration and robustness of uncertainty.
- Sensitivity analysis includes:
  - Impact of feature distance
  - Impact of missing training data (input space and target space)

## ▶️ Getting Started

1. Clone the repository:

```bash
git clone https://github.com/your_username/battery-eol-uncertainty.git
cd battery-eol-uncertainty
```
2. Install dependencies:

```bash
pip install -r requirements.txt
```
3. Run model training:

```bash
python src/train_gpr.py
```
## 📖 Citation
This repository is part of the following academic thesis:

Dexuan Meng, Comparison of ML-based Methods for the Uncertainty Estimation of the End-of-life Prediction for Li-Ion Batteries, TUM, 2022.

📄 License
MIT License © 2022 Dexuan Meng


