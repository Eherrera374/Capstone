# Mental Health Outcome Prediction

A machine learning project that builds and evaluates five classification models to predict patient mental health treatment outcomes using clinical and behavioral features.

---

## Project Overview

This project explores whether patient clinical data can predict which treatment outcomes are most likely — specifically identifying patients whose condition is likely to **Improve**. Two modelling strategies are compared: a full **multiclass approach** (3 outcome classes) and a **binary classification approach** (Improved vs. Not Improved) with class balancing and threshold tuning.

> ⚠️ **Important:** Models in their current state should not be used for clinical treatment decisions. Performance metrics indicate weak predictive signal, likely due to the synthetic nature of the dataset. See [Limitations](#limitations) and [Recommendations](#recommendations).

---

## Dataset

| Property | Value |
|---|---|
| Samples | 500 patients |
| Features | 16 clinical and behavioral variables |
| Target | Treatment outcome: Improved / No Change / Deteriorated |
| Missing values | None |
| Source | Synthetic dataset |

**Feature types include:**
- Demographic: Age, Gender
- Clinical: Diagnosis, Medication, Therapy Type
- Behavioral: Physical Activity, Adherence to Treatment %, Mood Score
- AI-derived: AI-Detected Emotional State, Treatment Progress Score

---

## Models

Five classification models are trained and evaluated:

| Model | Notes |
|---|---|
| K-Nearest Neighbors (KNN) | K selected via cross-validation |
| Decision Tree | `max_depth=5` to control overfitting |
| Random Forest | 200 estimators, hyperparameter tuned |
| Logistic Regression (L1) | L1 regularization for feature selection, C tuned |
| Support Vector Machine (SVM) | RBF kernel, C and kernel tuned |

All models are trained with `class_weight="balanced"` to handle class imbalance, and decision thresholds are tuned beyond the default 0.5 cutoff.

---

## Notebook Structure

```
1. Imports
2. Load & Preview Data
3. Preprocessing
   - Drop non-predictive columns (Patient ID, Treatment Start Date)
   - One-hot encode categorical features
   - Binary and multiclass target encoding
   - Stratified train/test split (80/20)
   - StandardScaler for distance/margin-based models
4. Metric Justification
5. Multiclass Classification (3 classes)
6. Binary Classification with Class Weights & Threshold Tuning
7. Hyperparameter Tuning (GridSearchCV)
8. Feature Importance & Interpretation
9. Final Summary & Recommendations
```

---

## Evaluation Metrics

Given the healthcare context, metrics are prioritized as follows:

| Metric | Priority | Reason |
|---|---|---|
| **Recall** | Highest | Missing a patient who would improve has real clinical consequences |
| **F1 Score** | High | Balances precision and recall when both error types matter |
| **ROC-AUC** | High | Threshold-independent comparison across models |
| **Precision** | Medium | Avoid flagging patients for treatments unlikely to help |
| **Accuracy** | Lowest | Misleading with imbalanced classes (~66% baseline by predicting majority) |

---

## Key Findings

- All models showed **weak predictive performance** (F1 < 0.40, ROC-AUC ≈ 0.50), suggesting the dataset does not contain strong clinical signals in its current form
- Applying `class_weight="balanced"` and threshold tuning significantly improved **recall** for the positive (Improved) class — Random Forest and SVM went from 0 recall to meaningful detection
- The **multiclass problem** (3 classes) proved harder than binary but preserved more outcome information
- **Feature importance was inconsistent** across models; some findings (e.g. negative adherence coefficient in Logistic Regression) are clinically counterintuitive and likely reflect data artifacts rather than true relationships
- Top predictive features identified by Random Forest: Age, Adherence to Treatment %, Physical Activity, Treatment Duration, Mood Score

---

## Requirements

```
pandas
numpy
matplotlib
scikit-learn
```

Install with:
```bash
pip install pandas numpy matplotlib scikit-learn
```

---

## Usage

1. Clone or download the repository
2. Place `mental_health.csv` in the same directory as the notebook
3. Open `mental_health_prediction_v2.ipynb` in Jupyter
4. Run all cells sequentially (`Kernel > Restart & Run All`)

---

## Limitations

- **Synthetic data** — the dataset is artificially generated; real clinical data would likely yield stronger and more interpretable signals
- **Small sample size** — 500 patients with many features after one-hot encoding creates a high-dimensional, low-sample problem
- **Suspicious feature relationships** — some coefficients contradict clinical intuition, suggesting data noise rather than genuine patterns
- **No external validation** — results have not been validated on an independent dataset or reviewed by clinical experts
- **No temporal features** — treatment trajectory over time is not modeled

---

## Recommendations

Based on the analysis, the following next steps are suggested:

1. **Collect real-world clinical data** with validated outcome measures and larger sample sizes
2. **Engage clinical domain experts** to validate all feature importance findings before drawing conclusions
3. **Deploy only as a screening tool** — flag patients for clinician review rather than making autonomous treatment recommendations
4. **Explore multiclass therapy recommendations** once data quality improves
5. **Test deep learning approaches** on larger datasets where neural networks can better leverage high-dimensional clinical features
6. **Address data quality** — investigate and resolve counterintuitive feature relationships before building on this model

---

## Project Structure

```
├── mental_health_prediction_v2.ipynb   # Main analysis notebook
├── mental_health.csv                   # Dataset (required)
└── README.md                           # This file
```

---

## Disclaimer

This project is for educational purposes only. The models and findings presented here should not be used to make real clinical or treatment decisions without rigorous validation by qualified healthcare professionals.
