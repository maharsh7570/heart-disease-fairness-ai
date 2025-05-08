#  Heart Disease Prediction with Fairness & Explainability

This project demonstrates a complete machine learning pipeline for predicting heart disease using a synthetic version of the Framingham dataset. It integrates advanced fairness, causal inference, and model explainability tools.

---

##  Project Highlights

- **Model**: Tuned XGBoost for performance, Logistic Regression optionally for interpretability.
- **Fairness**: Evaluated using Equalized Odds & Demographic Parity via `fairlearn`.
- **Explainability**: 
  - **SHAP** for global feature attribution.
  - **LIME** for instance-specific insights.
- **Causal Analysis**: Uses `DoWhy` to estimate the causal effect of smoking.

---

##  File Description

- `heart-disease-fairness-ai.py`: Main script containing all preprocessing, modeling, evaluation, fairness, and explainability steps.
- Input CSV Required: `synthetic_framingham.csv`

---

##  Requirements

Install the necessary dependencies:

```bash
pip install pandas numpy scikit-learn xgboost shap lime matplotlib imbalanced-learn fairlearn dowhy
```

---

##  How to Run

1. Place `synthetic_framingham.csv` in the same directory.
2. Run the script:

```bash
python heart-disease-fairness-ai.py
```

3. Output includes:
   - Accuracy and AUC-ROC
   - SHAP summary plots
   - LIME explanation (shown in notebook)
   - Causal effect estimates
   - Fairness metrics

---

##  Citation

> Maharshi Patel. *Improving Heart Disease Prediction with Causal AI, Explainable AI, and Fairness Checks*. MTSU, 2025.

---

##  License

This project is released for academic and non-commercial use only.
