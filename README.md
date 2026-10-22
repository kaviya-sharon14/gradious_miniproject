# Titanic Survival Prediction: End-to-End ML Project

Binary classification project that predicts whether a Titanic passenger survived, covering the full ML workflow: preprocessing, model training, tuning, evaluation and interpretation.

## Results (held-out test set, 179 passengers)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| SVM (RBF) | 0.799 | 0.709 | 0.812 | 0.757 | 0.858 |
| Logistic Regression | 0.782 | 0.692 | 0.783 | 0.735 | 0.823 |
| Random Forest | 0.788 | 0.725 | 0.725 | 0.725 | 0.842 |

Sex is the strongest predictor of survival, followed by fare, age and passenger class. See `Summary_Report.pdf` for details.

## Project structure

```
.
├── titanic_ml_project.ipynb   # main notebook (already executed, outputs included)
├── data/titanic.csv           # dataset (seaborn-data Titanic)
├── Summary_Report.pdf         # problem, methodology, results, insights
├── make_report.py             # regenerates the PDF from results/ and visuals/
├── results/                   # model_comparison.csv, best_params.json
├── visuals/                   # saved plots (EDA, confusion matrices, ROC, importances)
├── requirements.txt
└── README.md
```

## Techniques used

Missing-value imputation, IQR outlier detection and capping, log transform, one-hot encoding, standardisation, SMOTE for class imbalance, ANOVA feature selection, stratified train-test split, 5-fold stratified cross-validation with GridSearchCV, Logistic Regression / Random Forest / SVM.

## How to run

1. Install Python 3.9+.
2. (Recommended) create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Launch the notebook and run all cells (Kernel > Restart & Run All):
   ```bash
   jupyter notebook titanic_ml_project.ipynb
   ```
5. (Optional) rebuild the PDF report after running the notebook:
   ```bash
   python make_report.py
   ```

Run the notebook from the project root so the relative paths `data/`, `visuals/` and `results/` resolve. Everything uses `random_state=42`, so results are reproducible.

## Dependencies

numpy, pandas, scipy, matplotlib, seaborn, scikit-learn, imbalanced-learn, jupyter, nbformat, reportlab, pillow

## Dataset

Titanic passenger data from the [seaborn-data repository](https://github.com/mwaskom/seaborn-data).
