"""Generate the summary report PDF from saved results and visuals."""
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak

res = pd.read_csv("results/model_comparison.csv")
ss = getSampleStyleSheet()
body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=10, leading=14)
h1, h2 = ss["Heading1"], ss["Heading2"]
bullet = ParagraphStyle("bl", parent=body, leftIndent=14, bulletIndent=2)

def img(name, w=15.5):
    from PIL import Image as PI
    iw, ih = PI.open(f"visuals/{name}.png").size
    return Image(f"visuals/{name}.png", width=w * cm, height=w * cm * ih / iw)

def bullets(items):
    return [Paragraph(t, bullet, bulletText="•") for t in items]

S = []
S += [Paragraph("Titanic Survival Prediction: ML Project Report", ss["Title"]),
      Paragraph("Author: Kaviya Sharon &nbsp;|&nbsp; Binary classification &nbsp;|&nbsp; scikit-learn, imbalanced-learn", body),
      Spacer(1, 10)]

S += [Paragraph("1. Problem statement", h1),
      Paragraph("Predict whether a Titanic passenger survived (1) or not (0) from demographic and ticket information. "
                "The dataset has 891 passengers; 38% survived, so the target is moderately imbalanced.", body)]

S += [Paragraph("2. Methodology", h1)]
S += bullets([
    "<b>Cleaning:</b> removed leaky/redundant columns (alive, class, who, adult_male, embark_town, alone) and 'deck' (77% missing).",
    "<b>Feature engineering:</b> family_size and is_alone.",
    "<b>Split:</b> stratified 80/20 train-test split (712 / 179) performed before any fitting.",
    "<b>Missing values:</b> median imputation (age, fare), most-frequent imputation (embarked).",
    "<b>Outliers:</b> detected with the IQR rule; treated by capping (winsorisation) using training-set bounds.",
    "<b>Transformation:</b> log1p on the right-skewed fare feature.",
    "<b>Encoding and scaling:</b> one-hot encoding for sex, embarked, pclass; standardisation of numeric features.",
    "<b>Imbalance:</b> SMOTE oversampling applied to training folds only.",
    "<b>Feature selection:</b> SelectKBest (ANOVA F-score), top 10 features.",
    "<b>Models:</b> Logistic Regression, Random Forest, SVM (RBF), tuned with GridSearchCV (5-fold stratified CV, F1 scoring).",
    "All steps are in a single pipeline so no information from the test set leaks into training.",
])
S += [img("01_eda_target_sex_class"), Spacer(1, 6), img("04_fare_log_transform", 13), PageBreak()]

S += [Paragraph("3. Results (held-out test set, n = 179)", h1)]
cols = ["Model", "Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
data = [cols] + [[r["Model"]] + [f"{r[c]:.3f}" for c in cols[1:]] for _, r in res.iterrows()]
t = Table(data, hAlign="LEFT", colWidths=[4.2 * cm] + [2.4 * cm] * 5)
t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
                       ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                       ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                       ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#ecf0f1")]),
                       ("FONTSIZE", (0, 0), (-1, -1), 9)]))
S += [t, Spacer(1, 8),
      Paragraph("SVM (RBF) achieved the best F1 (0.757), recall (0.812) and ROC-AUC (0.858). Random Forest had the highest "
                "cross-validated F1 (0.747) and the best precision (0.725). Differences between models are small and the test set is "
                "only 179 passengers, so they should be read as comparable rather than decisively different.", body),
      Spacer(1, 8), img("07_confusion_matrices"), img("08_roc_curves", 10), PageBreak()]

S += [Paragraph("4. Insights", h1)]
S += bullets([
    "<b>Sex</b> is by far the strongest predictor of survival, followed by fare, age and passenger class.",
    "Family variables (family_size, is_alone) add a smaller signal; port of embarkation adds little.",
    "Random Forest importances and Logistic Regression coefficients agree on the main drivers, which supports the findings.",
    "All three models reach 78-80% accuracy, typical for this small tabular dataset.",
])
S += [img("09_rf_feature_importance", 12), Spacer(1, 6), img("10_lr_coefficients", 12)]

S += [Paragraph("5. Limitations and future work", h1)]
S += bullets([
    "Small dataset (891 rows); results on 179 test passengers have noticeable variance.",
    "Age was median-imputed; I did not run ablations to isolate the effect of SMOTE or outlier capping.",
    "Future work: extract titles from names, add cabin deck, try XGBoost / LightGBM, and use repeated cross-validation.",
])
SimpleDocTemplate("Summary_Report.pdf", pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                  topMargin=2 * cm, bottomMargin=2 * cm).build(S)
