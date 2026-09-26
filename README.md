# Feature Selection and KNN Classification (scikit-learn)

A statistical learning study of how **feature selection**, **feature scaling** and the **number of neighbors (k)** affect a K-Nearest Neighbors classifier, tested on simulated data and real datasets.

## Results

| Dataset | Setup | Result |
|---|---|---|
| Iris (3 classes, 150 samples) | Features chosen by chi-squared test (petal length, petal width), k = 7 | **100% test accuracy**, 96.7% train |
| Iris | 5-fold cross-validation over k | **96.7%** best CV accuracy |
| COVID-19 patient records (tabular) | Chi-squared feature ranking, KNN | ~66–68% best 5-fold CV accuracy |
| Simulated Gaussian data | With and without feature selection | 100% accuracy (clean, well-separated classes) |

The same method that is near-perfect on clean data drops sharply on noisy real-world medical records. That gap shows how much the dataset and the chosen features matter.

## Approach

1. **Simulated data:** generated K Gaussian classes where only the first p₁ of p features carry class information and the rest are noise. Varied noise level (σ²), number of features, relevant features and sample size to study their effect on error.
2. **Feature selection** with two methods:
   - **Variance threshold:** keep features whose variance exceeds a threshold.
   - **Chi-squared contingency test:** keep features most associated with the class label.
3. **Scaling:** Min-Max normalization, compared with unscaled features.
4. **Model tuning:** accuracy plotted against k, then `GridSearchCV` with 5-fold cross-validation to pick the best k.
5. **Evaluation:** accuracy, confusion matrix, classification report, ROC-AUC and decision-boundary plots.

## Files

| File | Contents |
|---|---|
| `sim_data.ipynb` | Simulated data experiments |
| `real_data2.ipynb` | Iris: chi-squared feature selection, scaling, decision boundaries, cross-validation |
| `SL_project.ipynb` | COVID-19 patient dataset: chi-squared tests, KNN, ROC-AUC, cross-validation |

## Reports

- [Project brief (PDF)](https://github.com/Bhuvanavenkatappa77/Python-Projects/files/11006653/ece730project.pdf)
- [Final project report (PDF)](https://github.com/Bhuvanavenkatappa77/Python-Projects/files/11006651/FINAL.PROJECT.REPORT.SL.pdf)

## How to run

```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn
jupyter notebook
```

## Tech

Python · scikit-learn · KNN · Feature selection · Chi-squared test · Cross-validation · GridSearchCV · pandas · SciPy · Matplotlib · Seaborn

## What I learned

- Picking the right features can matter more than tuning the model.
- KNN depends on distances, so feature scaling changes results.
- Cross-validation gives a more honest estimate than a single train/test split, especially on small datasets like Iris.
- Near-perfect scores on synthetic data don't transfer to messy real-world data.
