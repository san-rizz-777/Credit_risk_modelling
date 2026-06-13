## Credit Risk Modeling

### Problem
Classify loan default risk with limited data (552 samples, 55/45 class split)

### Approach
- XGBoost classifier with GridSearchCV hyperparameter tuning
- 5-fold stratified cross-validation
- Evaluated on: Accuracy, Recall, AUC-ROC, Precision

### Results
- **AUC-ROC: 0.762** (good discrimination ability)
- **Accuracy: 70%** (with 9.5% variance across folds)
- **Recall: 70%** (catches 70% of actual defaults)

### Key Learnings
- Accuracy alone misleading with imbalanced/small datasets
- AUC-ROC better metric for credit modeling
- High cross-validation variance indicates need for more data
- Data quality > model complexity with limited samples

### Limitations & Future Work
- Small sample size (552) limits generalization
- Data quality issues constrain performance ceiling
- Would need 1,500+ samples for production-ready model