# CVDprediction: An R Package for Cardiovascular Disease Prediction
A reproducible, user-friendly R package to preprocess tabular heart disease data, train random forest models, and generate CVD (Cardiovascular Disease) predictions—aligned with BIO215 course project requirements.

## Overview
CVDprediction simplifies the end-to-end workflow for cardiovascular disease prediction using machine learning. It addresses key challenges in medical data analysis:

The package provides:
- Prevents data leakage through training-only preprocessing
- Implements reproducible model training with cross-validation
- Provides clinically relevant evaluation metrics (AUROC, AUPRC, feature importance)
- Supports both direct R usage and integration with Shiny apps
- Follows R package best practices with full Roxygen2 documentation

---
## Dataset
The package is built around the publicly available Risk Factors for Cardiovascular Heart Disease Dataset (hosted on Kaggle by Kuzak Dempsy), containing over 70,000 samples of adult health metrics.

The dataset includes 14 features (clinical & demographic measurements) and a binary target variable:
Features: age (days), gender, height, weight, ap_hi (systolic blood pressure), ap_lo (diastolic blood pressure), cholesterol, gluc (glucose level), smoke, alco, active

## Installation

### Before installation, ensure that the dependency package is installed
```r
install.packages("devtools")    # or: install.packages("remotes"). # or: install.packages("randomForest")
```
### Install CVDprediction from GitHub
You can install the package from GitHub using remotes or devtools:
```r
devtools::install_github("ZihanHe-xjtlu/HeartPrediction")
library(HeartPrediction)
```

```r
remotes::install_github("ZihanHe-xjtlu/HeartPrediction") 
library(HeartPrediction)
```
---
## Example Usage

### Data Preprocessing
```r
set.seed(42) 
preprocess_res <- preprocess_heart_data(
  data_path = system.file("extdata", "heart_data.csv", package = "HeartPrediction")
)

head(preprocess_res$cleaned_data)  # Full cleaned dataset

```
### Model Trainings
```r
set.seed(42)
trained_model <- train_cvd_rf(
  cleaned_data = cleaned_data,
  cv_folds = 5,    # Number of cross-validation folds (default: 5)
  num_trees = 400  # Number of trees in the random forest (balances performance/speed)
)
cat("Best Hyperparameters:\n")
print(trained_model$model$bestTune)
```

### Evaluate model performance
```r
eval_results <- evaluate_cvd_model(model_obj = trained_model)

cat("\n=== Test Set Performance Metrics ===\n")
cat("AUROC:", round(eval_results$auroc, 3), "\n")          
cat("AUPRC:", round(eval_results$auprc, 3), "\n")        
cat("Accuracy:", round(eval_results$confusion_matrix$overall["Accuracy"], 3), "\n")
cat("Sensitivity:", round(eval_results$confusion_matrix$byClass["Sensitivity"], 3), "\n")
cat("Specificity:", round(eval_results$confusion_matrix$byClass["Specificity"], 3), "\n")  

ggplot(eval_results$roc_curve_data, aes(x = FPR, y = TPR)) +
  geom_line(linewidth = 1.2, color = "#2E86AB") +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray50") +
  labs(
    title = "ROC Curve (Test Set)",
    subtitle = paste("AUROC =", round(eval_results$auroc, 3)),
    x = "False Positive Rate",
    y = "True Positive Rate"
  ) +
  theme_classic(base_size = 12)

```

### Prediction
```r
new_patients <- tibble::tibble(
  age = c(22000, 18250),  # Age in days (~60 and ~50 years old)
  gender = factor(c(2, 1), levels = c(1, 2), labels = c("Female", "Male")),  # Male, Female
  height = c(178, 165),   # Height in centimeters
  weight = c(85, 68),     # Weight in kilograms
  ap_hi = c(145, 130),    # Systolic blood pressure (mmHg)
  ap_lo = c(92, 85),      # Diastolic blood pressure (mmHg)
  cholesterol = factor(c(2, 1), levels = c(1, 2, 3)),  # Above normal, Normal
  gluc = factor(c(1, 1), levels = c(1, 2, 3)),         # Normal blood glucose
  smoke = factor(c(0, 1), levels = c(0, 1)),           # Non-smoker, Smoker
  alco = factor(c(1, 0), levels = c(0, 1)),            # Alcohol consumer, Non-consumer
  active = factor(c(0, 1), levels = c(0, 1))           # Inactive, Physically active
)

predictions <- predict_cvd(
  model_obj = trained_model,
  new_data = new_patients
)

print(predictions)
```
---
## Model Performance
Our core Random Forest model demonstrates strong discriminative power for CVD risk prediction, validated on an independent test set:
### 1. ROC Curve
![ROC Curve](inst/images/ROC_Curve1.png)
ROC curve with AUROC 0.797: Strong ability to distinguish CVD cases

### 2. Precision-Recall Curve
![PRC Curve](inst/images/PRC_Curve1.png)
PRC curve with AUPRC 0.783: Reliable for imbalanced clinical screening.

### 3. Top 20 Permutation Importance
![Top 20 Permutation Importance](inst/images/Top20.png)
Key predictors: Systolic BP (ap_hi), diastolic BP (ap_lo), pulse pressure (clinically validated).

### 4. SHAP Beeswarm Plot (Top 8 Features)
![SHAP Beeswarm Plot](inst/images/SHAP_Plot.png)
SHAP values: Positive = increased CVD risk (e.g., high ap_hi/age), negative = decreased risk.
---
## License

MIT License
---
## Citation
If you use this package in articles, please cite:
Zihan He. (2025)   HeartPrediction – CVDprediction: An R Package for Cardiovascular Disease Prediction. GitHub: ZihanHe-xjtlu/HeartPrediction
---
## Contact
https://github.com/ZihanHe-xjtlu/HeartPrediction/issues￼
