# Model Performance Comparison

results <- data.frame(
  Model = c("Logistic", "LASSO Logistic", "Random Forest", "SVM", "XGBoost"),
  AUROC = c(logit_auc, lasso_auc, rf_auc, svm_auc, xgb_auc)
)
print(results)

# ROC Curve for Classification Models

logit_preds <- as.numeric(logit_preds)
lasso_preds <- as.numeric(lasso_preds)
rf_preds <- as.numeric(rf_preds)
svm_preds <- as.numeric(svm_preds)
xgb_preds <- as.numeric(xgb_preds)


pdf('./.../Classification_models_.7_AUROC.pdf', width = 6, height = 6)
plot(roc(test_data$Group, logit_preds), col = "red", main = "ROC Curve Comparison")
lines(roc(test_data$Group, lasso_preds), col = "blue")
lines(roc(test_data$Group, rf_preds), col = "green")
lines(roc(test_data$Group, svm_preds), col = "purple")
lines(roc(test_data$Group, xgb_preds), col = "black")
legend("bottomright", legend = c("Logistic", "LASSO", "RF", "SVM", "XGBoost"), col = c("red", "blue", "green", "purple", "black"), lty = 1)
dev.off()
# Define Evaluation Functions

# Function to compute classification metrics
evaluate_classification <- function(y_true, y_pred_prob, threshold = 0.5) {
  y_pred <- ifelse(y_pred_prob > threshold, 1, 0)
  
  confusion <- confusionMatrix(as.factor(y_pred), as.factor(y_true), positive = "1")
  auc <- roc(y_true, y_pred_prob)$auc
  
  return(data.frame(
    Accuracy = confusion$overall["Accuracy"],
    Sensitivity = confusion$byClass["Sensitivity"],
    Specificity = confusion$byClass["Specificity"],
    Precision = confusion$byClass["Precision"],
    F1_Score = confusion$byClass["F1"],
    Balanced_Accuracy = confusion$byClass["Balanced Accuracy"],
    AUROC = auc
  ))
}
# or 
source('./R/functions.R')



## model evaluation 
logit_metrics <- evaluate_classification(test_data$Group, logit_preds)
rf_metrics <- evaluate_classification(test_data$Group, rf_preds)
svm_metrics <- evaluate_classification(test_data$Group, attr(svm_preds, "probabilities")[, 2])
xgb_metrics <- evaluate_classification(test_data$Group, xgb_preds)


# Classification Results Table
classification_results <- rbind(
  cbind(Model = "Logistic Regression", logit_metrics),
  cbind(Model = "LASSO Logistic Regression", lasso_metrics),
  cbind(Model = "Random Forest", rf_metrics),
  cbind(Model = "Support Vector Machine", svm_metrics),
  cbind(Model = "XGBoost", xgb_metrics)
)

print(classification_results)
write.csv(classification_results, "./.../classification_model_evaluation_results.csv", row.names = FALSE)
