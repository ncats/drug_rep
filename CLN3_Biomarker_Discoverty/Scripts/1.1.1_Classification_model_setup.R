# Install missing packages if necessary
install.packages(c("glmnet", "randomForest", "e1071", "xgboost", "caret", "pROC", "pls"))

# Load libraries
library(dplyr)        
library(glmnet)       # LASSO and Ridge Regression
library(randomForest) # Random Forest
library(e1071)        # SVM
library(xgboost)      # Gradient Boosting
library(caret)        # Model training and cross-validation
library(pROC)         # ROC Curve and AUC
library(pls)          # Partial Least Squares Regression

# Load and Preprocess  case-control dataset (cln3 vs. non-cln3) 
("./.../rf_imputed_logistic_data.csv") %>% read.csv(row.names = 1) -> data

# Normalize data (Z-score standardization): optinal 
normalize <- function(x) { (x - mean(x, na.rm = TRUE)) / sd(x, na.rm = TRUE) }
data_normalized <- as.data.frame(lapply(data[, -c(1,2)], normalize))  # Excluding ID and Group

# Add back target variables
data_normalized$Group <- data$Group  
# set to Binary: 0 = Control, 1 = Case
data_normalized$Group <- factor(data$Group, levels = c('Control', 'CLN3'), labels = c(0, 1))


# Split into train and test 
set.seed(42)
trainIndex <- createDataPartition(data_normalized$Group, p = 0.7, list = FALSE)
train_data <- data_normalized[trainIndex, ]
test_data <- data_normalized[-trainIndex, ]

# Logistic Regression (Baseline)
logit_model <- glm(Group ~ ., data = train_data, family = binomial)
logit_preds <- predict(logit_model, test_data, type = "response")
logit_auc <- roc(test_data$Group, logit_preds)$auc
print(logit_auc)

# LASSO Logistic Regression (Feature Selection)
x_train <- as.matrix(train_data[, -ncol(train_data)])
y_train <- train_data$Group


lasso_model <- cv.glmnet(x_train, y_train, alpha = 1, family = "binomial")
lasso_best_lambda <- lasso_model$lambda.min

lasso_preds <- predict(lasso_model, as.matrix(test_data[, -ncol(test_data)]), s = lasso_best_lambda, type = "response")
lasso_auc <- roc(test_data$Group, lasso_preds)$auc
print(lasso_auc)

# Random Forest

rf_model <- randomForest(Group ~ ., data = train_data, ntree = 500)
rf_preds <- predict(rf_model, test_data, type = "prob")[, 2]
rf_auc <- roc(test_data$Group, rf_preds)$auc
print(rf_auc)

# Support Vector Machine (SVM)
svm_model <- svm(Group ~ ., data = train_data, probability = TRUE)
svm_preds <- predict(svm_model, test_data, probability = TRUE)
svm_auc <- roc(test_data$Group, attr(svm_preds, "probabilities")[, 2])$auc
print(svm_auc)


# XGBoost

xgb_train <- xgb.DMatrix(data = as.matrix(train_data[, -ncol(train_data)]), label = as.numeric(as.character(train_data$Group)))
xgb_test <- xgb.DMatrix(data = as.matrix(test_data[, -ncol(test_data)]))

xgb_model <- xgboost(data = xgb_train, max_depth = 3, eta = 0.1, nrounds = 100, objective = "binary:logistic")
xgb_preds <- predict(xgb_model, xgb_test)
xgb_auc <- roc(test_data$Group, xgb_preds)$auc
print(xgb_auc)

