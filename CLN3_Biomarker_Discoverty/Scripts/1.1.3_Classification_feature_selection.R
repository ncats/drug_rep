# load environment  ####
# Load libraries
rm(list = ls())
library(dplyr)        
library(glmnet)       # LASSO and Ridge Regression
# library(randomForest) # Random Forest
# library(e1071)        # SVM
# library(xgboost)      # Gradient Boosting
library(caret)        # Model training and cross-validation
# library(pROC)         # ROC Curve and AUC
# library(pls)          # Partial Least Squares Regression

# library(magrittr)
# library(readxl)

source('./R/functions.R')
## read data ####
dir_out = "./../" 
("./.../rf_imputed_logistic_data.csv") %>% read.csv(row.names = 1) -> data

# Normalize data (Z-score standardization):  
normalize <- function(x) { (x - mean(x, na.rm = TRUE)) / sd(x, na.rm = TRUE) }
data_normalized <- as.data.frame(lapply(data[, -c(1,2)], normalize))  # Excluding ID and Group

# Add back target variables
data_normalized$Group <- data$Group  
# set to Binary: 0 = Control, 1 = Case
data_normalized$Group <- factor(data$Group, levels = c('Control', 'CLN3'), labels = c(0, 1))

set.seed(42)
trainIndex <- createDataPartition(data_normalized$Group, p = 0.7, list = FALSE)
train_data <- data_normalized[trainIndex, ]
test_data <- data_normalized[-trainIndex, ]

x_train <- as.matrix(train_data[, -ncol(train_data)])
y_train <- train_data$Group
x_test <- as.matrix(test_data[, -ncol(train_data)])
y_test <- test_data$Group

######   build model  - LASSO ####
# this is to test codes to successfully building up the lasso model
glmnet_fit <- cv.glmnet(x_train, y_train, family = "binomial", alpha = 1)


### top feature selection using Bootstrapped feature selection ####

library(glmnet)

# Function to perform bootstrapped LASSO feature selection
stability_selection <- function(x, y, alpha = 0.2, n_boot = 100, threshold = 0.8) {
  selected_counts <- matrix(0, nrow = n_boot, ncol = ncol(x))
  colnames(selected_counts) <- colnames(x)  # Store feature names
  
  for (i in 1:n_boot) {
    # Bootstrap resampling
    boot_idx <- sample(1:nrow(x), replace = TRUE)
    x_boot <- x[boot_idx, ]
    y_boot <- y[boot_idx]
    
    # Fit LASSO with cross-validation
    model <- cv.glmnet(x_boot, y_boot, family = "binomial", alpha = alpha, nfolds = 10)
    
    # Extract non-zero coefficients (selected features)
    selected_features <- coef(model, s = "lambda.min")[-1]  # Remove intercept
    selected_features <- as.numeric(selected_features != 0)
    
    # Store selected features for this bootstrap
    selected_counts[i, ] <- selected_features
  }
  
  # Calculate feature selection frequency
  feature_freq <- colMeans(selected_counts)
  
  # Select stable features appearing in at least `threshold` proportion of runs
  stable_features <- names(feature_freq[feature_freq >= threshold])
  
  return(list(stable_features = stable_features, selection_frequencies = feature_freq))
}

# Apply bootstrapped feature selection
set.seed(123)
result <- stability_selection(x_train, y_train, alpha = 0.2, n_boot = 100, threshold = 0.8)

# Output selected features
cat("Stable Features Selected:\n", result$stable_features)


write.csv(result$stable_features, paste0(dir_out, "logi_boostrap_stable_features_nb500.csv"), row.names = FALSE)
saveRDS(result, paste0(dir_out,"feature_selection_results_nb500.rds"))
