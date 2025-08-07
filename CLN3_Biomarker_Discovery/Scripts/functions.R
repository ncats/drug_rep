calculate_auc <- function(predictions, true_labels) {
  library(pROC)
  roc_obj <- roc(true_labels, predictions)
  auc(roc_obj)
}


calculate_mse <- function(predictions, true_labels) {
  mse <- mean((predictions - true_labels)^2)
  return(mse)
}

calculate_rmse <- function(predictions, true_labels) {
  rmse <- sqrt(mean((predictions - true_labels)^2))
  return(rmse)
}

calculate_mae <- function(predictions, true_labels) {
  mae <- mean(abs(predictions - true_labels))
  return(mae)
}

calculate_metrics <- function(predictions, true_labels) {
  mse <- mean((predictions - true_labels)^2)
  rmse <- sqrt(mse)
  mae <- mean(abs(predictions - true_labels))
  ss_total <- sum((true_labels - rowMeans(true_labels))^2)
  ss_residual <- sum((true_labels - predictions)^2)
  r_squared <- 1 - (ss_residual / ss_total)
  
}

write_list_csv <-function(my_list, defalt = "result") {   
  
  l <- my_list %>% length()
  for (i in 1:l) {
    name <- names(my_list)[i]
    my_list[[i]] %>% as.matrix() %>% write.table(file = paste0(defalt, "_", name, ".csv" ))
    
    }
  
}


# Function to calculate RMSE
calculate_rmse <- function(imputed_data) {
  imputed_values <- imputed_data[pseudo_missing_mask]  # Extract imputed values at masked positions
  
  # Ensure valid values (no NAs)
  valid_indices <- !is.na(observed_values) & !is.na(imputed_values)
  
  # Compute RMSE
  rmse <- sqrt(mean((observed_values[valid_indices] - imputed_values[valid_indices])^2))
  
  return(rmse)
}


# function caculate missing value portion
missing_values_rate <- function(df) {
  (sum(is.na(df)) / (nrow(df) * ncol(df))) * 100
}

# Function to randomly mask 10% of observed values in a given column range
mask_random_values <- function(df, col_start, col_end) {
  for (col in col_start:col_end) {
    observed_indices <- which(!is.na(df[, col]))  # Get indices of observed (non-NA) values
    num_to_mask <- round(0.1 * length(observed_indices))  # 10% of observed values
    if (num_to_mask > 0) {
      masked_indices <- sample(observed_indices, num_to_mask)  # Randomly select indices to mask
      df[masked_indices, col] <- NA  # Apply masking
    }
  }
  return(df)
}

# Define model performance Evaluation Functions

## Function to compute classification metrics
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

## Function to compute regression metrics
evaluate_regression <- function(y_true, y_pred) {
  return(data.frame(
    MSE = mse(y_true, y_pred),
    RMSE = rmse(y_true, y_pred),
    MAE = mae(y_true, y_pred),
    R_Squared = cor(y_true, y_pred)^2,
    MAPE = mean(abs((y_true - y_pred) / y_true)) * 100
  ))
}

