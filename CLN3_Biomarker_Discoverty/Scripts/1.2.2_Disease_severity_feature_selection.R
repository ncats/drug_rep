# load environment  ####
# Load libraries
rm(list = ls())
library(dplyr)        
library(glmnet)       # LASSO and Ridge Regression
library(randomForest) # Random Forest
library(randomForestExplainer)
library(fastshap)
library(randomForestSRC) # Multivariate Random Forest (RF) (RF )

# library(e1071)        # SVM
# library(xgboost)      # Gradient Boosting
library(caret)        # Model training and cross-validation
library(openxlsx)
# library(pROC)         # ROC Curve and AUC
# library(pls)          # Partial Least Squares Regression

# library(magrittr)
# library(readxl)


source('./R/functions.R')
## read data ####
dir_out = "./.../" 

# Load and Preprocess Data
("./.../PCA_imputed_Linear_data.csv") %>% read.csv() -> data

# Normalize data (Z-score standardization)
normalize <- function(x) { (x - mean(x, na.rm = TRUE)) / sd(x, na.rm = TRUE) }
severity_normalized <- as.data.frame(lapply(data[, -(1:6)], normalize)) # Excluding ID and target severity scores, include age as variable 

severity_normalized <- cbind(data[, c("UBDRS_PHYSICAL_WEIGHTED_SCORE", "UBDRS_CAPABILITY_WEIGHTED_SCORE", 
                                      "UBDRS_CGI_WEIGHTED_SCORE", "COG_VERBAL_IQ", "Vine_COMPOSITE_STANDARD_SCORE")], severity_normalized)


# Prepare data
x <- severity_normalized[, -(1:5)]
y <- severity_normalized[, 1:5]

data_combined <- as.data.frame(cbind(y, x))  # Combine predictors & responses


# Fit separate random forest models for each response variable (as impoartance cannot be evaluted in combined response model cannot)
importance_list <- list()
for (i in 1:ncol(y)) {
  response_name <- colnames(y)[i]
  formula <- as.formula(paste(response_name, "~ ."))
  rf_model <- randomForest(formula, data = cbind(y[, i, drop = FALSE], x), importance = TRUE)
  importance <- importance(rf_model, type = 1)
  importance_list[[response_name]] <- importance
}

# save the list: Load necessary library
library(openxlsx)

wb <- createWorkbook()
for (response_name in names(importance_list)) {
  sheet_data <- data.frame(Feature = rownames(importance_list[[response_name]]), 
                           Importance = importance_list[[response_name]])
  addWorksheet(wb, response_name)
  writeData(wb, sheet = response_name, x = sheet_data)
}
saveWorkbook(wb,  paste0(dir_out, "RF_feature_importance.xlsx"), overwrite = TRUE)


# Aggregate importance values
importance_df <- do.call(cbind, importance_list)
importance_means <- rowMeans(importance_df, na.rm = TRUE)
importance_df <- data.frame(
  Feature = rownames(importance_df),
  Importance = importance_means
)

# Sort by importance
importance_df <- importance_df[order(importance_df$Importance, decreasing = TRUE), ]
write.csv(importance_df,  paste0(dir_out, "RF_aggregated_feature_importance.csv"), row.names = FALSE)

# Visualize feature importance using ggplot2
top_n <- 20
top_features <- head(importance_df, top_n)
pdf(paste0(dir_out, "RF_top_20_features_barplot.pdf"), width = 6, height = 6)
ggplot(top_features, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  labs(title = "Top Feature Importance", x = "Features", y = "Importance")

dev.off()


### shap
# Load necessary libraries
library(fastshap)
library(ggplot2)

# Assuming `rf_model` is the random forest model for one of the response variables
# For demonstration, let's use the first response variable model from the list
response_name <- colnames(y)[1]
rf_model <- randomForest(as.formula(paste(response_name, "~ .")), data = cbind(y[, 1, drop = FALSE], x), importance = TRUE)

# Use fastshap to explain the model
shap_values <- fastshap::explain(
  object = rf_model,
  X = x,
  pred_wrapper = function(object, newdata) {
    predict(object, newdata = newdata, type = "response")
  },
  nsim = 50  # Number of Monte Carlo simulations
)

# Convert SHAP values to a long format for ggplot2
shap_long <- shap_values %>%
  reshape2::melt(varnames = c("Feature", "Sample"), value.name = "SHAP") %>%
  dplyr::mutate(Feature = factor(Feature, levels = names(sort(colMeans(shap_values), decreasing = TRUE))))

# Visualize SHAP values for the top features
top_n <- 10
top_features <- head(names(sort(colMeans(shap_values), decreasing = TRUE)), top_n)
shap_top <- dplyr::filter(shap_long, Feature %in% top_features)

ggplot(shap_top, aes(x = Feature, y = SHAP)) +
  geom_boxplot() +
  coord_flip() +
  labs(title = paste("SHAP Values for", response_name), x = "Features", y = "SHAP Value")








# Define the formula with multiple response variables
response_vars <- paste(colnames(y), collapse = " + ")
formula <- as.formula(paste("cbind(", paste(colnames(y), collapse = ", "), ") ~ ."))

# Fit multivariate Random Forest
rf_multi <- rfsrc(formula, data = data_combined, ntree = 500, importance = TRUE)



# Extract feature importance using the vimp function
vimp_values <- vimp(rf_multi)

# Convert importance values to a data frame
importance_df <- data.frame(
  Feature = names(vimp_values$importance),
  Importance = as.vector(vimp_values$importance)
)


# # Extract feature importance using randomForestSRC's vimp function
# # importance_frame <- vimp(rf_multi)
# # importance_values <- importance_frame$importance
# importance_values <- rf_multi$importance
# 
# # Check if importance_values is a matrix, and if so, convert it to a vector
# if (is.matrix(importance_values)) {
#   importance_values <- rowMeans(importance_values, na.rm = TRUE)
# }
# 
# # Convert importance values to a data frame
# importance_df <- data.frame(
#   Feature = names(importance_values),
#   Importance = importance_values
# )


# Sort by importance
importance_df <- importance_df[order(importance_df$Importance, decreasing = TRUE),]

# Visualize feature importance using ggplot2
top_n <- 10
top_features <- head(importance_df, top_n)
ggplot(top_features, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  labs(title = "Top Feature Importance", x = "Features", y = "Importance")
# Extract feature importance using randomForestSRC's vimp function
importance_frame <- vimp(rf_multi)
importance_values <- importance_frame$importance

# Convert importance values to a data frame
importance_df <- data.frame(
  Feature = names(importance_values),
  Importance = importance_values
)

# Sort by importance
importance_df <- importance_df[order(importance_df$Importance, decreasing = TRUE), ]

# Visualize feature importance using ggplot2
top_n <- 10
top_features <- head(importance_df, top_n)
ggplot(top_features, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  labs(title = "Top Feature Importance", x = "Features", y = "Importance")


# Explain feature importance using fastshap
shap_values <- fastshap::explain(rf_multi, X = x, pred_wrapper = function(object, newdata) predict(object, newdata = newdata)$predicted, nsim = 50)

# Visualize SHAP values for the top features
library(ggplot2)
for (feature in rownames(top_features)[1:10]) {
  shap_plot <- autoplot(shap_values, feature = feature)
  print(shap_plot)
}


####### boostraping of individual models #############

# Prepare data
x <- severity_normalized[, -(1:5)]
y <- severity_normalized[, 1:5]

# Number of bootstrapped samples
n_bootstraps <- 50

# Initialize list to store aggregated importance
aggregated_importance <- list()

# Fit random forest models for each response variable on bootstrapped samples
for (i in 1:ncol(y)) {
  response_name <- colnames(y)[i]
  formula <- as.formula(paste(response_name, "~ ."))
  
  # Initialize list to store importance frames for bootstrapped samples
  importance_frames <- list()
  
  for (b in 1:n_bootstraps) {
    # Create a bootstrap sample
    bootstrap_indices <- sample(1:nrow(data_combined), replace = TRUE)
    bootstrap_sample <- data_combined[bootstrap_indices, ]
    
    # Fit random forest model on the bootstrap sample
    rf_model <- randomForest(formula, data = bootstrap_sample, localImp = TRUE, na.action = na.roughfix)
    
    # Evaluate feature importance using randomForestExplainer
    importance_frame <- measure_importance(rf_model, mean_sample = "relevant_trees")
    
    # Rename columns to avoid duplication during merge
    colnames(importance_frame) <- paste(colnames(importance_frame), b, sep = "_")
    colnames(importance_frame)[1] <- "variable"  # Keep the variable column name consistent
    importance_frames[[b]] <- importance_frame
  }
  
  # Aggregate the importance frames across bootstrapped samples
  aggregated_frame <- Reduce(function(x, y) merge(x, y, by = "variable", all = TRUE), importance_frames)
  aggregated_frame <- aggregated_frame %>%
    rowwise() %>%
    mutate(aggregated_importance = mean(c_across(starts_with("mean_min_depth")), na.rm = TRUE)) %>%
    ungroup()
  
  # Store the aggregated importance frame in the list
  aggregated_importance[[response_name]] <- aggregated_frame
  
  # Save the aggregated importance frame
  save(aggregated_frame, file = paste(dir_out, paste0("RF_aggregated_importance_frame_", response_name, ".rda"), sep = ""))
  
  # Plot the aggregated importance measures
  pdf(paste(dir_out, paste0("RF_Aggregated_Importance_", response_name, ".pdf"), sep = ""), width = 9, height = 6)
  ggplot(aggregated_frame, aes(x = reorder(variable, aggregated_importance), y = aggregated_importance)) +
    geom_bar(stat = "identity") +
    coord_flip() +
    labs(title = paste("Aggregated Feature Importance for", response_name), x = "Features", y = "Aggregated Importance")
  dev.off()
}


# Save the aggregated importance list to an Excel file
wb <- createWorkbook()

for (response_name in names(aggregated_importance)) {
  sheet_data <- aggregated_importance[[response_name]]
  addWorksheet(wb, response_name)
  writeData(wb, sheet = response_name, x = sheet_data)
}

saveWorkbook(wb, paste(dir_out, "RF_Boostrap_Aggregated_Feature_Importance.xlsx", sep = ""), overwrite = TRUE)



#### feature selection from boostraped models ####
# Load necessary libraries
library(dplyr)
library(openxlsx)
library(ggplot2)

# File paths
# dir_out <- "./Outpst/"

# Load the aggregated importance data
wb <- loadWorkbook(paste(dir_out, "RF_Boostrap_Aggregated_Feature_Importance.xlsx", sep = ""))


##### z-score ####

# Load necessary libraries
library(dplyr)
library(openxlsx)
library(ggplot2)

# File paths
dir_out <- "./Outpst/"

# Load the aggregated importance data
wb <- loadWorkbook(paste(dir_out, "aggregated_importance.xlsx", sep = ""))

# Initialize list to store top features
top_features_list <- list()

# Define the z-score cutoff for selecting significant features
z_score_cutoff <- 2

### top features with aggregated significance

for (response_name in names(wb)) {
  # Read the sheet
  aggregated_frame <- read.xlsx(wb, sheet = response_name)
  
  # Calculate the mean and standard deviation of aggregated_importance
  mean_importance <- mean(aggregated_frame$aggregated_importance, na.rm = TRUE)
  sd_importance <- sd(aggregated_frame$aggregated_importance, na.rm = TRUE)
  
  # Calculate the z-score for each feature
  aggregated_frame <- aggregated_frame %>%
    mutate(z_score = (aggregated_importance - mean_importance) / sd_importance)
  
  # Select features with z-scores greater than the cutoff
  significant_features <- aggregated_frame %>%
    filter(z_score > z_score_cutoff) %>%
    arrange(desc(z_score)) %>%
    select(variable, aggregated_importance, z_score)  # Keep necessary columns
  
  # Store the significant features
  top_features_list[[response_name]] <- significant_features
}


# Save the significant features to a new Excel file
wb_top_features <- createWorkbook()
for (response_name in names(top_features_list)) {
  addWorksheet(wb_top_features, response_name)
  writeData(wb_top_features, sheet = response_name, top_features_list[[response_name]])
}
saveWorkbook(wb_top_features, paste(dir_out, "RF_boostrap_zScore2_top_features.xlsx", sep = ""), overwrite = TRUE)


