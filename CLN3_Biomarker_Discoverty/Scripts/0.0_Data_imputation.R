## 1. missing value check #### 

library(dplyr)
path_in <- "./.."

# checking missing value rate
prot_data <- read.csv(paste0(path_in, "pea+ms.csv"), row.names = 1)

# define a function cacluate missing value portion
missing_values_rate <- function(df) {
  (sum(is.na(df)) / (nrow(df) * ncol(df))) * 100
}

#check missing value rates
paste0(path_in, "pea.csv") %>% read.csv(row.names = 1) %>% missing_values_rate()
paste0(path_in, "ms.csv") %>% read.csv(row.names = 1) %>% missing_values_rate()
paste0(path_in, "lab.csv") %>% read.csv(row.names = 1) %>% missing_values_rate()
paste0(path_in, "UBDRS_Clean.csv") %>% read.csv(row.names = 1) %>% missing_values_rate()



## 2. Imputation method evaluation #### 
#### 2.1 Imputation of classification dataset  ####
# Load necessary libraries
library(DMwR2)         # kNN imputation
library(mice)         # MICE imputation
library(missMDA)      # PCA-based imputation
library(caret)        # Model evaluation
library(ggplot2)      # Visualization
library(softImpute)   # Soft-Impute
library(missForest)   # Random Forest-Based Imputation 
library(VIM)          #  Hot-Deck Imputation and Regression Imputation
library(foreach)
library(doParallel)


# read raw data
prot_data <- read.csv(paste0(path_in, "pro.data.csv"), row.names = 1)
data <- prot_data[,-(1:2)]

# Replace missing values with NA
data[data == ""] <- NA
pseudo_missing <- data  # Replace "data" with  actual dataset
pseudo_missing %>% dim()
#### 2.1.1 step 1. Create masks based on  observed missingness pattern ####
# a. Mask values in columns 1:1467 for specific rows
rows_group1 <- which(rowSums(is.na(data[, 1:1467])) > (1467*0.9))  
observed_indices_group1 <- setdiff(1:nrow(data), rows_group1)
masked_indices_group1 <- sample(1:length(observed_indices_group1), size = round(0.1 * length(observed_indices_group1)))
pseudo_missing[masked_indices_group1,1:1467] <- NA

# b. Mask values in columns 1468:4786 for specific rows
rows_group2 <- which(rowSums(is.na(data[, 1468:4786])) > (4786 - 1466) * 0.9)   
observed_indices_group2 <- setdiff(1:nrow(data), rows_group2)
masked_indices_group2 <- sample(1:length(observed_indices_group2), size = round(0.1 * length(observed_indices_group2)))
pseudo_missing[masked_indices_group2, 1468:4786] <- NA

# now pseudo_missing contains the original missing values + pseudo missing values
saveRDS(pseudo_missing, "./.../pseudo_missinga.RDS")

####  2.1.2. step 2. imputation using different methods ####

######  method 1: PCA-Based Imputation ####
# nb <- estim_ncpPCA(pseudo_missing, method.cv = "Kfold", verbose = FALSE) 
# nb$ncp
pca_imputed <- imputePCA(pseudo_missing, 2)  # 'ncp' is the number of components to retain
pca_imputed_data <- pca_imputed$completeObs
saveRDS(pca_imputed_data, "./.../pca_imputed_data.RDS")

######  method 2: Soft-Impute #### 

library(softImpute)
pseudo_missing[is.infinite(pseudo_missing)] <- NA  # Replace Inf with NA
pseudo_missing[is.nan(pseudo_missing)] <- NA      # Replace NaN with NA

# Soft-Impute
set.seed(123)
lambda_opt <- softImpute::lambda0(pseudo_missing, maxit = 100)
soft_imputed <- softImpute::softImpute(pseudo_missing, rank.max = 50, lambda = lambda_opt, type = "svd")

# Convert imputed matrix to data frame
soft_imputed_data <- softImpute::complete(pseudo_missing, soft_imputed)
saveRDS(soft_imputed_data, "./.../soft_imputed_data.RDS")


######  method 3: Random Forest-Based Imputation (missforest package) ####
library(foreach)
library(doParallel)

num_cores <- detectCores() - 1  # Use all cores except one
cl <- makeCluster(num_cores)
registerDoParallel(cl)

# Run missForest in parallel
set.seed(123)
missforest_imputed <- missForest(pseudo_missing, 
                                 maxiter = 10, 
                                 ntree = 100, 
                                 parallelize = "variables")  # Use "forests" for tree-level parallelism

# Stop parallel backend (after finishing imputation)
stopCluster(cl)
rf_imputed_data <- missforest_imputed$ximp
saveRDS(rf_imputed_data, "./.../rf_imputed_data.RDS")


######  method 4:  Hot-Deck Imputation ####
set.seed(123)

hotdeck_imputed <- hotdeck(pseudo_missing, ord_var = NULL)
hotdeck_imputed <- hotdeck_imputed[, colnames(pseudo_missing)]  ## remove extra columns created during imputation
hotdeck_imputed_data <- as.data.frame(hotdeck_imputed)
saveRDS(hotdeck_imputed_data, "./.../hotdeck_imputed_data.RDS")



#### 2.1.3 Step 3: Evaluate Imputation Results ####

#### Generate imputation evaluation metrics table
# Load necessary libraries
library(Metrics)   # For MAE, RMSE, MSE
library(pROC)      # For AUC (if applicable)
library(ggplot2)   # For visualization
library(dplyr)     # For data manipulation
library(tidyr)     # For reshaping data

# Load imputed data
imputed_data <- readRDS("./.../imputed_data.RDS")

# Extract observed values (ground truth)
observed <- imputed_data$Original

# Remove observed column to compare different imputation methods
imputed_methods <- imputed_data %>% select(-Original)

# Function to compute evaluation metrics
compute_metrics <- function(imputed_values, observed_values) {
  # Ensure matched length
  valid_idx <- !is.na(observed_values) & !is.na(imputed_values)
  obs <- observed_values[valid_idx]
  imp <- imputed_values[valid_idx]
  
  # Compute metrics
  MAE <- mae(obs, imp)
  RMSE <- rmse(obs, imp)
  MSE <- mse(obs, imp)
  RB <- mean(imp - obs)  # Raw Bias
  PB <- 100 * abs(RB / mean(obs))  # Percent Bias
  
  # Compute confidence intervals across imputed estimates
  CI_low <- quantile(imp, probs = 0.025)  # 2.5% quantile
  CI_high <- quantile(imp, probs = 0.975) # 97.5% quantile
  
  # Coverage Rate (CR): Proportion of confidence intervals containing the true value
  CR <- mean(CI_low < obs & obs < CI_high)  # This now returns a proportion instead of 0/1
  
  # Average Width (AW): Width of the confidence interval
  AW <- mean(CI_high - CI_low)  # Now properly averaged across all imputations
  
  # Return as named vector
  return(c(MAE = MAE, RMSE = RMSE, MSE = MSE, RB = RB, PB = PB, CR = CR, AW = AW))
}

# Apply function to each imputation method
metrics_results <- imputed_methods %>% summarise(across(everything(), ~ compute_metrics(.x, observed), .names = "{.col}"))

# Reshape results into a formatted table
metrics_table <- as.data.frame(t(metrics_results))
colnames(metrics_table) <- c("MAE", "RMSE", "MSE", "RB", "PB (%)", "CR", "AW")
metrics_table <- round(metrics_table, 4)  # Round values for publication

# Print results
print(metrics_table)

# Save results as a CSV for publication
write.csv(metrics_table, "./..../Logistic_Imputation_Evaluation_Metrics.csv", row.names = FALSE)




#### 2.1.4 Visualize Imputation Results: Density Plot ####
library(ggplot2)
library(reshape2)


#### Step 1: Extract Observed and Imputed Values 

# Extract ground truth values (before pseudo-missing mask)
observed_values <- data[pseudo_missing_mask]  

# Extract imputed values for different methods
imputed_pca_values <- pca_imputed_data[pseudo_missing_mask]
imputed_soft_values <- soft_imputed_data[pseudo_missing_mask]
imputed_rf_values <- rf_imputed_data[pseudo_missing_mask]
imputed_hotdeck_values <- hotdeck_imputed_data[pseudo_missing_mask]


# Step 2: Prepare Imputed Data for Visualization 
# Combine observed values and imputed values into a data frame
imputed_data <- data.frame(
  Original = c(observed_values),
  PCA = c(imputed_pca_values),
  SoftImpute = c(imputed_soft_values),
  RandomForest = c(imputed_rf_values),
  HotDeck = c(imputed_hotdeck_values)
)
saveRDS(imputed_data, "./.../imputed_data.RDS")

imputed_data <- readRDS("./.../imputed_data.RDS")

# Convert to long format for ggplot
imputation_long <- melt(imputed_data, variable.name = "Method", value.name = "Value")

#### Step 3: Density Plot 
ggplot(imputation_long, aes(x = Value, fill = Method)) +
  geom_density(alpha = 0.3) +
  theme_minimal() +
  labs(title = "Density Comparison of Imputed Values", x = "Value", y = "Density")
## revise 1
pdf("./.../Density_Plot.pdf", width = 8, height = 6)
ggplot(imputation_long, aes(x = Value, fill = Method)) +
  geom_density(alpha = 0.4, color = NA) +  # Set alpha to 0.3 for transparency and remove curve outlines
  scale_fill_manual(values = c("PCA" = "red", 
                               "SoftImpute" = "green", 
                               "RandomForest" = "orange", 
                               "HotDeck" = "purple",
                               "Original" = "darkblue")) +  # Customize the colors for each method
  theme_minimal() +
  labs(title = "Density Comparison of Imputed Values", 
       x = "Value", 
       y = "Density") +
  scale_x_continuous(limits = c(-10, 10)) +  # Narrow the x-axis range to (-20, 20)
  theme(legend.title = element_blank(),  # Remove the legend title
        legend.position = "top",  # Position the legend at the top
        plot.title = element_text(hjust = 0.5),  # Center the title
        legend.text = element_text(size = 10))  # Adjust legend text size
dev.off()


#### Step 4: Violin plot

pdf("./.../Violin_Plot.pdf", width = 8, height = 6)
ggplot(imputation_long, aes(x = Method, y = log(Value), fill = Method)) +
  geom_violin(alpha = 0.5, color= NA) +
  theme_minimal() +
  labs(title = "Violin Plot Comparison of Imputation Methods", x = "Method", y = "Value")
dev.off()


# backup: Density Plot Comparison: Visualize distribution of original vs imputed data
ggplot(imputed_data, aes(x = Original)) +
  geom_density(color = "black", size = 1) +
  geom_density(aes(x = PCA), color = "green", linetype = "solid") +
  geom_density(aes(x = SoftImpute), color = "blue", linetype = "dashed") +
  geom_density(aes(x = RandomForest), color = "red", linetype = "dotted") +
  geom_density(aes(x = HotDeck), color = "purple", linetype = "dotdash") +
  labs(title = "Density Plot: Original vs. Imputed Values",
       x = "Values",
       y = "Density") +
  theme_minimal()

#### Step 5: Scatter Plot Comparison 

library(ggplot2)
library(ggpubr)

# Scatter plot for Observed vs PCA Imputation
scatter_data <- data.frame(
  Observed = observed_values,
  PCA = imputed_pca_values,
  SoftImpute = imputed_soft_values,
  RandomForest = imputed_rf_values,
  HotDeck = imputed_hotdeck_values
)

# Scatterplot: Observed vs PCA-Imputed Values
p1 <- ggplot(scatter_data, aes(x = Observed, y = PCA)) +
  geom_point(alpha = 0.5, color = "green") +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed") +
  labs(title = "Scatterplot: Observed vs. PCA-Imputed Values",
       x = "Observed Values",
       y = "PCA-Imputed Values") +
  theme_minimal()

# Scatterplot: Observed vs Soft-Imputed Values
p2 <- ggplot(scatter_data, aes(x = Observed, y = SoftImpute)) +
  geom_point(alpha = 0.5, color = "blue") +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed") +
  labs(title = "Scatterplot: Observed vs. Soft-Imputed Values",
       x = "Observed Values",
       y = "Soft-Imputed Values") +
  theme_minimal()

# Scatterplot: Observed vs RandomForest-Imputed Values
p3 <- ggplot(scatter_data, aes(x = Observed, y = RandomForest)) +
  geom_point(alpha = 0.5, color = "red") +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed") +
  labs(title = "Scatterplot: Observed vs. RandomForest-Imputed Values",
       x = "Observed Values",
       y = "RandomForest-Imputed Values") +
  theme_minimal()

# Scatterplot: Observed vs HotDeck-Imputed Values
p4 <- ggplot(scatter_data, aes(x = Observed, y = HotDeck)) +
  geom_point(alpha = 0.5, color = "purple") +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed") +
  labs(title = "Scatterplot: Observed vs. HotDeck-Imputed Values",
       x = "Observed Values",
       y = "HotDeck-Imputed Values") +
  theme_minimal()

# Arrange plots in a 2x2 grid
combined_plot <- ggarrange(p1, p2, p3, p4, 
                           ncol = 2, nrow = 2, 
                           labels = c("A", "B", "C", "D"))

# Save as PDF
ggsave("./.../Imputation_Scatterplots.pdf", combined_plot, width = 10, height = 8)





## 3 impute proteomics data using the optimal method (random forest) ####

# Set up parallel backend
num_cores <- detectCores() - 2  # Use all cores except one
cl <- makeCluster(num_cores)
registerDoParallel(cl)

# Run missForest in parallel
set.seed(123)
missforest_imputed <- missForest(data, 
                                 maxiter = 10, 
                                 ntree = 100, 
                                 parallelize = "variables")  # Use "forests" for tree-level parallelism

# Stop parallel backend (after finishing imputation)
stopCluster(cl)
rf_imputed_data <- missforest_imputed$ximp
rf_imputed_data <-   cbind(prot_data[,1:2], rf_imputed_data)

write.csv(rf_imputed_data, "./.../rf_imputed_logistic_data.csv")






## 4. imputation method evaluation of seveirty data set  ####

# check missing value
paste0(path_in, "severity_data_all_raw.csv") %>% read.csv(row.names = 1) %>% missing_values_rate()
# [1] 32.7209
# create pseudo missing value and compare methods, as above
# one-step imputation 
linear_data <- read.csv(paste0(path_in, "severity_data_all_raw.csv"), row.names = 1)


# Replace missing values with NA
data[data == ""] <- NA
pseudo_missing <- data  # Replace "data" with your actual dataset
pseudo_missing %>% dim()
# [1]   42 5018

####  Create masks based on  observed missingness pattern
set.seed(123)  

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

# Apply the masking separately to each experiment's columns
pseudo_missing <- mask_random_values(pseudo_missing, 1, 232)    # Masking for first experiment
pseudo_missing <- mask_random_values(pseudo_missing, 233, 5018) # Masking for second experiment
saveRDS(pseudo_missing, "./.../pseudo_missing_linear.RDS")


# imputation using different methods ##
# PCA
pca_imputed <- imputePCA(pseudo_missing, 2)  # 'ncp' is the number of components to retain
pca_imputed_data <- pca_imputed$completeObs
saveRDS(pca_imputed_data, "./.../pca_imputed_data_linear.RDS")

#soft impuation
set.seed(123)
lambda_opt <- softImpute::lambda0(pseudo_missing, maxit = 100)
soft_imputed <- softImpute::softImpute(pseudo_missing, rank.max = 41, lambda = lambda_opt, type = "svd")
soft_imputed_data <- softImpute::complete(pseudo_missing, soft_imputed)
saveRDS(soft_imputed_data, "./.../soft_imputed_data_linear.RDS")

# random forest 
rf_imputed_data <- missforest_imputed$ximp
saveRDS(rf_imputed_data, "./.../rf_imputed_data_linear.RDS")


# hot deck
# set.seed(123)
hotdeck_imputed <- hotdeck(pseudo_missing, ord_var = NULL)
hotdeck_imputed <- hotdeck_imputed[, colnames(pseudo_missing)]  ## remove extra columns created during imputation
hotdeck_imputed_data <- as.data.frame(hotdeck_imputed)
saveRDS(hotdeck_imputed_data, "./.../hotdeck_imputed_data_linear.RDS")



#### imputation evaluation metrics table ####
# Load necessary libraries
library(Metrics)   # For MAE, RMSE, MSE
library(pROC)      # For AUC (if applicable)
library(ggplot2)   # For visualization
library(dplyr)     # For data manipulation
library(tidyr)     # For reshaping data

# Load imputed data
imputed_data <- readRDS("./.../imputed_data_linear.RDS")

# Extract observed values (ground truth)
observed <- imputed_data$Original

# Remove observed column to compare different imputation methods
imputed_methods <- imputed_data %>% select(-Original)

# Function to compute evaluation metrics
compute_metrics <- function(imputed_values, observed_values) {
  # Ensure matched length
  valid_idx <- !is.na(observed_values) & !is.na(imputed_values)
  obs <- observed_values[valid_idx]
  imp <- imputed_values[valid_idx]
  
  # Compute metrics
  MAE <- mae(obs, imp)
  RMSE <- rmse(obs, imp)
  MSE <- mse(obs, imp)
  RB <- mean(imp - obs)  # Raw Bias
  PB <- 100 * abs(RB / mean(obs))  # Percent Bias
  
  # Compute confidence intervals across imputed estimates
  CI_low <- quantile(imp, probs = 0.025)  # 2.5% quantile
  CI_high <- quantile(imp, probs = 0.975) # 97.5% quantile
  
  # Coverage Rate (CR): Proportion of confidence intervals containing the true value
  CR <- mean(CI_low < obs & obs < CI_high)  # This now returns a proportion instead of 0/1
  
  # Average Width (AW): Width of the confidence interval
  AW <- mean(CI_high - CI_low)  # Now properly averaged across all imputations
  
  # Return as named vector
  return(c(MAE = MAE, RMSE = RMSE, MSE = MSE, RB = RB, PB = PB, CR = CR, AW = AW))
}

metrics_results <- imputed_methods %>% summarise(across(everything(), ~ compute_metrics(.x, observed), .names = "{.col}"))

# Reshape results into a formatted table
metrics_table <- as.data.frame(t(metrics_results))
colnames(metrics_table) <- c("MAE", "RMSE", "MSE", "RB", "PB (%)", "CR", "AW")
metrics_table <- round(metrics_table, 4)  # Round values for publication

print(metrics_table)

write.csv(metrics_table, "./.../Linear_Imputation_Evaluation_Metrics.csv", row.names = FALSE)


### visulization
# Extract ground truth values (before pseudo-missing mask)
observed_values <- data[pseudo_missing_mask]  

# Extract imputed values for different methods
imputed_pca_values <- pca_imputed_data[pseudo_missing_mask]
imputed_soft_values <- soft_imputed_data[pseudo_missing_mask]
imputed_rf_values <- rf_imputed_data[pseudo_missing_mask]
imputed_hotdeck_values <- hotdeck_imputed_data[pseudo_missing_mask]


# Prepare Imputed Data for Visualization 
# Combine observed values and imputed values into a data frame
imputed_data <- data.frame(
  Original = c(observed_values),
  PCA = c(imputed_pca_values),
  SoftImpute = c(imputed_soft_values),
  RandomForest = c(imputed_rf_values),
  HotDeck = c(imputed_hotdeck_values)
)
saveRDS(imputed_data, "./.../imputed_data_linear.RDS")

# Convert to long format for ggplot
imputation_long <- melt(imputed_data, variable.name = "Method", value.name = "Value")

## revise 1
pdf("./.../Density_Plot_linear.pdf", width = 8, height = 6)
ggplot(imputation_long, aes(x = Value, fill = Method)) +
  geom_density(alpha = 0.4, color = NA) +  # Set alpha to 0.3 for transparency and remove curve outlines
  scale_fill_manual(values = c("PCA" = "red", 
                               "SoftImpute" = "green", 
                               "RandomForest" = "orange", 
                               "HotDeck" = "purple",
                               "Original" = "darkblue")) +  # Customize the colors for each method
  theme_minimal() +
  labs(title = "Density Comparison of Imputed Values", 
       x = "Value", 
       y = "Density") +
  scale_x_continuous(limits = c(-10, 10)) +  # Narrow the x-axis range to (-20, 20)
  theme(legend.title = element_blank(),  # Remove the legend title
        legend.position = "top",  # Position the legend at the top
        plot.title = element_text(hjust = 0.5),  # Center the title
        legend.text = element_text(size = 10))  # Adjust legend text size
dev.off()


### scatter plot
library(ggplot2)
library(ggpubr)

# Scatter plot for Observed vs PCA Imputation
scatter_data <- data.frame(
  Observed = observed_values,
  PCA = imputed_pca_values,
  SoftImpute = imputed_soft_values,
  RandomForest = imputed_rf_values,
  HotDeck = imputed_hotdeck_values
)

# Scatterplot: Observed vs PCA-Imputed Values
p1 <- ggplot(scatter_data, aes(x = Observed, y = PCA)) +
  geom_point(alpha = 0.5, color = "green") +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed") +
  labs(title = "Scatterplot: Observed vs. PCA-Imputed Values",
       x = "Observed Values",
       y = "PCA-Imputed Values") +
  theme_minimal()

# Scatterplot: Observed vs Soft-Imputed Values
p2 <- ggplot(scatter_data, aes(x = Observed, y = SoftImpute)) +
  geom_point(alpha = 0.5, color = "blue") +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed") +
  labs(title = "Scatterplot: Observed vs. Soft-Imputed Values",
       x = "Observed Values",
       y = "Soft-Imputed Values") +
  theme_minimal()

# Scatterplot: Observed vs RandomForest-Imputed Values
p3 <- ggplot(scatter_data, aes(x = Observed, y = RandomForest)) +
  geom_point(alpha = 0.5, color = "red") +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed") +
  labs(title = "Scatterplot: Observed vs. RandomForest-Imputed Values",
       x = "Observed Values",
       y = "RandomForest-Imputed Values") +
  theme_minimal()

# Scatterplot: Observed vs HotDeck-Imputed Values
p4 <- ggplot(scatter_data, aes(x = Observed, y = HotDeck)) +
  geom_point(alpha = 0.5, color = "purple") +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed") +
  labs(title = "Scatterplot: Observed vs. HotDeck-Imputed Values",
       x = "Observed Values",
       y = "HotDeck-Imputed Values") +
  theme_minimal()

# Arrange plots in a 2x2 grid
combined_plot <- ggarrange(p1, p2, p3, p4, 
                           ncol = 2, nrow = 2, 
                           labels = c("A", "B", "C", "D"))

# Save as PDF
ggsave("./.../Imputation_Scatterplots_linear.pdf", combined_plot, width = 10, height = 8)


####  Violin plot

pdf("./.../Violin_Plot_linear.pdf", width = 8, height = 6)
ggplot(imputation_long, aes(x = Method, y = log(Value), fill = Method)) +
  geom_violin(alpha = 0.5, color= NA) +
  theme_minimal() +
  labs(title = "Violin Plot Comparison of Imputation Methods", x = "Method", y = "Value")
dev.off()





## 5 impute linear data using optimal method (PCA-based)####

# PCA
nb <- estim_ncpPCA(data, method.cv = "Kfold", verbose = FALSE) 
nb$ncp
pca_imputed <- imputePCA(data, 2)  # 'ncp' is the number of components to retain
pca_imputed_data <- pca_imputed$completeObs
pca_imputed_data <- as.data.frame(pca_imputed_data)
pca_imputed_data <- cbind(linear_data[,1], pca_imputed_data)
write.csv(pca_imputed_data, "./.../PCA_imputed_Linear_data.csv", row.names = FALSE)



