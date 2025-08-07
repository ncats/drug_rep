#### dot-line plot of normalized scores
# Load necessary libraries
library(readr)
library(dplyr)
library(ggplot2)
library(reshape2)
library(scales)
library(tidyverse)

# Load the data
df <- read.csv('./.../all_proteins_five_centroality_scores.csv')

# Initialize MinMaxScaler and normalize the scores
df_norm  <- df %>%
  mutate(
    DC_Score_norm = rescale(DC_Score),
    BC_Score_norm = rescale(BC_Score),
    CC_Score_norm = rescale(CC_Score),
    MCC_Score_norm = rescale(MCC_Score),
    DMNC_Score_norm = rescale(DMNC_Score)
  )



# Create plot_data by explicitly selecting and renaming columns from df_norm

plot_data <- bind_rows(
  df_norm[1:10, ] %>%
    transmute(Rank = 1:10, Gene = DC_gene, Score = DC_Score_norm, Metric = "DC"),
  df_norm[1:10, ] %>%
    transmute(Rank = 1:10, Gene = BC_gene, Score = BC_Score_norm, Metric = "BC"),
  df_norm[1:10, ] %>%
    transmute(Rank = 1:10, Gene = CC_gene, Score = CC_Score_norm, Metric = "CC"),
  df_norm[1:10, ] %>%
    transmute(Rank = 1:10, Gene = MCC_gene, Score = MCC_Score_norm, Metric = "MCC"),
  df_norm[1:10, ] %>%
    transmute(Rank = 1:10, Gene = DMNC_gene, Score = DMNC_Score_norm, Metric = "DMNC")
)


# Plot the data
# Set color palette for different metrics
metric_colors <- c("DC" = "blue", "BC" = "green", "CC" = "red", "MCC" = "purple", "DMNC" = "orange")

# Define Python-style color palette
python_colors <- c(
  "DC" = "#1f77b4",   # blue
  "BC" = "#ff7f0e",   # orange
  "CC" = "#2ca02c",   # green
  "MCC" = "#d62728",  # red
  "DMNC" = "#9467bd"  # purple
)

# Create the plot
pdf('./.../PPI_high_influence_nodes.pdf', width = 9, height = 6)
ggplot(plot_data, aes(x = Rank, y = Score, group = Metric, color = Metric)) +
  geom_line() +
  geom_point(size = 3) +
  geom_text(aes(label = Gene), vjust = -0.8, size = 3) +
  scale_x_continuous(breaks = 1:10) +
  scale_color_manual(values = python_colors) +
  labs(
    title = "Top 10 Genes Ranked by Different Centrality Scores",
    x = "Rank (1 = Top Gene)",
    y = "Normalized Centrality Score"
  ) +
  theme_minimal()

dev.off()



#### re-rank based on five scores 

# 1. pivot gene columns to long format
gene_long <- df %>%
  select(ends_with("_gene")) %>%
  mutate(RowID = row_number()) %>%
  pivot_longer(
    cols = -RowID,
    names_to = "Metric",
    values_to = "Gene"
  ) %>%
  mutate(Metric = sub("_gene$", "", Metric))

# 2. Pivot score columns to long format
score_long <- df %>%
  select(ends_with("_Score")) %>%
  mutate(RowID = row_number()) %>%
  pivot_longer(
    cols = -RowID,
    names_to = "Metric",
    values_to = "Score"
  ) %>%
  mutate(Metric = sub("_Score$", "", Metric))

# 3. Join gene and score tables
long_df <- left_join(gene_long, score_long, by = c("RowID", "Metric")) %>%
  filter(!is.na(Gene) & !is.na(Score))


# 4. Normalize scores within each metric
long_df <- long_df %>%
  group_by(Metric) %>%
  mutate(Norm_Score = rescale(as.numeric(Score))) %>%
  ungroup()

# 5. Average normalized score per gene
overall_scores <- long_df %>%
  group_by(Gene) %>%
  summarize(Overall_Centrality = mean(Norm_Score, na.rm = TRUE)) %>%
  arrange(desc(Overall_Centrality))

# 6. Top 20 hub genes
top20_genes <- overall_scores %>%
  slice_head(n = 20)

# Output result
top20_genes


## merge back and save

# --- Step 1: Normalize and Rank within each Metric ---
long_df_ranked <- long_df %>%
  group_by(Metric) %>%
  mutate(
    Norm_Score = rescale(as.numeric(Score)),
    Rank = rank(-Norm_Score, ties.method = "min")  # Rank high scores as top
  ) %>%
  ungroup()

# --- Step 2: Calculate Overall Centrality ---
overall_scores <- long_df_ranked %>%
  group_by(Gene) %>%
  summarize(
    Overall_Centrality = mean(Norm_Score, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(desc(Overall_Centrality))

# --- Step 3: Join Individual Ranks and Scores back ---
# Convert wide so that each gene has one row, and columns for each metric's Score and Rank
per_metric_data <- long_df_ranked %>%
  select(Gene, Metric, Score, Rank) %>%
  pivot_wider(
    names_from = Metric,
    values_from = c(Score, Rank),
    names_sep = "_"
  )

# --- Step 4: Combine with Overall Score and Top 20 ---
final_table <- overall_scores %>%
  left_join(per_metric_data, by = "Gene") %>%
  arrange(desc(Overall_Centrality))

# Optional: Select top 20 for saving
top20_final <- final_table %>% slice_head(n = 20)

# View or save the top 20
print(top20_final)

# Save to CSV if needed
write.csv(final_table, "./.../all_protines_ranked_by_overall_metrics.csv", row.names = FALSE)
