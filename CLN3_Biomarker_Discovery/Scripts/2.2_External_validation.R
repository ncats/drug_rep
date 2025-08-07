rm(list = ls())
# library(magrittr)
# library(glmnet)
# library(readxl)
library(dplyr)

##  GSE22225 re-analysis ####
if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
BiocManager::install(c("affy", "hgu133plus2.db", "AnnotationDbi"))

library(dplyr)
library(affy)
library(hgu133plus2.db)
library(AnnotationDbi)

#### 1 Read and Normalize CEL Files ####
# Point to the directory with your CEL files
cel_dir <- "./.../GSE22225_CEL/"
cel_files <- list.files(cel_dir, pattern = "\\.CEL$", full.names = TRUE)

# Read CEL files (this will take a minute or two)
rawData <- ReadAffy(filenames = cel_files)

# Normalize using RMA (robust multi-array average)
eset <- rma(rawData)
exprs_mat <- exprs(eset) # rows = probe sets, columns = samples
colnames(exprs_mat) <- gsub("\\.CEL$", "", basename(colnames(exprs_mat)))


#### 2 Map Probe Sets to Gene Symbols ####
# Map probe set IDs to gene symbols
probe2gene <- AnnotationDbi::select(
  hgu133plus2.db,
  keys = rownames(exprs_mat),
  columns = "SYMBOL",
  keytype = "PROBEID"
)

# Join expression data with gene symbols
exprs_df <- as.data.frame(exprs_mat)
exprs_df$PROBEID <- rownames(exprs_df)
exprs_df <- merge(exprs_df, probe2gene, by = "PROBEID")

# Some probe sets map to multiple genes, keep only those with SYMBOL
exprs_df <- exprs_df[!is.na(exprs_df$SYMBOL), ]

#### 3 Collapse to Gene-Level Expression (Average Across Probes) ####
# Aggregate by gene symbol (mean expression across all probes per gene)
exprs_gene <- aggregate(
  . ~ SYMBOL,
  data = exprs_df[ , c("SYMBOL", colnames(exprs_mat))],
  FUN = mean
)

rownames(exprs_gene) <- exprs_gene$SYMBOL
exprs_gene <- exprs_gene[ , -1] # Remove SYMBOL column

write.csv(exprs_gene, "./.../GSE22225_allgene_expr.csv", row.names=TRUE)

#top 20 protein biomarker candidates
biomarkers <- c('EGFR', 'HIF1A', 'CXCL12', 'CSF1', 'VEGFC', 'COL4A3', 'VEGFD', 'ERBB4', 'OSM', 
                'NPM1', 'CD74', 'BRD4', 'TFRC', 'STAG2', 'LMNB1', 'MYH9', 'IL6R', 'COLGALT2', 'COL6A1', 'CTCF') # 

expr_20 <- exprs_gene[rownames(exprs_gene) %in% biomarkers, ]

#check
setdiff(biomarkers, rownames(expr_20))

write.csv(expr_20, "./.../GSE22225_biomarker_expr.csv", row.names=TRUE)


# Read sample info
sample_info <- read.csv("./.../GSE22225_sample_info.csv")
expr_20_df <- data.frame(Gene=rownames(expr_20), expr_20, check.names=FALSE)
expr_20_long <- expr_20_df %>%
  pivot_longer(-Gene, names_to="Sample", values_to="Expression")
expr_20_long$Group <- sample_info$Group[match(expr_20_long$Sample, sample_info$SampleName)]


### plot
library(tidyr)
library(dplyr)
library(ggplot2)

expr_20_long$Group <- factor(expr_20_long$Group, levels = c("Control", "Slow", "Average", "Rapid"))
# Calculate median for each Gene x Group
medians <- expr_20_long %>%
  group_by(Gene, Group) %>%
  summarize(Median = median(Expression, na.rm=TRUE), .groups = "drop")


library(ggplot2)
pdf("./../GSE22225_biomarker_expr.pdf", height = 12, width = 14 )
ggplot(expr_20_long, aes(x=Group, y=Expression, fill=Group)) +
  geom_violin(trim=FALSE, alpha=0.5) +
  geom_boxplot(width=0.1, outlier.shape=NA, alpha=0.6) +
  geom_jitter(width=0.15, size=1, alpha=0.8) +
  # Median trend line per gene:
  geom_line(data=medians, aes(x=Group, y=Median, group=Gene), color="black", size=1.2) +
  geom_point(data=medians, aes(x=Group, y=Median), color="black", size=2) +
  facet_wrap(~Gene, scales="free_y") +
  theme_bw(base_size=14) +
  labs(title="Expression of 20 Biomarkers by Progression Group",
       y="Expression", x="Group") +
  theme(axis.text.x=element_text(angle=30, hjust=1))
dev.off()


# box plot 
ggplot(expr_20_long, aes(x=Group, y=Expression, fill=Group)) +
  geom_boxplot(outlier.shape=NA, alpha=0.6) +
  geom_jitter(width=0.15, size=1, alpha=0.7) +
  facet_wrap(~Gene, scales="free_y") +
  theme_bw(base_size=14) +
  labs(title="Expression of 20 Biomarkers by Progression Group",
       y="Expression", x="Group")

### regroup to con3 vs. control
expr_20_long$Group2 <- ifelse(expr_20_long$Group == "Control", "Control", "CLN3")
expr_20_long$Group2 <- factor(expr_20_long$Group2, levels = c("Control", "CLN3"))

pdf("./.../GSE22225_biomarker_expr_merged.pdf", height = 12, width = 14 )
ggplot(expr_20_long, aes(x=Group2, y=Expression, fill=Group2)) +
  geom_violin(trim=FALSE, alpha=0.5) +
  geom_boxplot(width=0.1, outlier.shape=NA, alpha=0.6) +
  geom_jitter(width=0.15, size=1, alpha=0.8) +
  facet_wrap(~Gene, scales="free_y") +
  theme_bw(base_size=14) +
  labs(title="Expression: Control vs. CLN3",
       y="Expression", x="Group") +
  theme(axis.text.x=element_text(angle=30, hjust=1))
dev.off()



#### 3 AUROC  ####
library(pROC)

# Prepare the data

sample_info$Group2 <- ifelse(sample_info$Group == "Control", "Control", "CLN3")
sample_info$Group2 <- factor(sample_info$Group2, levels = c("Control", "CLN3"))


expr_20_t <- as.data.frame(t(expr_20))
expr_20_t$Sample <- rownames(expr_20_t)
df_roc <- merge(expr_20_t, sample_info, by.x="Sample", by.y="SampleName")
df_roc <- df_roc[df_roc$Group2 %in% c("Control", "CLN3"), ]  # or whatever your two groups are
df_roc$Group_bin <- ifelse(df_roc$Group2 == "CLN3", 1, 0)

# Calculate and plot ROC for each biomarker gene
auc_results <- data.frame(Gene=character(), AUC=numeric())

pdf("./.../GSE22225_biomarker_all_ROC.pdf", height = 12, width = 14 )
par(mfrow=c(4,5)) # if you have 20 genes, adjust as needed
for (gene in rownames(expr_20)) {
  roc_obj <- roc(df_roc$Group_bin, df_roc[[gene]], quiet=TRUE)
  auc_val <- auc(roc_obj)
  auc_results <- rbind(auc_results, data.frame(Gene=gene, AUC=auc_val))
  plot(roc_obj, main=paste0(gene, " (AUC = ", round(auc_val,3), ")"))
}
dev.off()

par(mfrow=c(1,1))

write.csv(auc_results, "./.../GSE22225_biomarker_ROC.csv", row.names = FALSE)


# Mean of top 3 genes by AUROC
auc_results <- auc_results[1:20,] 
top_genes <- auc_results %>% arrange(desc(AUC)) %>% pull(Gene) %>% head(6)
 
pdf("./.../GSE22225_biomarker_to6_ROC.pdf", height = 6, width = 9)
par(mfrow=c(2,3)) # if you have 20 genes, adjust as needed
for (gene in top_genes) {
  roc_obj <- roc(df_roc$Group_bin, df_roc[[gene]], quiet=TRUE)
  auc_val <- auc(roc_obj)
  auc_results <- rbind(auc_results, data.frame(Gene=gene, AUC=auc_val))
  plot(roc_obj, main=paste0(gene, " (AUC = ", round(auc_val,3), ")"))
}

dev.off()

### express of top 6 genes 

## 1. CLN3 vs. control

violin_data <- expr_20_long[expr_20_long$Gene %in% top_genes, ]

# p1 <- ggplot(violin_data, aes(x=Group, y=Expression, fill=Group)) +
#   geom_violin(trim=FALSE, alpha=0.6) +
#   geom_boxplot(width=0.1, outlier.shape=NA, alpha=0.6) +
#   geom_jitter(width=0.15, size=1, alpha=0.8) +
#   facet_wrap(~Gene, scales="free_y", nrow=2) +
#   theme_bw(base_size=14) +
#   labs(title="Top 6 Genes by Group", y="Expression", x="Group") +
#   theme(axis.text.x=element_text(angle=30, hjust=1), legend.position="none")

medians <- violin_data %>%
  group_by(Gene, Group) %>%
  summarize(Median = median(Expression, na.rm=TRUE), .groups = "drop")

pdf("./.../GSE22225_biomarker_to6_expr.pdf", height = 6, width = 9)
ggplot(violin_data, aes(x=Group, y=Expression, fill=Group)) +
  geom_violin(trim=FALSE, alpha=0.5) +
  geom_boxplot(width=0.1, outlier.shape=NA, alpha=0.6) +
  geom_jitter(width=0.15, size=1, alpha=0.8) +
  # Median trend line per gene:
  geom_line(data=medians, aes(x=Group, y=Median, group=Gene), color="black", size=1.2) +
  geom_point(data=medians, aes(x=Group, y=Median), color="black", size=2) +
  facet_wrap(~Gene, scales="free_y") +
  theme_bw(base_size=14) +
  labs(title="Expression of Top6 Biomarkers by Progression Group",
       y="Expression", x="Group") +
  theme(axis.text.x=element_text(angle=30, hjust=1))
dev.off()


## 2. cross different progresion rates
pdf("./.../GSE22225_biomarker_top6_expr_merged.pdf", height = 6, width = 9 )
ggplot(violin_data, aes(x=Group2, y=Expression, fill=Group2)) +
  geom_violin(trim=FALSE, alpha=0.5) +
  geom_boxplot(width=0.1, outlier.shape=NA, alpha=0.6) +
  geom_jitter(width=0.15, size=1, alpha=0.8) +
  facet_wrap(~Gene, scales="free_y") +
  theme_bw(base_size=14) +
  labs(title="Expression of Top6 Biomarkers: Control vs. CLN3",
       y="Expression", x="Group") +
  theme(axis.text.x=element_text(angle=30, hjust=1))
dev.off()

