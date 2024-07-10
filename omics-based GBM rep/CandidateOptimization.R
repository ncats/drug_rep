# calculation of Regulation score (RS) and overall coverage (OC)
  

### 0.  environment loading  #### 

rm(list = ls())

library(dplyr)       
library(pheatmap)
library(ggplot2)     
library(tidyr)
library(magrittr)
library(dplyr)
library(stats)

pth <- "~./"

### 1. data preparation #### 
###### 1.1 read data ####

dt <- read.csv(paste(pth, "merged_sig_all_new.csv", sep = ""), header = T, row.names = 1)

drug.list <- read.csv(paste(pth, "Signature_clustering_new.csv", sep = ""), header = T, row.names = )


# check distribution of signature clusters

clu.list <- drug.list[which(drug.list$Cluster == 1 | drug.list$Cluster == 2 | drug.list$Cluster == 3 | drug.list$Cluster == 5 | drug.list$Cluster == 8 | drug.list$Cluster == 7),] 

d.list <- clu.list$Drug %>% unique() %>% length()
# [1] 39
## note: only removed cluster 4, in which most are missing values .. (less than 20 genes)


###### 1.2 gene's median fold change in each cluster ####
df <- data.frame(matrix(ncol = 40, nrow = 318))
colnames(df) <- d.list
rownames(df) <- dt$GeneSymbol


for (i in 1:length(d.list)) {
  
  
  id.list <- clu.list$Signature[which(clu.list$Drug == d.list[i])]
  
  dt.t <- dt[,id.list, drop = FALSE]
  
  df[,d.list[i]] <-  apply(dt.t, 1, median, na.rm=T) 
}


write.csv(df, file = paste(pth, "Medians_FC_perDrug.csv", sep = ""))



###### 1.3 calculation of regulated fold change #### 
df <- read.csv(file = paste(pth, "Medians_FC_perDrug.csv", sep = ""), header = T, row.names = 1)

df <- df[order(df$GBM, decreasing=FALSE),]


for (i in 1:nrow(df)) {
  
  if (df$GBM[i] > 0) {
    
    df[i,2:40] <- -df[i,2:40]
    
  }
  
}

df  %>% write.csv(paste(pth, "regultedGenes_perDrug.csv", sep = ""))



### 2. regulation score calculation ####


##### 2.1 define function #### 

regulation_score <- function(df_drug, df_disease, score ){
  

  sumd <- sum(abs(df_disease))
  
    # drop rows if (df_drug*df_disease) > 0
    c <- cbind("df_drug"=df_drug, "df_disease"=df_disease) %>% as.data.frame()
    c <- c[which((c$df_drug*c$df_disease)<0),]
    rs <- (abs(c$df_drug-c$df_disease)*abs(c$df_disease)/sumd) %>% sum() 
    return(rs)
  
}


##### 2.2  read data  ####
paste(wd, "Medians_FC_perDrug.csv", sep = "") %>%
  read.csv(header = T, row.names = 1) %>% arrange(desc(GBM)) -> df


df[is.na(df)] <- 0 # needed for combination calculation
df_drug <- df$Dasatinib
df_disease <- df$GBM

###### 2.3  calculating RS for all candidates   ####

d.list <- df[,2:40] %>% colnames()

res <- data.frame(matrix(ncol = 3, nrow = 0))
colnames(res) <- c("DrugA", "DrugB", "regulation_score")


for (i in 1:39) {
  for (j in 1:39) {
    
    if (i != j ) {
      
      s <- df[,d.list[i]]+ df[,d.list[j]]
      rs <- regulation_score(s, df$GBM, score = "rs") %>% round(3)
    }
  
    res[nrow(res)+1,] <- c(d.list[i], d.list[j], rs)
    
  }
  
}

write.csv(res, paste(wd,"regulation_score.csv", sep = "" ))


### 3. result overview and analysis ####
# read data # 
scores_path <- "~/candidateEvaluation.csv"
cb <- scores_path %>% read.csv(header = T, row.names = 1) 
## path to drug gene expression median LFCs
lfc_path <- "~./Medians_FC_perDrug.csv"
df <- read.csv(lfc_path, header = T, row.names = 1)


##### 3.1  get the list of top combinations  ####

d.list <- scores_path %>% read.csv(header = T, row.names = 1) %>% filter(difference_rate > 0)  

# GET the ranked list

s.list <- scores_path %>% read.csv(header = T, row.names = 1) %>% filter(difference_rate == 0)
s.list <- s.list[order(s.list$regulation_score, s.list$overall_coverage, decreasing = TRUE),]

s.list %>% head()


###### 3.2 bar plot of top 5 candidates  ####

pdf( "barplot_top.single.new.pdf",  width = 8, height = 8)
# png("barplot_top.single.new.png", width = 8, height = 8, units = "in", res = 300 )
par(mfrow=c(3,2))
for (i in 1:6) {
  
  a <- s.list$DrugA[i]
  
  df[,a] %>% as.matrix() %>% t() %>% barplot(col="blue", main = a, border = NA, ylim = c(-6,6))
  lines(df$GBM, col="red",lty="dashed", lwd=2)
  text(150,-4,paste("RS = ", round(s.list$regulation_score[i],2), ", OC = ", round(s.list$overall_coverage[i],2), sep = ""), pos=3)
  
}

dev.off()


###### 3.3 bar plot of all candidates  ####


pdf( "barplot_ALL.single.new.pdf",  width = 8, height = 8)
# png("barplot_ALL.single.png", width = 16, height = 6, units = "in", res = 300 )
par(mfrow=c(3,2))
for (i in 1:nrow(s.list)) {
  
  a <- s.list$DrugA[i]
  
  df[,a] %>% as.matrix() %>% t() %>% barplot(col="blue", main = a, border = NA, ylim = c(-6,6))
  lines(df$GBM, col="red",lty="dashed", lwd=2)
  text(150,-4,paste("RS = ", round(d.list$regulation_score[i],2), ", OC = ", round(d.list$overall_coverage[i],2), sep = ""), pos=3)
  
}

dev.off()



