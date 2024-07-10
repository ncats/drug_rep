
### 0. R environment loading #### 
rm(list = ls())
library(pheatmap)
library(ggplot2)
library(ggpubr)
library(PCAtools)
library(ComplexHeatmap)
library(tidyr)
library(dplyr)

### 1. clustering of candidates' gene expression signatures comparing to GGEP  ####
"~/merged_sig_all_new.csv" %>% read.csv() -> df

######  1.1 attempt a: pheatmap package ####
rownames(df) <- df$GeneSymbol
df <- df[,-1]
df1 <- df[order(df$GBM, decreasing=TRUE),]


pdf("~/Heatmap_merged_sig_all.pdf", width = 30, height = 18)
png(file = "~/Heatmap_merged_sig_all.png", width = 30, height = 18, units = "in", res = 300)
pheatmap(df1,  
         cluster_row = FALSE,
         # cluster_cols = FALSE
         cutree_cols = 20
         
)

dev.off()



######  1.2 attempt b: Complexheatmap package ####
## draw heatmaps comparing  all distance+clustering method combinations
cluster.dist <- c("euclidean", "maximum", "manhattan", "canberra", "binary", "minkowski", "pearson", "spearman", "kendall")  # distance methods
cluster.meth <- c("ward.D", "ward.D2", "single", "complete", "average" , "mcquitty" , "median" ,  "centroid")  # clustering methods
pdf("~/CommplexHeatmap_compare.pdf", width = 32, height = 18)

for (i in 1:length(cluster.dist)) {
  d <- cluster.dist[i]
  for (j in 1:length(cluster.meth)) {
    m <- cluster.meth[j]
    
    t <- paste(d,m, sep = "_")
    
    ht = ComplexHeatmap::Heatmap(df1,
                                 cluster_rows = FALSE,
                                 clustering_distance_columns = d,
                                 clustering_method_columns = m,
                                 # column_split = 20
                                 column_title = t
    )
    
    draw(ht)
  }
  
}
dev.off()


#### result:  the comparison showed that the method combo of minkowski + ward.D had the expected clustering.
## using km to clustering required number of groups


# df2 <- na.omit(df1)
# replot for saving file
pdf("~/CommplexHeatmap_clustered_new.pdf", width = 32, height = 18)
png(file = "~/CommplexHeatmap_clustered_new.png", width = 30, height = 18, units = "in", res = 300)
ComplexHeatmap::Heatmap(df1,
                        cluster_rows = FALSE,
                        clustering_distance_columns = "minkowski",
                        clustering_method_columns = "ward.D",
                        column_split = 8
                        # column_title = t
                        # row_split = 5,
                        # column_km = 7,
                        # column_km_repeats = 20
)


dev.off()


ht <- ComplexHeatmap::Heatmap(df1,
                              cluster_rows = FALSE,
                              clustering_distance_columns = "minkowski",
                              clustering_method_columns = "ward.D",
                              column_split = 8
                              # column_title = t
                              # row_split = 5,
                              # column_km = 7,
                              # column_km_repeats = 20
)




###### 1.3 extract clustered group number for each signature/candidate ####

df.c <- data.frame(matrix(ncol = 2, nrow = 351))
colnames(df.c) <- c("Drug", "Cluster")
df.c$Drug <- colnames(df1)


for (i in 1:8) {
  cl <- column_order(ht)[[i]]
  for (j in 1:length(cl)) {
    k <- cl[j]
    
    df.c[k,2] <- i
    
    
  }
  
}

write.csv(df.c, file = "~/Signature_clustering_new.csv")



###### 1.4 get averaged log foldchange for each cluster and plot (cluster numver 1,3,8) ####

sig.list <-  read.csv("~/Signature_clustering_new.csv", header = T, row.names = 1) %>% 
  filter(Cluster == 8 | Cluster == 1 | Cluster == 3 | Cluster == 7) %>%
  select(Drug) 


## get the signatures (foldchanges) for specified clusters 
df <- data.frame(matrix(ncol = 4, nrow = 318))
colnames(df) <- c("GBM", "Cluster.1", "Cluster.3", "Cluster.8")

df$GeneSymbol <- read.csv("~/merged_sig_all_new.csv", header = T, row.names  = 1) %>% select(GeneSymbol) 
rownames(df) <- df$GeneSymbol[,]
df <- df[,-5]
cluster.list <- c(7,1,3,8)

for (i in 1:length(cluster.list)) {
  
  sig.list <-  read.csv("~/Signature_clustering_new.csv", header = T, row.names = 1) %>% filter(Cluster ==  cluster.list[i]) %>% select(Drug) 
  
  
  df[,i] <- read.csv("~/merged_sig_all_new.csv", header = T, row.names  = 1) %>% select(sig.list$Drug) %>% rowMeans(na.rm = T)
  
  
}


write.csv(df,"~/SignatureCluster_average.csv")  ## save the averaged fold changes of each cluster 



###### 1.5  scatter plots comparing GGEP and signatures  ####
# "~/SignatureCluster_average.csv" %>% read.csv(header = T) -> df 
plot(df$GBM)
df$GBM %>% sort(decreasing = FALSE) %>% plot()
df <- df[order(df$GBM, decreasing=FALSE),]
df$rank <- 1:nrow(df)

### plot
df$GBM %>% plot(pch=20, col="red", cex=0.8)
# loess(df$GBM ~ df$rank) %>% predict() %>% lines(col="red", lwd = 3)
smooth.spline(df$GBM ~ df$rank) %>% lines(col="red", lwd = 3)

df$Cluster.8 %>% points(pch=20, col="green", cex=0.8)
# loess(df$Cluster.8 ~ df$rank, na.action = na.exclude) %>% predict() %>% lines(col="green", lwd = 3)
lm(df$Cluster.8 ~ df$rank, na.action = na.exclude) %>% predict() %>% lines(col="green", lwd = 3)
# with(df[!is.na(df$Cluster.8),], smooth.spline(rank ,Cluster.8)) %>% lines(col="green", lwd = 3)

df$Cluster.1 %>% points(pch=20, col="blue", cex=0.8)
# loess(df$Cluster.1 ~ df$rank , na.action = na.exclude) %>% predict() %>% lines(col="blue", lwd = 3)
lm(df$Cluster.1 ~ df$rank , na.action = na.exclude) %>% predict() %>% lines(col="blue", lwd = 3)

# with(df[!is.na(df$Cluster.1),],smooth.spline(rank ,Cluster.1)) %>% lines(col="blue", lwd = 3)

df$Cluster.3 %>% points(pch=20, cex=0.8)
# loess(df$Cluster.3 ~ df$rank, na.action = na.exclude) %>% predict() %>% lines(col="darkviolet", lwd = 3)
lm(df$Cluster.3 ~ df$rank, na.action = na.exclude) %>% predict() %>% lines( lwd = 3)
# with(df[!is.na(df$Cluster.3),], smooth.spline(rank ,Cluster.3)) %>% lines(col="darkviolet", lwd = 3)
text(x=rep(max(x)+3, 2), y=c(mean(y), mean(y2)), pos=4, labels=c('black line', 'red line'))
### plot end

###### 1.6  alternative: compare cluster in separate plots ####

pdf("~/iLINCS/scattor_clusters.pdf", width = 16, height = 6)
png(file = "~iLINCS/scattor_clusters.png", width = 16, height = 6, units = "in", res = 300)
png(file = "~/scattor_clusters_1.png", width = 6, height = 16, units = "in", res = 300)
par(mfrow = c(1,3))

#plot cluster 1
df$GBM %>% plot(pch=20, col="red", cex=1, main="Cluster 1")
# loess(df$GBM ~ df$rank) %>% predict() %>% lines(col="red", lwd = 3)
smooth.spline(df$GBM ~ df$rank) %>% lines(col="red", lwd = 3)
df$Cluster.1 %>% points(pch=20, col="blue", cex=1)
# loess(df$Cluster.1 ~ df$rank , na.action = na.exclude) %>% predict() %>% lines(col="blue", lwd = 3)
lm(df$Cluster.1 ~ df$rank , na.action = na.exclude) %>% predict() %>% lines(col="blue", lwd = 3)
abline(h=0,col="gray50", lwd=2)


#plot cluster 3
df$GBM %>% plot(pch=20, col="red", cex=1, main="Cluster 3")
# loess(df$GBM ~ df$rank) %>% predict() %>% lines(col="red", lwd = 3)
smooth.spline(df$GBM ~ df$rank) %>% lines(col="red", lwd = 3)
df$Cluster.3 %>% points(pch=20, cex=1, col="darkviolet")
# loess(df$Cluster.3 ~ df$rank, na.action = na.exclude) %>% predict() %>% lines(col="darkviolet", lwd = 3)
lm(df$Cluster.3 ~ df$rank, na.action = na.exclude) %>% predict() %>% lines(col="darkviolet", lwd = 3)
abline(h=0,col="gray50", lwd=2)

#plot cluster 8
df$GBM %>% plot(pch=20, col="red", cex=1, main="Cluster 8")
# loess(df$GBM ~ df$rank) %>% predict() %>% lines(col="red", lwd = 3)
smooth.spline(df$GBM ~ df$rank) %>% lines(col="red", lwd = 3)

df$Cluster.8 %>% points(pch=20, col="springgreen4", cex=1)
# loess(df$Cluster.8 ~ df$rank, na.action = na.exclude) %>% predict() %>% lines(col="green", lwd = 3)
lm(df$Cluster.8 ~ df$rank, na.action = na.exclude) %>% predict() %>% lines(col="springgreen4", lwd = 3)
abline(h=0,col="gray50", lwd=2)

dev.off()


### 2  heat map individual level drug response #### 
library(dplyr)
library(pheatmap)
library(ggplot2)
library(ComplexHeatmap)


pdf("~/GBM/Candidates_analysis/CommplexHeatmap_DrugLevel_s.pdf", width = 8, height = 10)
png(file = "~/GBM/Candidates_analysis/CommplexHeatmap_DrugLevel_s.png", width = 8, height = 10, units = "in", res = 300)

paste("./Candidates_analysis/", "Medians_FC_perDrug.csv", sep = "") |>
  read.csv(header = T, row.names = 1) |> 
  arrange(desc(GBM)) |>
  ComplexHeatmap::Heatmap(cluster_rows = FALSE,
                          # clustering_distance_columns = "minkowski",
                          # clustering_method_columns = "ward.D",
                          # clustering_distance_columns = d,
                          # clustering_method_columns = m,
                          column_split = 6
                          # column_title = t
  )

dev.off()  



pheatmap(df1,  
         cluster_row = FALSE,
         # cluster_cols = FALSE
         cutree_cols = 20
         
)





### 3. reversal score/strength calculation ####

###### 3.1 data prepare:  calculate the reverse strength on each gene at cluster level ####
## idea: if  gene's expression in GBM are reversed by drug, the make the fold change positive, 
# stands for a fold change that the drug can reverse/restore; other wise, make the fold change negative, 
# stands for the drug can make the GBM worse.
"~/SignatureCluster_average.csv" %>% read.csv(header = T, row.names = 1) -> df1
df1 <- df[,1:4]


for (i in 1:nrow(df1)) {
  
  if (df1$GBM[i] > 0) {
    
    df1[i,2:4] <- -df1[i,2:4]
    
  }
  
}
write.csv(df1,"~/iLINCS/Gene.regulateion_Cluster.average.csv")



###### 3.2 data prepare:  calculate the reverse strength on each gene at drug level ####

df1 <- read.csv("~/iLINCS/merged_sig_all_new.csv", header = T, row.names = 1) 

for (i in 1:nrow(df)) {
  
  if (df$GBM[i] > 0) {
    
    df[i,3:352] <- -df[i,3:352]
    
  }
  
}
write.csv(df,"~/iLINCS/Gene.regulateion_per.signature.csv")



