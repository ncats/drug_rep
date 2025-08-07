##     A Computational Biomarker Discovery Framework for CLN3 Disease

#### CLN3 disease is a rare, inherited neurodegenerative disorder characterized by progressive cognitive decline, seizures, and vision loss. This folder contains the information related to a computational biomarker discovery framework that built various machine learning models to identify novel biomarker candidates for CLN3 disease by analyzing proteomics data and laboratory tests collected from participants in a CLN3 prospective, observational study.  
#### This framework is composed of three components: 1) machine learning models with optimized imputation methods to analyze proteomics and laboratory data from CLN3 patients to identify protein candidates; 2) PPI network-based network analysis to prioritize candidates; and 3) corroboration using external gene expression datasets.

#### Scripts folder  
- 0.0_Data_imputation.R: Cleans the original proteomics data and laboratory tests collected from the clinical trial; evaluates different imputation methods and imputes the missing values using the best performing method.   
- 1.1.1_Classification_model_setup.R: Builds up various machine learning models for the classification subset.  
- 1.1.2_Classification_model_evaluation.R: Evaluates the models built in 1.1.1_Classification_model_setup.R to get the model with the best performance.  
- 1.1.3_Classification_feature_selection.R: Selects a list of protein features from the classification subset using the best model identified in 1.1.2_Classification_model_evaluation.R.  
- 1.2.1_Disease_severity_model_setup_and_evaluation_py3.6.ipynb: Builds up various machine learning models for the severity subset and evaluates model performance to identify the best model.  
- 1.2.2_Disease_severity_feature_selection.R: Uses the best model identified from 1.2.1_Disease_severity_model_setup_and_evaluation_py3.6.ipynb to select a list of features from the severity subset.   
- 2.1.2_PPI_hub_genes.R: Evaluates the centrality of the selected protein features based on their Protein-Protein Interaction (PPI) network.  
- 2.2_External_validation.R: Uses an external transcriptomic dataset (GEO accession: GSE22225) to evaluate the diagnostic potential of the identified candidate biomarkers.  
- functions.R: Contains self-defined functions used in the computational framework.  

#### Results folder  
- PCA_imputed_Linear_data.csv: Contains the severity dataset after PCA-based imputation (resulted from 0.0_Data_imputation.R).   
- rf_imputed_logistic_data.csv: Contains the classification dataset after Random Forest-based imputation (resulted from 0.0_Data_imputation.R).  
- logi_boostrap_stable_features_nb500.csv: Contains stable features selected from the classification subset (resulted from 1.1.3_Classification_feature_selection.R).  
- RF_boostrap_zScore2_top_features.xlsx: Contains the features selected from the severity subset (resulted from 1.2.2_Disease_severity_feature_selection.R).  
- all_protines_ranked_by_overall_metrics.csv: Contains the protein features ranked by centrality based on their Protein-Protein Interaction (PPI) network (resulted from 2.1.2_PPI_hub_genes.R).  
- RF_top_20_features_barplot.pdf: Bar plots of the top 20 features’ importance from the random forest model (resulted from 1.2.2_Disease_severity_feature_selection.R).   


<img width="468" height="647" alt="image" src="https://github.com/user-attachments/assets/1a520619-5aae-4d71-9e59-3fde3e8f65cb" />
