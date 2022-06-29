# Drug Repurposing for Rare Diseases 

**smiles_to_inchikey.Rmd**

This file **identify compound-gene relationships** for the 7170 compounds (recall: 8971 original compounds --> data cleaning & keeping only compounds with complete bioassay data --> 7,170 compounds) using data from Pharos and Drug Repurposing Hub. **7,030** compounds with smiles (6503 from parent smiles list and 527 from the aggregated tox21 data). 

**enrichment_analysis.Rmd**

This file uses compound-gene relationships (pulled from the drug repurposing hub and Pharos) to identify a list of **enriched** gene targets for each cluster. 
