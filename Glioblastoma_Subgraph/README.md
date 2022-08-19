<h2 align="center">Network Analysis-Based Drug Repurposing for Glioblastoma</h2>

This [folder](https://github.com/ncats/drug_rep/tree/main/Glioblastoma_Subgraph) 


### Code files

- **Create_Node_and_Edge_Lists.ipynb**: 

- **Merge_Nodes.ipynb**: This code takes a list of graph nodes and a list of graph edges as input. It then "merges" (see documentation within file for details) all nodes connected by edges labeled “I_CODE”, “N_Name”, "R_exactMatch", "R_equivalentClass", and "PAYLOAD" and produces the modified node and edges lists as an output. 



### Data files 

- **GBN_Node_List.csv**: A list of the 1,466 nodes contained in the GBN after associated nodes were merged via [Merge_Nodes.ipynb](https://github.com/ncats/drug_rep/blob/main/Glioblastoma_Subgraph/Merge_Nodes.ipynb). Can be used in conjunction with [GBN_Edge_List.csv](https://github.com/ncats/drug_rep/blob/main/Glioblastoma_Subgraph/GBN_Edge_List.csv) to reconstruct the GBN in a visualization tool (e.g. Gephi).

- **GBN_Edge_List.csv**: A list of the 107,423 edges contained in the GBN after associated nodes were merged via [Merge_Nodes.ipynb](https://github.com/ncats/drug_rep/blob/main/Glioblastoma_Subgraph/Merge_Nodes.ipynb). As edges are directed, each is denoted by a "Source" and "Target" node. Can be used in conjunction with [GBN_Node_List.csv](https://github.com/ncats/drug_rep/blob/main/Glioblastoma_Subgraph/GBN_Node_List.csv) to reconstruct the GBN in a visualization tool (e.g. Gephi).

- **GBN_Centrality_Scores.csv**: 

- **mc_GBN_Node_Modularity_Classes.csv**: 

- **mc_GBN_Node_Lists folder**: 

- **mc_GBN_Edge_Lists folder**: 

- **Modularity_Class_Centrality_Tables folder**:



### Result files

- **mc_GBN_Modularity_Class_Descriptions.pdf**: 

- **Centrality_Score_Tables.pdf**: 

- **Top_Candidate_Reference_List.xlsx**: 
