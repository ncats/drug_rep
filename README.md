<h2 align="center">Drug Repurposing for Rare Diseases</h2>

This [repository](https://github.com/ncats/drug_rep) contains the code for an analytical pipeline that utilizes data from open-source chemical databases to identify novel compound-gene target relationships for rare disease drug prioritization. More specifically, the pipeline includes clustering of compounds from the [Toxicology in the 21st Century (Tox21)](https://ntp.niehs.nih.gov/whatwestudy/tox21/index.html) program using SOM, identifying compound-gene target relationships using data from NIH's [Pharos](https://pharos.nih.gov/) and Broad Institute's [Drug Repurposing Hub](https://www.broadinstitute.org/drug-repurposing-hub), and performing gene enrichment analysis to obtain a list of enriched gene targets for each cluster. 

### Key markdown files

- **clustering.Rmd**: Cleans the original Tox21 dataset `tox21_data.txt` and uses the kohonen R package to perform SOM clustering. 

- **compound_gene_relationships.Rmd**: Identify compound-gene relationships for the 7170 compounds (note: 8971 original compounds --> data cleaning & keeping only compounds with complete bioassay data --> 7,170 compounds) using the databases Pharos and Drug Repurposing Hub. In the end, we have *7,030* compounds with the chemical structure identifier SMILES (6503 from parent smiles list and 527 from the aggregated tox21 data).

- **gene_enrichment_analysis.Rmd**: Uses results from `compound_gene_relationships.Rmd` to identify a list of *enriched* gene targets for each cluster. The Bum class is used to address the issue of multiple comparisons. 

### Key folders 
- **data**
- **results** 


### Key data files 
- **tox21_data.txt**: The original data file for the entire project; contains the bioassay activity profile data for each of the Tox21 compounds. A list of bioassays used in the Tox21 program can be found [here](https://tripod.nih.gov//tox21/pubdata/). Compound activity was measured by curve rank, a value between -9 and 9 determined by the potency, efficacy and quality of the concentration-response curve (a large positive number represents strong activation and a large negative number represents strong inhibition of the assay target). The file contains 8,971 compounds and 243 variables. 

- **tox21_aggregated.txt**: Additional aggregated chemical information for the Tox21 compounds; variables of interest include the CAS RN number, sample name, assay outcomes(i.e., active, inactive, inconclusive), SMILES, TOX21_ID, and etc. 

- **hgnc_data.txt**: Contains the HGNC ID, approved gene symbol, and approved gene name for all the gene targets. Custom downloads could be obtained [here](https://www.genenames.org/download/custom/). This dataset is used to better understand what each gene target represents (e.g., A1BG = "alpha-1-B glycoprotein", ACACA	= "acetyl-CoA carboxylase alpha").

- **clusters.txt**: Contains the results from SOM clustering (i.e., which cluster does each compound belongs to); generated from `clustering.Rmd`.

- **tox21_10k_cas_smiles_without_salts.txt**: Contains the [CAS](https://www.cas.org/cas-data/cas-registry) Registry Number and the parent SMILES for a selection of the Tox21 compounds.

- **broad_reformatted.txt**: Data pulled and organized from the Broad Institut's Drug Rep Hub. The data version used is from *3/24/2020* and can be obtained [here](https://clue.io/repurposing#download-data). Variables include `CompoundName`, `TargetGene`,	`PubchemCID`, `InKey`, `MOA`, and etc. 

- **pharos_all_target_ligands-p1/2.csv**: Data from NIH's Pharos database; can be obtained [here](https://pharos.nih.gov/targets). Variables of interest include `UniProt`, `Symbol`, `Ligand Name`, `Ligand SMILES`, and etc. Note that the data file is split in two parts due to its large file size. 

