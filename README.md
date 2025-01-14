This repository provides the python/r code and raw data file used in research "Optimizing human SIRT6 protein with deep learning of 3D structures based on maximum lifespan". 
This is <b>my first research project completed independently :smile:</b>, it developed a comprehensive framework of sequence optimization methods for functional proteins whose efficiency is relatively harder to obtain or measure directly, therefore corresponding code can not only be used for the generation and site prediction of SIRT6 variants, but also for other protein sequences.

# Related Publication & patent
1. Zhang, Y., & Ruan, Z. (2024). Optimizing human SIRT6 protein with deep learning of 3D structures based on maximum lifespan. Highlights in Science, Engineering and Technology, 102, 51-60. https://doi.org/10.54097/8b4msf08

2. Zhang, YF. Human SIRT6 mutant, CN202311362427.1, 2023.10.19 (Patent in Chinese)
   
# Key Sites in SIRT6 mutation
![image](https://github.com/user-attachments/assets/70cff515-83bf-415b-b25d-2399168c6eec)

# File Description
|File Name|Function|
|---|---|
|<b>SIRT6fullarticle.pdf|Complete paper for participate in S.-T. Yau High School Science Award 2023, with more detailed content than the published paper|
|<b>/data||
|<b>anage_data.txt|Maximum lifespan (MLS) retrieved from the AnAge database (a sub-library of HAGR, https://genomics.senescence.info/species/index.html)|
|<b>mammal_meta_info.csv|Supplement document of the study "Lu AT, Fei Z, Haghani A, et al. Universal DNA methylation age across mammalian tissues. Nat Aging. Published online August 10, 2023. doi:10.1038/s43587-023-00462-6", provide additional MLS data|
|<b>sirt6_sequence_1307.fasta|SIRT6 sequences retrieved from NCBI ortholog database (https://www.ncbi.nlm.nih.gov/gene/51548/ortholog/?scope=40674&term=SIRT6), FASTA format|
|<b>/Script||
|<b>1_NCBIdata.r| Arrange raw NCBI sequences into a data table format| 
|<b>1_dataclean.py | Combine MLS and sequence data files |
|<b>3_RoseTTAFold-transfer_mod.ipynb | RoseTTAFold model for calculating 3D PDB structures of all sequences, modified from the officially provided notebook file RoseTTAFold.ipynb, colab python code, https://colab.research.google.com/drive/1whVfMQ-syuXFCzv7RokTMt6_g6B6JEep?usp=sharing | 
|<b>3_compute_AA_distance142.py | Calculates distance matrices and dihedral angle data based on predicted PDB file, and performs preliminary variable screening |
|<b>3_Age_predict_CNN_cv_refine.py |  analysis dataset for CNN model |
|<b>4_transfer_RoseTTA_mtx.py | Extracts and organizes the working matrices of RoseTTAFold | 
|<b>5_Translearning_model.py | Transfer CNN model based on working matrices | Organized working matrix files | 
|<b>6_transfer_learning_ESM.py | ESM prediction model | 
|<b>6_proteinmpnn_in_jax.ipynb | MPNN model used for human SIRT6 mutant generation, colab python code, https://colab.research.google.com/drive/1EpHMqmEp1d8_ufBuDa2zN4kEksGNLYRX?usp=sharing |
|<b>6_genrawstr_diffusion.py | Supplements the sequences generated by proteinMPNN to a length of 355, calculates predicted MLS, and screens for optimized sequences | 
|<b>7_sitesearch.py | Calculates the importance and sorts the sites included in the analysis |
|<b>8_result_graph.py | Draws some statistical graphs for the article | 


# Abstract
DNA damage, particularly double-strand breaks (DSB), plays an important role in aging, carcinogenesis, and other diseases. The efficacy of the DSB repair protein SIRT6 is known to correlate with the maximum lifespan (MLS) across species. However, it is still unclear whether the function of SIRT6 can be further optimized through protein sequence engineering. Here, we used RoseTTAFold to predict the structural variance of SIRT6 sequences across 142 mammalian species. We then analyzed the association between the MLS, sequence, 3D structures, and amino acid selection of SIRT6 and found that sequence and spatial information are correlated with MLS. By fine-tuning the ESM-fold model, we were able to accurately predict the MLS of the species from SIRT6 sequence (Pearson’s r = 0.818, MAE = 8.608 years). We further generated mutant sequences of the human SIRT6 using ProteinMPNN and analyzed different sites’ importance based on their impact on predicted MLS. A subset of 37 amino acids sites that play a key role in sequence function was found, and among them, 20 sites located in the NAD+ banding area and β1-sheet were found to have the greatest impact. We then tested the DNA repair efficiency of 2 novel SIRT6 sequences (with predicted MLS improvement of 16.5% and 20.1%), and immunofluorescence showed that their DSB repair efficiency is indeed higher than human ortholog SIRT6 sequence. Together, our study demonstrates that despite being highly conservative in evolution, there is still room for optimization of human SIRT6. We not only identified crucial sites correlated with sequence optimization but also designed optimized SIRT6 protein with longer MLS, thereby providing novel insights into potential anti-aging or anti-cancer interventions. This study also developed a comprehensive framework of sequence optimization methods for functional proteins whose efficiency is relatively harder to obtain or measure directly.
