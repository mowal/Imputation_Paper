**Results**

Several scoring metrics for all models on both compound-based and assay-based splits can be found in [Scores](Scores). This folder also contains the results for the investigations on chemical similarity and data availability. For the ToxCast dataset, scores for different models are aggregated in summary files ([general and GHOST](Scores/sores_toxcast_incl_GHOST.csv),[sparsity](Scores/scores_toxcast_sparse.csv),[Aromatase with assay selection](Scores/scores_toxcast_aromatase_medians.csv)).

The folder [Predictions](Predictions) contains the predictions for indivudual compounds for the models on assay-based splits as required for the analysis on chemical similarity, data availability and GHOST analysis.

The files [info_metrics_Ames](info_metrics_Ames.csv), [info_metrics_Tox21](info_metrics_Tox21.csv) and [info_metrics_Toxcast.csv](info_metrics_Toxcast.csv) contain the computed information theory metrics used in the analysis on pairwise Feature Net models and auxiliary assay selection. 
