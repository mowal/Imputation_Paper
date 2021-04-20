**Code**

To run the scripts, you can use the Anaconda environment [imputation_env.yml](imputation_env.yml) or [macau_env.yml](macau_env.yml) for the Macau models. Instructions on how the environments can be created from the .yml files can be found in the [Anaconda documentation](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file). 


The chemical standardisation and subsequent aggregation of duplicate SMILES is done in [Ames_standardisation_aggregation.py](Ames_standardisation_aggregation.py) and [Tox21_standardisation_aggregation.py](Tox21_standardisation_aggregation.py).

The generation of train-test splits is done in [Ames_splitting.py](Ames_splitting.py).

The training of the machine learning models can be found in the model.py files (e.g. [XGB_model.py](XGB_model.py)) for the example of the Ames dataset and assay-based splits. The scripts can be easily adapted for the Tox21 dataset and/or compound-based splits by inserting the resepective train/test files.

The analysis on the role of chemical similarity in imputation models was done using the script [imputation_chemical_similarity.py](imputation_chemical_similarity). Note that the Tanimoto similarity matrices are not stored in this repository due to their size. They can be generated using the script [compute_Tanimoto_similarity.py](compute_Tanimoto_similarity.py).

The analysis on the role of data availability in imputation models was done using the script [imputation_data_availability.py](imputation_data_availability.py).

The pairwise Feature Net models were trained in the script [XGB_FN_pairwise_model.py](XGB_FN_pairwise_model.py). The information theory metrics to measure relatedness between assays were computed in [information_metrics.py](information_metrics.py).
