# E-KAN: Ensemble Kolmogorov-Arnold Network

E-KAN, or Ensemble-KAN, leverages the non-linear modeling capabilities of Kolmogorov-Arnold Networks (KANs) to enhance accuracy, especially in cases with multiple data sources. The model was tested against traditional machine learning models for discriminating between recent-onset psychosis (ROP) or depression (ROD) and healthy controls using multimodal environmental and neuroimaging data. E-KAN demonstrated superior performance over these traditional models.

## Key Features

- **Ensemble Learning**: Combines multiple KAN models, each trained on a different subset of features, to improve overall predictive performance.
- **Feature Selection**: Utilizes statistical tests to select the most relevant features for each KAN model, improving individual model performance.
- **Meta-Learner**: A final KAN model integrates the outputs from the ensemble layer, allowing for complex non-linear combinations of model predictions.
- **Explainability**: Provides subject-specific patterns through SHAP (Shapley Additive Explanations), facilitating the identification of key predictive features.

## Architecture

1. **Data Preprocessing**: Mitigates confounding effects through a preprocessing pipeline.
2. **First Feature Selection (FS) Step**: Subdivides the dataset into uniform feature subsets and selects key features using an F-statistic similarity test.
3. **KAN Base Learners**: Each subset of features is fed into a KAN model to predict class labels.
4. **Model Prediction Ranking**: Selects the most relevant predictions from the ensemble layer using the χ2 test.
5. **KAN Meta-Learner**: Combines the predictions from the base learners to make the final decision, leveraging a non-linear modeling framework.

![Project Logo](https://github.com/brainpolislab/E-KAN/blob/main/KAN_ensamble.png)

## Models Hyperparameters

The following table summarizes the hyperparameters used for each model in our study, the default ones are not specified:

| Model                  | Hyperparameters                             |
|------------------------|---------------------------------------------|
| E-KAN (our method)     | k1 = 4, k2 = 4, N = 8                       |
| XGB (Default)          | n_estimators = 100                          |
| XGB (n_estimators=20)  | n_estimators = 20                           |
| RandomForest           | n_estimators = 100                          |
| RandomForest (n_estimators=20) | n_estimators = 20                   |
| SVM                    | Default parameters                          |
| Adaboost               | Default parameters                          |
| TabNet                 | n_steps = 10, optimizer_fn = torch.optim.Adam, lambda_sparse = 1e-4, momentum = 0.3, scheduler_params = {"step_size": 10, "gamma": 0.9}, scheduler_fn = torch.optim.lr_scheduler.StepLR, max_epochs = 200, patience = 300, batch_size = 52, virtual_batch_size = 52, weights = 1, drop_last = False |
| KAN base learner       | G = 6, K = 3, steps = 20 , width=[input_size,1,2] |                   |
| Deep KAN               | G = 6, K = 3, steps = 20, width=[input_size,8,4,2], lamb=0.9 |

## Description of the repo
This is the repo with all the code needed to run the experiments described in the related article under review for the MICCAI workshop AMAI.

## Notes about the file organization
With CV-KAN you create a corrected version of the folds according to the correction pipeline. With KAN validation you use the corrected data to cross-validate the models and gather the results.
With KAN explainability you train the algorithm and explain it (also here are defined the methods to train the model and a method to use it to make predictions. the model architecture used for CV is defined in PRONIA functions

## Notes about the usability of this repo

Currently, this version is intended to support the results of the proceedings so it contains the scripts that define the structure of the algorithm and the validation framework
