# E-KAN



![Project Logo](https://github.com/GianlucaDF/E-KAN/blob/main/KAN_ensamble.png)

## Description
This is the repo with all the code needed to run the experiments described in the related article under review for the MICCAI workshop AMAI
## Notes about the file organization
With CV-KAN you create a corrected version of the folds according to the correction pipeline
with KAN validation you use corrected data to cross-validate the models and gather the results
with KAN explainability you train the algorithm and explain it (also here are defined the methods to have a trained model and a method to use it to make predictions. the model architecture used for CV is defined in PRONIA functions

## Models Hyperparameters

The following table summarizes the hyperparameters used for each model in our study:

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
| Deep KAN               | G = 6, K = 3, steps = 20, width=[input_size,8,4,2] |

## Notes about the usability of this repo

Currently, this version is intended to support the results of the proceedings so it contains the scripts that define the structure of the algorithm and the validation framework
