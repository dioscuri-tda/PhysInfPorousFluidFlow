# PhysInfPorousFluidFLow
![Header](images/header.png)
## Physics-informed convolutional neural networks for fluid flow through porous media

This repository contains the codebase for the paper

*Physics-informed convolutional neural networks for fluid flow through porous media*

## Model
The weights of the trained champion model, i.e. ResNet101 as indicated by Table 2 in the manuscript, is available HERE.
After downloading put the `best_model.pth` file into `models/` directory. The MD5sum for the model is
```
0c673bf4bafe997a026f964f82521669  best_model.pth
```

## Dataset
Here we publish the dataset of randomly deposited circles and squares, Pcirce=0.5. 
This dataset is the one use to produce row `shape, Pcircle = 0.5` in Table 3 in the manuscript. 
The dataset consists of 490 structures and their LBM solutions.
* `dataset.evaluation/structures`: contains the images of the structures
* `dataset.evaluation/velocities_lbm`: contains the LBM solutions. The filename matches the filename of the structure. Velocities fields are written as flattened NumPy arrays.

## Model inference

![Header](images/table.png)


Here we demonstrate how to make a predictions using the model weights and reproduce some of the results from the paper. 
Run the `run_inference.sh` which performs the evaluation and saves the results into the `dataset.evaluation/velocities_prediction`.
```
bash run_inference.sh
```
The output contains:
* `evaluation.csv_crop256`: a file that aggregates the inference results. For each structure in the dataset the MSE of predicted velocity field is given together with predicted and ground truth permeability and tortuosity. The file should look like this:
```
filename,mse,penalty_inside,penalty_div,tortuosity_target,tortuosity_pred,permeability_target,permeability_pred
pcircle=0.5000_radmin=3_radmax=8_trivialporosity=0.800873,3.0496872568619438e-05,3.3787271149776643e-06,3.7280046853993554e-06,1.1785376071929932,1.1886672973632812,18541.455078125,19398.82421875
pcircle=0.5000_radmin=3_radmax=8_trivialporosity=0.855301,4.177633672952652e-05,3.0870128284732345e-06,5.521895218407735e-06,1.1389719247817993,1.125132441520691,39148.82421875,39782.59765625
[...]
```
* `evaluation.csv_crop256_metrics.json`: basic metrics
* `predictions_evaluation.pickle`: binary file the stores the predicted velocity fields

The notebook `notebooks/VisualizePrediction.ipynb` shows how reproduce all the metric from Table 3 in the manuscript. 
It provides also visualization of the structures, solutions of the LBM method and corresponding predictions of the model.

![Header](images/prediction_squares.png)
