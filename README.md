# Graph classification using kernel methods

This repository is the work produced by Ludovic De Matteïs and Matias Etcheverry in the course *Machine learning with kernel methods* given by Michael Arbel, Alessandro Rudi, Jean-Philippe Vert and Julien Mairal at the MVA.

## Objective

The goal of this repository is to implement machine learning algorithms, for a classification task on graph data. 

## Installation

In order to run this repository, you need to run the following:

```
# clone the repository
git clone git@github.com:MatiasEtcheve/KM-graph-classification.git
cd KM-graph-classification

# install the dependencies
pip install -r requirements.txt
```

## Inference

The Command Line Interface can be used efficiently to compute predictions on the dataset.

In order to use our best model, you can run:
```
python start.py
```
This will train 2 kernels: 
* a Weisfeiler Lehman kernel on the edges of the graphs
* a Weisfeiler Lehman kernel on the nodes of the graphs
A linear combination is then applied between the logits of the 2 models.

However, you can also tune the learning, with the CLI options:

| Option name | Type | Description | Default Value |
|---|---|---|---|
| `kernels` | list | Kernels to train on the data. If multiple kernels are provided, <br>it will individually train on each kernel, then do a linear <br>combination of the logits.  Must be one (or multiples) of <br>"EH", "VH", "SP", "GL", "WL-Edges", "WL-Nodes" | `"[WL-Edges,WL-Nodes]"` (care quotes !) |
| `combination` | list | list of coefficient for the combination of kernels | `"[1.59,1.35]"` (care quotes !) |
| `max-alpha` | float | Max value of the alpha coefficient in SVM. **Note:** multiple alphas <br>can be higher than C, when `class_weight=balanced`. | `100` |
| `sigma` | float >= 0 | Sigma in the RBF wrapper. If 0, a linear wrapper is applied instead. | Defaults to `1` |
| `src` | folder | Path to .pkl datasets. | `data/` |
| `train-val-split` | float or int | Train val split, in ratio or in number of elements, eg 0.7 or 4200. <br>Usefull when training takes time. | `0.7` |
| `do-predict` | flag | whether to do the prediction on the test set. | `True` |
| `predict-filename` | filename | path to the prediction (if `do-predict` flag is present). | `test_pred.csv` |

**Example**:
Examples of working command lines:
* Default command: `python start.py --kernels "[WL-Edges,WL-Nodes]" --combination "[1.59,1.35]" --src data/ --train-val-split 0.7 --do-predict --predict-filename data/predictions.csv`  
* another command: `python start.py --kernels "[EH,VH]" --combination "[1,1]" --max-alpha 1 --sigma 0 --src data/ --train-val-split 0.4 --do-predict --predict-filename data/predictions.csv`  