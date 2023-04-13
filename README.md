## Objective

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
python main.py
```
This will train 2 kernels: 
* a Weisfeiler Lehman kernel on the edges of the graphs
* a Weisfeiler Lehman kernel on the nodes of the graphs
A linear combination is then applied between the logits of the 2 models.

However, you can also tune the learning:
  * `kernels` (list): Kernels to train on the data. If multiple kernels are provided, it will individually train on each kernel, then do a linear combination of the logits.  Must be one (or multiples) of "EH", "VH", "SP", "GL", "WL-Edges", "WL-Nodes". Defaults to `"[WL-Edges,WL-Nodes]"` (care quotes !)
  * `combination` (list): list of coefficient for the combination of kernels. Defaults to `"[1.59,1.35]"` (care quotes !)
  * `src` (str): path to .pkl datasets. Defaults to `data/`
  * `train-val-split` (float or int): Train val split, in ratio or in number of elements, eg 0.7 or 4200. Usefull when training can take time. Defaults to `0.7`.
  * `do-predict`: whether to do the prediction on the test set. Defaults to `True`
  * `predict-filename` (str): path to the prediction (if `do-predict` flag is present). Defaults to `data/predictions.csv`

**Example**:
Examples of working command lines:
* Default command: `python main.py --kernels "[WL-Edges,WL-Nodes]" --combination "[1.59,1.35]" --src data/ --train-val-split 0.7 --do-predict --predict-filename data/predictions.csv`  
* Default command: `python main.py --kernels "[EH,VH]" --combination "[1,1]" --src data/ --train-val-split 0.4 --do-predict --predict-filename data/predictions.csv`  