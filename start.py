import ast
import os
import pickle
import time
from functools import partial
from typing import List

import click
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from kernels import EdgeHistKernel, GraphletKernel, NodeHistKernel, SPKernel, WLKernel
from solver import SVM
from utils import relabel

np.random.seed(42)


NAME_TO_KERNEL = {
    "EH": EdgeHistKernel,
    "VH": NodeHistKernel,
    "SP": SPKernel,
    "GL": GraphletKernel,
    "WL-Edges": partial(WLKernel, edge_attr="labels", node_attr=None),
    "WL-Nodes": partial(WLKernel, edge_attr=None, node_attr="labels"),
}


class ConvertStrToList(click.Option):
    def type_cast_value(self, ctx, value) -> List:
        try:
            value = str(value)
            assert value.count("[") == 1 and value.count("]") == 1
            list_as_str = value.replace('"', "'").split("[")[1].split("]")[0]
            list_of_items = [item.strip().strip("'") for item in list_as_str.split(",")]
            return list_of_items
        except Exception:
            raise click.BadParameter(value)


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option(
    "--kernels",
    cls=ConvertStrToList,
    default=["WL-Edges", "WL-Nodes"],
    show_default=True,
    help="""Kernels to train on the data. If multiple kernels are provided,
    it will individually train on each kernel, then do a linear  
    combination of the logits to produce the best classifier. 
    Must be one (or multiples) of "EH", "VH", "SP", "GL", "WL-Edges", "WL-Nodes". """,
)
@click.option(
    "--combination",
    cls=ConvertStrToList,
    default=[1.59, 1.35],
    show_default=True,
    help="""If multiple kernels are prodived, coefficients of the linear combination.""",
)
@click.option(
    "--max-alpha",
    type=float,
    default=100,
    show_default=True,
    help="Max value of the alpha coefficient in SVM. Note: multiple alphas can be higher than C, when `class_weight=balanced`",
)
@click.option(
    "--sigma",
    type=float,
    default=1,
    show_default=True,
    help="Sigma in the RBF wrapper. If 0, a linear wrapper is applied instead.",
)
@click.option(
    "--src",
    type=click.Path(exists=True, file_okay=False, readable=True),
    default="data/",
    show_default=True,
    help="Path to .pkl datasets",
)
@click.option(
    "--train-val-split",
    type=float,
    default=0.7,
    show_default=True,
    help="Train val split (in ratio or in number of elements)",
)
@click.option(
    "--do-predict",
    is_flag=True,
    show_default=True,
    default=True,
    help="Whether to do the prediction on the test set",
)
@click.option(
    "--predict-filename",
    type=click.File(mode="w"),
    default="test_pred.csv",
    show_default=True,
    help="Path to csv prediction",
)
def main(
    kernels,
    combination,
    max_alpha,
    sigma,
    src,
    train_val_split,
    do_predict,
    predict_filename,
):
    # Extract data
    train_data_path = os.path.join(src, "training_data.pkl")
    train_label_path = os.path.join(src, "training_labels.pkl")
    test_data_path = os.path.join(src, "test_data.pkl")

    with open(train_data_path, "rb") as f:
        graphs = np.array(pickle.load(f), dtype="object")
        relabel(graphs)
    with open(train_label_path, "rb") as f:
        labels = np.array(pickle.load(f))
        # put labels in [-1, 1]
        labels = 2 * labels - 1
    if do_predict:
        with open(test_data_path, "rb") as f:
            graphs_test = np.array(pickle.load(f), dtype="object")
            relabel(graphs_test)

    if train_val_split > 1 and int(train_val_split) == float(train_val_split):
        train_val_split /= len(graphs)
    graphs_train, graphs_val, labels_train, labels_val = train_test_split(
        graphs,
        labels,
        train_size=train_val_split,
    )
    graphs_train = np.array(graphs_train, dtype="object")
    graphs_val = np.array(graphs_val, dtype="object")
    labels = {"train": labels_train, "val": labels_val}
    print(f"Size training set:   {len(graphs_train):>6}")
    print(f"Size validation set: {len(graphs_val):>6}")

    # Instanciate kernels
    kernels = [NAME_TO_KERNEL[name](sigma=sigma) for name in kernels]
    print(f"Number of kernels: {len(kernels)}")

    # Instanciate SVMs
    # 1 per kernel
    class_weight = "balanced"
    models = [
        SVM(C=max_alpha, kernel=kernel, class_weight=class_weight) for kernel in kernels
    ]

    # Train SVMs
    # 1 fit per kernel
    start_time = time.time()
    for idx_model, model in enumerate(models):
        print("--------------------------------------")
        print(
            f"Training kernel {idx_model+1}/{len(models)}: {str(kernels[idx_model])}... "
        )
        model.fit(
            graphs_train,
            labels["train"],
        )
        print("Elapsed time: {:.3f} seconds".format(time.time() - start_time))
        start_time = time.time()

    # Compute Logits
    # 1 set of logits per kernel
    logits = []  # set of predictions per kernel
    global_logits = {
        "train": 0,
        "val": 0,
        "test": 0,
    }  # linear combination of all predictions
    for idx_model, model in enumerate(models):
        train_logits = model.decision_function(
            graphs_train,
        )
        val_logits = model.decision_function(
            graphs_val,
        )

        logits.append({"train": train_logits, "val": val_logits})
        global_logits["train"] = (
            global_logits["train"] + float(combination[idx_model]) * train_logits
        )
        global_logits["val"] = (
            global_logits["val"] + float(combination[idx_model]) * val_logits
        )

        if do_predict:
            test_logits = model.decision_function(
                graphs_test,
            )
            logits[-1]["test"] = test_logits
            global_logits["test"] = (
                global_logits["test"] + float(combination[idx_model]) * test_logits
            )

    # Display metrics
    for idx_kernel, kernel in enumerate(kernels):
        print("--------------------------------------")
        print(
            f"Evaluating {str(kernels[idx_kernel])}:",
        )
        for split_name in ["train", "val"]:
            current_logits = logits[idx_kernel][split_name]
            predictions = np.sign(current_logits)
            print(
                f"{split_name:<6} Accuracy: {accuracy_score(labels[split_name], predictions):>6.3f} |",
                f"{split_name:<6} Recall: {recall_score(labels[split_name], predictions):>6.3f} |",
                f"{split_name:<6} F1: {f1_score(labels[split_name], predictions):>6.3f} |",
                f"{split_name:<6} AUC: {roc_auc_score(labels[split_name], predictions):>6.3f}",
            )

    if len(models) > 1:
        print("--------------------------------------")
        print(
            f"Evaluating linear combination of all models:",
        )
        for split_name in ["train", "val"]:
            current_logits = global_logits[split_name]
            predictions = np.sign(current_logits)
            print(
                f"{split_name:<6} Accuracy: {accuracy_score(labels[split_name], predictions):>6.3f} |",
                f"{split_name:<6} Recall: {recall_score(labels[split_name], predictions):>6.3f} |",
                f"{split_name:<6} F1: {f1_score(labels[split_name], predictions):>6.3f} |",
                f"{split_name:<6} AUC: {roc_auc_score(labels[split_name], predictions):>6.3f}",
            )

    if do_predict:
        df = pd.DataFrame(global_logits["test"], columns=["Predicted"])
        df.index += 1
        df.to_csv(predict_filename, index_label="Id")
        print("--------------------------------------")
        print(f"Predictions saved at {predict_filename.name}")

    print(
        "Disclaimer: our Kaggle submissions might differ a bit, for several reasons:",
        "\n\t* the seed of our algorithm was not fixed",
        "\n\t* when submitting on Kaggle, we train our model on the whole dataset.",
    )


if __name__ == "__main__":
    main()
