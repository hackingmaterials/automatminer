"""
Use megnet v0.2.2 with some customized modification (These issues may would be
fixed in later updates of megnet repo.):
i) pass y_scaler to ModelCheckpointMAE in train_from_graphs/train;
ii) make ReduceLRUponNan read the best model in same logic with
ModelCheckpointMAE, namely reads the best file not the latest file

Please check the official megnet repo for more inspect.
"""
import os
import gzip
import json
import pickle
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
from keras.models import load_model
from keras.backend import set_session
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split
from tensorflow.python.client import device_lib
from megnet.models import MEGNetModel
from megnet.layers import Set2Set, MEGNetLayer
from megnet.activations import softplus2
from megnet.data.crystal import CrystalGraph
from megnet.data.graph import GaussianDistance
from megnet.losses import mean_squared_error_with_scale


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser("megnet benchmark.")
    add_arg = parser.add_argument
    add_arg("-o", "--output_path", required=True, help="output path")
    add_arg("-i", "--input_file", default="../data/jdft2d.pickle.gz", help="input file")
    add_arg(
        "-g",
        "--graph_file",
        default=None,
        help="graph file. If not None, load graphs from graph file, otherwise, create and save graphs",
    )
    add_arg(
        "-e",
        "--embedding_file",
        default=None,
        help="embedding file. If not None, transfer learning from embedding file, otherwise, not use transfer learning",
    )
    add_arg(
        "-t",
        "--type",
        default="regression",
        help="task type, regression or classification",
    )
    add_arg("-p", "--property", default="gap pbe", help="target name")
    add_arg(
        "-w",
        "--warm_start",
        default=None,
        help="warm start model file. If not None, warm start from file.",
    )
    add_arg("-n", "--n_works", default=1, type=int, help="n works for dataloader")
    add_arg("-m", "--max_epochs", default=3000, type=int, help="max epochs")
    add_arg("-b", "--batch_size", default=128, type=int, help="mini batch size")
    add_arg("-l", "--loss", default="mse", help="loss function")
    add_arg("-lr", "--learning_rate", default=None, type=float, help="learning rate")
    add_arg("-cv", "--cv", default=5, type=int, help="cv number")
    add_arg("-kf", "--k_folds", default="0,1,2,3,4", help="kfold indexes list")
    add_arg(
        "-r",
        "--radius",
        default="4.0",
        type=float,
        help="megnet parameter, radius: distance cutoff for neighbors",
    )
    add_arg(
        "-np",
        "--n_pass",
        default=2,
        type=int,
        help="megnet parameter, n_pass: number of recurrent steps in Set2Set layer",
    )
    add_arg(
        "-nb",
        "--n_blocks",
        default=3,
        type=int,
        help="megnet parameter, n_blocks: number of MEGNetLayer blocks",
    )
    add_arg(
        "-s",
        "--save_best_only",
        action="store_false",
        help="save best score models only or save all epoch models",
    )
    add_arg(
        "-norm", "--normalize", action="store_true", help="Normalize targets or not"
    )
    add_arg("-dc", "--disable_cuda", action="store_true", help="disable cuda or not")

    return parser.parse_args()


def train():
    # Parse args
    args = parse_args()
    radius = args.radius
    n_works = args.n_works
    warm_start = args.warm_start
    output_path = args.output_path
    graph_file = args.graph_file
    prop_col = args.property
    learning_rate = args.learning_rate
    embedding_file = args.embedding_file
    k_folds = list(map(int, args.k_folds.split(",")))
    print("args is : {}".format(args))

    print(
        "Local devices are : {}, \n\n Available gpus are : {}".format(
            device_lib.list_local_devices(), K.tensorflow_backend._get_available_gpus()
        )
    )

    # prepare output path
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    # Get a crystal graph with cutoff radius A
    cg = CrystalGraph(
        bond_convertor=GaussianDistance(np.linspace(0, radius + 1, 100), 0.5),
        cutoff=radius,
    )

    if graph_file is not None:
        # load graph data
        with gzip.open(graph_file, "rb") as f:
            valid_graph_dict = pickle.load(f)
        idx_list = list(range(len(valid_graph_dict)))
        valid_idx_list = [
            idx for idx, graph in valid_graph_dict.items() if graph is not None
        ]
    else:
        # load structure data
        with gzip.open(args.input_file, "rb") as f:
            df = pd.DataFrame(pickle.load(f))[["structure", prop_col]]
        idx_list = list(range(len(df)))

        # load embedding data for transfer learning
        if embedding_file is not None:
            with open(embedding_file) as json_file:
                embedding_data = json.load(json_file)

        # Calculate and save valid graphs
        valid_idx_list = list()
        valid_graph_dict = dict()
        for idx in idx_list:
            try:
                graph = cg.convert(df["structure"].iloc[idx])
                if embedding_file is not None:
                    graph["atom"] = [embedding_data[i] for i in graph["atom"]]
                valid_graph_dict[idx] = {
                    "graph": graph,
                    "target": df[prop_col].iloc[idx],
                }
                valid_idx_list.append(idx)
            except RuntimeError:
                valid_graph_dict[idx] = None

        # Save graphs
        with gzip.open(os.path.join(output_path, "graphs.pkl.gzip"), "wb") as f:
            pickle.dump(valid_graph_dict, f)

    # Split data
    kf = KFold(n_splits=args.cv, random_state=18012019, shuffle=True)
    for fold, (train_val_idx, test_idx) in enumerate(kf.split(idx_list)):
        print(fold)
        if fold not in k_folds:
            continue
        fold_output_path = os.path.join(output_path, "kfold_{}".format(fold))
        fold_model_path = os.path.join(fold_output_path, "model")
        if not os.path.exists(fold_model_path):
            os.makedirs(fold_model_path, exist_ok=True)

        train_idx, val_idx = train_test_split(
            train_val_idx, test_size=0.25, random_state=18012019, shuffle=True
        )

        # Calculate valid train validation test ids and save it
        valid_train_idx = sorted(list(set(train_idx) & (set(valid_idx_list))))
        valid_val_idx = sorted(list(set(val_idx) & (set(valid_idx_list))))
        valid_test_idx = sorted(list(set(test_idx) & (set(valid_idx_list))))
        np.save(os.path.join(fold_output_path, "train_idx.npy"), valid_train_idx)
        np.save(os.path.join(fold_output_path, "val_idx.npy"), valid_val_idx)
        np.save(os.path.join(fold_output_path, "test_idx.npy"), valid_test_idx)

        # Prepare training graphs
        train_graphs = [valid_graph_dict[i]["graph"] for i in valid_train_idx]
        train_targets = [valid_graph_dict[i]["target"] for i in valid_train_idx]

        # Prepare validation graphs
        val_graphs = [valid_graph_dict[i]["graph"] for i in valid_val_idx]
        val_targets = [valid_graph_dict[i]["target"] for i in valid_val_idx]

        # Normalize targets or not
        if args.normalize:
            y_scaler = StandardScaler()
            train_targets = y_scaler.fit_transform(
                np.array(train_targets).reshape(-1, 1)
            ).ravel()
            val_targets = y_scaler.transform(
                np.array(val_targets).reshape((-1, 1))
            ).ravel()
        else:
            y_scaler = None

        # Initialize model
        if warm_start is None:
            #  Set up model
            if learning_rate is None:
                learning_rate = 1e-3
            model = MEGNetModel(
                100,
                2,
                nblocks=args.n_blocks,
                nvocal=95,
                npass=args.n_pass,
                lr=learning_rate,
                loss=args.loss,
                graph_convertor=cg,
                is_classification=True if args.type == "classification" else False,
                nfeat_node=None if embedding_file is None else 16,
            )

            initial_epoch = 0
        else:
            # Model file
            model_list = [
                m_file
                for m_file in os.listdir(
                    os.path.join(warm_start, "kfold_{}".format(fold), "model")
                )
                if m_file.endswith(".hdf5")
            ]
            if args.type == "classification":
                model_list.sort(
                    key=lambda m_file: float(m_file.split("_")[3].replace(".hdf5", "")),
                    reverse=False,
                )
            else:
                model_list.sort(
                    key=lambda m_file: float(m_file.split("_")[3].replace(".hdf5", "")),
                    reverse=True,
                )

            model_file = os.path.join(
                warm_start, "kfold_{}".format(fold), "model", model_list[-1]
            )

            #  Load model from file
            if learning_rate is None:
                full_model = load_model(
                    model_file,
                    custom_objects={
                        "softplus2": softplus2,
                        "Set2Set": Set2Set,
                        "mean_squared_error_with_scale": mean_squared_error_with_scale,
                        "MEGNetLayer": MEGNetLayer,
                    },
                )

                learning_rate = K.get_value(full_model.optimizer.lr)
            # Set up model
            model = MEGNetModel(
                100,
                2,
                nblocks=args.n_blocks,
                nvocal=95,
                npass=args.n_pass,
                lr=learning_rate,
                loss=args.loss,
                graph_convertor=cg,
                is_classification=True if args.type == "classification" else False,
                nfeat_node=None if embedding_file is None else 16,
            )
            model.load_weights(model_file)
            initial_epoch = int(model_list[-1].split("_")[2])
            print(
                "warm start from : {}, \nlearning_rate is {}.".format(
                    model_file, learning_rate
                )
            )

        # Train
        model.train_from_graphs(
            train_graphs,
            train_targets,
            val_graphs,
            val_targets,
            batch_size=args.batch_size,
            epochs=args.max_epochs,
            verbose=2,
            initial_epoch=initial_epoch,
            use_multiprocessing=False if n_works <= 1 else True,
            workers=n_works,
            dirname=fold_model_path,
            y_scaler=y_scaler,
            save_best_only=args.save_best_only,
        )


if __name__ == "__main__":
    # Initialize GPU configuration
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

    train()
