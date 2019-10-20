import argparse
import os
import json
import gzip
import pickle
import pandas as pd
import cgcnn
import torch
import torch.distributed as dist
from sklearn.model_selection import KFold, train_test_split
from matminer.featurizers.structure import CGCNNFeaturizer

__authors__ = "Qi Wang <qwang3@lbl.gov>"


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser("CGCNN Benchmark.")
    add_arg = parser.add_argument
    add_arg("-o", "--output_path", required=True, help="Output path")
    add_arg(
        "-i",
        "--input_file_prefix",
        default="../data/{}.pickle.gz",
        help="input file prefix",
    )
    add_arg(
        "-t",
        "--task",
        default="regression",
        help="task type, regression or classification",
    )
    add_arg("-d", "--dataset", default="jdft2d", help="pymatgen dataset name")
    add_arg("-p", "--property", default="exfoliation_en", help="Pridict propertys list")
    add_arg(
        "-ki",
        "--kf_indices",
        default="0,1,2,3,4",
        help='kfold indices str, split by "," if use more than one',
    )
    add_arg("-w", "--warm_start", default=None, help="warm start file")
    add_arg("-b", "--batch_size", default=128, type=int, help="mini batch size")
    add_arg("-n", "--n_works", type=int, default=1, help="n works for multiprocessing")
    add_arg("-m", "--max_epochs", default=300, type=int, help="Max epochs")
    # cgcnn model parameters
    add_arg("-nc", "--n_conv", default="4", type=int, help="n cgcnn convs")
    add_arg("-lr", "--learning_rate", default=0.02, type=float, help="learning rate")
    add_arg("-ts", "--test", action="store_true", help="predict test or not")
    add_arg("-pm", "--pin_memory", action="store_true", help="use pin_memory or not")
    add_arg("-dc", "--disable_cuda", action="store_true", help="disable cuda or not")
    add_arg("-dt", "--distributed", action="store_true", help="distributed or not")
    return parser.parse_args()


def train():
    # Parse args
    args = parse_args()
    print(args)

    output_path = args.output_path
    dataset_name = args.dataset
    prop_col = args.property
    disable_cuda = args.disable_cuda
    distributed = args.distributed
    kf_indices = list(map(int, args.kf_indices.split(",")))

    if not disable_cuda:
        print("Cuda is on? {}".format(torch.cuda.is_available()))
        print(
            "Cuda current device is : {}, it's name is : {}".format(
                torch.cuda.current_device(), torch.cuda.get_device_name(0)
            )
        )

    if distributed:
        dist.init_process_group(backend="mpi")

    atom_feature_path = os.path.join(
        os.path.dirname(cgcnn.__file__), "..", "data", "sample-classification"
    )

    with open(os.path.join(atom_feature_path, "atom_init.json")) as f:
        atom_features = json.load(f)

    tmp_output_path = os.path.join(output_path, dataset_name, prop_col)
    if not os.path.exists(tmp_output_path):
        os.makedirs(tmp_output_path, exist_ok=True)

    with gzip.open(args.input_file_prefix.format(dataset_name), "rb") as f:
        df = pd.DataFrame(pickle.load(f))[["structure", prop_col]].dropna()
    idx_list = list(range(len(df)))

    kf = KFold(n_splits=5, random_state=18012019, shuffle=True)
    for kf_idx, (remain_index, test_index) in enumerate(kf.split(idx_list)):
        if kf_idx in kf_indices:
            kf_tmp_output_path = os.path.join(
                tmp_output_path, "kfold_{}".format(kf_idx)
            )
            if not os.path.exists(kf_tmp_output_path):
                os.makedirs(kf_tmp_output_path, exist_ok=True)
            train_index, val_index = train_test_split(
                remain_index, test_size=0.25, random_state=18012019, shuffle=True
            )

            cgcnnfz = CGCNNFeaturizer(
                task=args.task,
                distributed=distributed,
                n_works=args.n_works,
                disable_cuda=disable_cuda,
                save_idx=kf_tmp_output_path,
                output_path=kf_tmp_output_path,
                atom_init_fea=atom_features,
                use_batch=False,
                test=args.test,
                dropout_percent=0.5,
                batch_size=args.batch_size,
                warm_start_file=args.warm_start,
                warm_start_latest=True,
                use_pretrained=False,
                save_model_to_dir=os.path.join(kf_tmp_output_path, "model"),
                save_checkpoint_to_dir=os.path.join(kf_tmp_output_path, "checkpoint"),
                checkpoint_interval=10,
                num_epochs=args.max_epochs,
                print_freq=10,
                optim="SGD",
                momentum=0.9,
                lr_milestones=[800],
                lr=args.learning_rate,
                weight_decay=0.0,
                h_fea_len=32,
                pin_memory=args.pin_memory,
                n_conv=args.n_conv,
                n_h=1,
                del_checkpoint=False,
                use_idxes=True,
                train_set=train_index,
                val_set=val_index,
                test_set=test_index,
                atom_fea_len=64,
                log_dir=os.path.join(kf_tmp_output_path, "logger.log"),
                simple_log_dir=os.path.join(kf_tmp_output_path, "simple_logger.log"),
            )

            cgcnnfz.fit(X=df["structure"], y=df[prop_col])


if __name__ == "__main__":
    train()
