# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import egg.core as core
import random
import numpy as np
import torch

def get_other_opts(parser):
    group = parser.add_argument_group("other")
    group.add_argument("--dummy", type=float, default=1.0, help="dumy option")


def get_opts(params):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--n_epochs_sv",
        type=int,
        default="100",
        help="total epochs to be trained during SV phase",
    )

    parser.add_argument(
        "--n_epochs_selfcomm",
        type=int,
        default="100",
        help="total epochs to be trained during Comm phase",
    )

    parser.add_argument(
        "--generations",
        type=int,
        default="0",
        help="total generations to be trained",
    )

    # Data opts
    parser.add_argument(
        "--language",
        type=str,
        default="fix",
        help="language type",
    )
    
    parser.add_argument(
        "--do_padding",
        type=bool,
        default=True,
        help="pad msg produced by speaker",
    )

    parser.add_argument(
        "--dataset_folder",
        type=str,
        default="data",
        help="Dataset location",
    )

    parser.add_argument(
        "--dataset_filename",
        type=str,
        default="meaning_phrase.txt",
        help="Dataset filename",
    )

    parser.add_argument(
        "--trainset_proportion",
        type=float,
        default=0.667,
        help="Dataset split",
    )

    parser.add_argument(
        "--comm_trainset_length",
        type=int,
        default=32,
        help="Dataset split",
    )

    parser.add_argument(
        "--log_dir",
        type=str,
        default='training_log/train.txt',
    )

    parser.add_argument(
        "--dump_output",
        type=bool,
        default=True,
    )

    parser.add_argument(
        "--outputs_dir",
        type=str,
        default='outputs',
    )

    parser.add_argument(
        "--dump_dir",
        type=str,
        default='dump',
    )

    parser.add_argument(
        "--dump_every",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--save_model_dir",
        type=str,
        default='saved_models',
    )

    parser.add_argument(
        "--num_workers", type=int, default=1, help="Workers used in the dataloader"
    )
    parser.add_argument(
        "--pdb",
        action="store_true",
        default=False,
        help="Run the game with pdb enabled",
    )

    parser.add_argument(
        "--meaning_embedding_dim", type=int, default=8, help="meaning embedding dim"
    )

    parser.add_argument(
        "--speaker_hidden_size", type=int, default=16, help="Speaker hidden size"
    )

    parser.add_argument(
        "--listener_hidden_size", type=int, default=16, help="Listener hidden size"
    )

    parser.add_argument(
        "--rnn", type=str, default='gru', help="rnn cell"
    )

    parser.add_argument(
        "--spk_max_len", type=int, default=10, help="spk decoder max len"
    )

    parser.add_argument(
        "--patience", type=int, default=10, help="early_stopping_patience"
    )

    parser.add_argument(
        "--acc_stopper_delayed", type=int, default=10, help="clever_stopping compared to {n_epochs} before"
    )
    parser.add_argument(
        "--minimal_epoch", type=int, default=10, help="clever_stopping check after {minimal_epoch} epoch"
    )
    parser.add_argument(
        "--drop_percentage", type=int, default=0.05, help="clever_stopping drop {percentage} as wait+1"
    )
    parser.add_argument(
        "--clever_stop_suffix", type=str, default='', help="clever_stopping suffix"
    )
    parser.add_argument(
        "--num_total_agent", type=int, default=2
    )
    parser.add_argument(
        "--num_communications", type=int, default=2
    )
    parser.add_argument(
        "--spk_entropy_coeff", type=float, default=0.1
    )

    get_other_opts(parser)

    opts = core.init(arg_parser=parser, params=params)
    return opts


def cutting_length(message, receiver_output, labels):
    target_max_len = labels.size(1) - 1
    msg = message[:, :target_max_len]
    if receiver_output:
        receiver_out = receiver_output[:target_max_len]
    else:
        receiver_out = None
    return msg, receiver_out


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def update_trainer_start_epoch(trainer, new_start):
    trainer.start_epoch = new_start
