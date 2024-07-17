# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import sys
sys.path.insert(0, './pytorch-seq2seq/')

import os
# ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

from pathlib import Path
ROOT_DIR = Path(__file__).absolute().parent.parent.parent.parent
sys.path.insert(1, str(ROOT_DIR))

from datetime import datetime, timedelta
from typing import List

import pandas as pd
import itertools
from itertools import product, permutations, combinations
from argparse import Namespace

import torch
import torch.nn as nn

import egg.core as core
from egg.zoo.nellcom_x.data import get_dataloader, MyDataset, my_selected_split, my_selected_resplit, my_split_multilang
from egg.zoo.nellcom_x.game_callbacks import get_callbacks, v2_get_callbacks, v3_get_callbacks_no_earlystop, \
    v4_get_callbacks_cleverstopper, v5_get_callbacks

from egg.zoo.nellcom_x.game_callbacks import ConsoleLogSaver

from egg.zoo.nellcom_x.games import build_game_after_supervised
from egg.zoo.nellcom_x.games_comm import build_game_comm_spk, build_game_comm_lst, v2_build_game_comm_spk, \
    tied_build_game_comm_lst, tied_v2_build_game_comm_spk
from egg.zoo.nellcom_x.utils import get_opts, set_seed, update_trainer_start_epoch
from egg.zoo.nellcom_x.utils_extra import load_iter_datasets, load_model, teacher_dataset_dump, create_basic_split, \
    set_model_saver_saving_mode

from egg.zoo.nellcom_x.archs import SpeakerListener

import pickle as pkl

import wandb
import random
import numpy as np
import math

# USING_EARLYSTOP = False
USING_EARLYSTOP = True

# wandb.init(project="my-test-project", entity="ylian")

from egg.zoo.nellcom_x.utils_extra import load_f_df, get_dump, multi_run, arrange_dump_v2


class My_Agent_tied():
    def __init__(self, rand_seed=1000, suffix='', word_embedding_dim=128, meaning_embedding_dim=16, name='agent_', generation=0,
                 dataset_filename='meaning_phrase.txt', parent=None, opts=None, tying_embeddings=True, listener_type='with_embedding', lst_word_embedding_dim=128):
        self.name = name
        self.rand_seed = rand_seed
        set_seed(self.rand_seed)

        self.path_dict = self.setup_fpath(opts, suffix, dataset_filename)
        self.comm_dict = dict()
        self.self_comm_dict = dict()
        self.self_comm_dict.update({-1: self.path_dict})

        self.language = opts.language
        self.generation = generation
        self.parent_randseed = parent.rand_seed if generation !=0 else 0
        self.suffix = suffix

        self.train_loader_test, self.test_loader_test, self.full_dataset = self.setup_data_loader(opts)
        meaning_vocab_size, uttr_vocab_size = self.full_dataset.get_vocab_size()
        # simplify later
        if tying_embeddings:
            assert lst_word_embedding_dim == word_embedding_dim, "tied agents should have unique word_embedding_dim!!"
        self.word_embedding_dim = word_embedding_dim
        self.meaning_embedding_dim = meaning_embedding_dim

        self.listener_hidden_size = lst_word_embedding_dim
        self.speaker_hidden_size = word_embedding_dim

        self.shared_word_embedding = nn.Embedding(uttr_vocab_size, word_embedding_dim)
        self.shared_meaning_embedding = nn.Embedding(meaning_vocab_size, meaning_embedding_dim)

        self.tied = tying_embeddings
        if not self.tied:
            # self.shared_xxx_embedding as spk_embedding
            self.lst_word_embedding = nn.Embedding(uttr_vocab_size, lst_word_embedding_dim)
            self.lst_meaning_embedding = nn.Embedding(meaning_vocab_size, meaning_embedding_dim)

        self.lst_type = listener_type
        # ['with_embedding', 'tacl', 'relu'], default='with_embedding'

        self.listener = None
        self.speaker = None
        self.game_lst = None
        self.game_spk = None

        self.eval_trainer = None
        self.activate_round_counter = 0
        self.total_selfcomm_during_interactive = 0

    def set_sv_games(self, lst_game, spk_game):
        self.game_lst = lst_game
        self.game_spk = spk_game

    def set_speaker(self, speaker):
        self.speaker = speaker

    def set_listener(self, listener):
        self.listener = listener

    def setup_fpath(self, opts, suffix, dataset_filename):
        save_model_dir = os.path.join(opts.save_model_dir, suffix)
        if not os.path.exists(save_model_dir):
            os.mkdir(save_model_dir)

        log_dir_lst = opts.log_dir.split('.')[0] + '_' + suffix + '_lst.txt'
        outputs_dir_lst = os.path.join(opts.outputs_dir, suffix + '_lst')
        if not os.path.exists(outputs_dir_lst):
            os.mkdir(outputs_dir_lst)
        dump_dir_lst = os.path.join(opts.dump_dir, suffix + '_lst')
        if not os.path.exists(dump_dir_lst):
            os.mkdir(dump_dir_lst)

        log_dir_spk = opts.log_dir.split('.')[0] + '_' + suffix + '_spk.txt'
        outputs_dir_spk = os.path.join(opts.outputs_dir, suffix + '_spk')
        if not os.path.exists(outputs_dir_spk):
            os.mkdir(outputs_dir_spk)
        dump_dir_spk = os.path.join(opts.dump_dir, suffix + '_spk')
        if not os.path.exists(dump_dir_spk):
            os.mkdir(dump_dir_spk)

        log_dir_comm = opts.log_dir.split('.')[0] + '_' + suffix + '_comm.txt'
        outputs_dir_comm = os.path.join(opts.outputs_dir, suffix + '_comm')
        if not os.path.exists(outputs_dir_comm):
            os.mkdir(outputs_dir_comm)
        dump_dir_comm = os.path.join(opts.dump_dir, suffix + '_comm')
        if not os.path.exists(dump_dir_comm):
            os.mkdir(dump_dir_comm)

        save_splits_dir = os.path.join('splits', suffix)
        if not os.path.exists(save_splits_dir):
            os.mkdir(save_splits_dir)

        dataset_dir = os.path.join(opts.dataset_folder, opts.language, dataset_filename)

        path_dict = {'save_model_dir':save_model_dir, 'save_splits_dir':save_splits_dir, 'dataset_dir':dataset_dir,
                     'log_dir_lst':log_dir_lst, 'log_dir_spk':log_dir_spk, 'log_dir_comm':log_dir_comm,
                     'outputs_dir_lst':outputs_dir_lst, 'outputs_dir_spk':outputs_dir_spk, 'outputs_dir_comm':outputs_dir_comm,
                     'dump_dir_lst':dump_dir_lst, 'dump_dir_spk':dump_dir_spk, 'dump_dir_comm':dump_dir_comm}

        return path_dict

    def update_data_loader(self, train_loader_test, test_loader_test, full_dataset):
        self.train_loader_test = train_loader_test
        self.test_loader_test = test_loader_test
        self.full_dataset = full_dataset

    def resplit_train_combo(self, train_d_combo, test_d, full_dataset, opts, external_seed=None, partner_seed=None, use_comm_trainset_length=False):

        training_data_size = opts.trainset_proportion

        train_size = opts.comm_trainset_length if use_comm_trainset_length else int(training_data_size * len(full_dataset))
        dev_size = len(train_d_combo) - train_size

        selected = False if use_comm_trainset_length else True

        if external_seed is None:
            set_seed(self.rand_seed)
            print(f'external_seed is None, resplit using self seed:{self.rand_seed}')
        else:
            set_seed(external_seed)
            print(f'resplit using seed: {external_seed}')

        # train_d_resampled, dev_d = my_selected_resplit(train_d_combo, [train_size, dev_size], selected=selected)
        # # print(self.name)
        # # print("resplit indices")
        # # print(train_d_resampled.indices[:10])

        # new_fulldataset = MyDataset(csv_file=self.path_dict['dataset_dir'], language=opts.language, gen=self.generation)
        train_d_combo, test_d = my_split_multilang(self.full_dataset, [train_size + dev_size, len(test_d)], test_d)
        train_d_resampled, dev_d = my_selected_resplit(train_d_combo, [train_size, dev_size], selected=selected)

        # to save datasplits
        if partner_seed is not None:
            save_splits_dir = self.comm_dict[partner_seed]['save_splits_dir']
        else:
            save_splits_dir = self.path_dict['save_splits_dir']
        # print(save_splits_dir)
        torch.save(train_d_resampled, f"{save_splits_dir}/train.pkl")
        torch.save(test_d, f"{save_splits_dir}/test.pkl")
        torch.save(dev_d, f"{save_splits_dir}/dev.pkl")
        torch.save(self.full_dataset, f"{save_splits_dir}/full.pkl")

        train_loader_test = get_dataloader(
            train_dataset=train_d_resampled,
            batch_size=opts.batch_size,
            num_workers=opts.num_workers,
            is_distributed=opts.distributed_context.is_distributed,
            seed=self.rand_seed if external_seed is None else external_seed,
        )
        test_loader_test = get_dataloader(
            train_dataset=test_d,
            batch_size=len(test_d),
            num_workers=opts.num_workers,
            is_distributed=opts.distributed_context.is_distributed,
            seed=self.rand_seed if external_seed is None else external_seed,
            drop_last=False,
            shuffle=False,
        )
        self.update_data_loader(train_loader_test, test_loader_test, self.full_dataset)
        return

    def setup_data_loader(self, opts):
        full_dataset = MyDataset(csv_file=self.path_dict['dataset_dir'], language=opts.language, gen=self.generation)

        training_data_size = opts.trainset_proportion

        train_size = int(training_data_size * len(full_dataset))
        test_size = len(full_dataset) - train_size

        # train_d, test_d = torch.utils.data.random_split(full_dataset, [train_size, test_size])
        train_d, test_d = my_selected_split(full_dataset, [train_size, test_size], selected=True)
        # selected=True:    all elements appear in the train set,
        #                   train_size MUST sent first!!!

        # to save datasplits
        save_splits_dir = self.path_dict['save_splits_dir']
        torch.save(train_d, f"{save_splits_dir}/train.pkl")
        torch.save(test_d, f"{save_splits_dir}/test.pkl")
        torch.save(full_dataset, f"{save_splits_dir}/full.pkl")

        train_loader_test = get_dataloader(
            train_dataset=train_d,
            batch_size=opts.batch_size,
            num_workers=opts.num_workers,
            is_distributed=opts.distributed_context.is_distributed,
            seed=self.rand_seed,
        )
        test_loader_test = get_dataloader(
            train_dataset=test_d,
            batch_size=len(test_d),
            num_workers=opts.num_workers,
            is_distributed=opts.distributed_context.is_distributed,
            seed=self.rand_seed,
            drop_last=False,
            shuffle=False,
        )
        return train_loader_test, test_loader_test, full_dataset

    def new_init_model(self, opts):

        if self.tied:
            game_lst = tied_build_game_comm_lst(
                train_data=self.full_dataset,
                encoder_hidden_size=self.listener_hidden_size,
                meaning_embedding_dim=self.meaning_embedding_dim,
                is_distributed=opts.distributed_context.is_distributed,
                rnn_cell=opts.rnn,
                shared_word_embedding=self.shared_word_embedding,
                shared_meaning_embedding=self.shared_meaning_embedding,
                lst_type=self.lst_type
            )
        else:
            game_lst = tied_build_game_comm_lst(
                train_data=self.full_dataset,
                encoder_hidden_size=self.listener_hidden_size,
                meaning_embedding_dim=self.meaning_embedding_dim,
                is_distributed=opts.distributed_context.is_distributed,
                rnn_cell=opts.rnn,
                shared_word_embedding=self.lst_word_embedding,
                shared_meaning_embedding=self.lst_meaning_embedding,
                lst_type=self.lst_type
            )

        game_spk = tied_v2_build_game_comm_spk(
            train_data=self.full_dataset,
            meaning_embedding_size=self.meaning_embedding_dim,
            decoder_hidden_size=self.speaker_hidden_size,
            is_distributed=opts.distributed_context.is_distributed,
            rnn_cell=opts.rnn,
            spk_max_len=opts.spk_max_len,
            shared_word_embedding=self.shared_word_embedding,
            shared_meaning_embedding=self.shared_meaning_embedding
        )

        self.set_listener(game_lst.model)
        self.set_speaker(game_spk.model)
        self.set_sv_games(game_lst, game_spk)

        return

    def supervised_one_agent_trainer_init(self, opts):

        # begin = datetime.now() + timedelta(hours=9)
        # print(f"| STARTED JOB at {begin}...")

        optimizer_scheduler = None

        print()
        print(f'sl_listening_lr:')
        optimizer_lst = core.build_optimizer(self.game_lst.parameters())
        callbacks_lst = v3_get_callbacks_no_earlystop(log_dir=self.path_dict['log_dir_lst'], acc_threshhold=0.999, patience=opts.patience,
                                                      dump_output=opts.dump_output, outputs_dir=self.path_dict['outputs_dir_lst'],
                                                      dump_every=opts.dump_every, save_model_dir=self.path_dict['save_model_dir'])

        print()
        print('sl_speaking_lr:')
        optimizer_spk = core.build_optimizer(self.game_spk.parameters())
        callbacks_spk = v3_get_callbacks_no_earlystop(log_dir=self.path_dict['log_dir_spk'], acc_threshhold=0.999, patience=opts.patience,
                                                      dump_output=opts.dump_output, outputs_dir=self.path_dict['outputs_dir_spk'],
                                                      dump_every=opts.dump_every, save_model_dir=self.path_dict['save_model_dir'])

        start_epoch = 0
        trainer_lst = core.Trainer(
            game=self.game_lst,
            optimizer=optimizer_lst,
            optimizer_scheduler=optimizer_scheduler,
            train_data=self.train_loader_test,
            validation_data=self.test_loader_test,
            callbacks=callbacks_lst,
            start_epoch=start_epoch
        )

        trainer_spk = core.Trainer(
            game=self.game_spk,
            optimizer=optimizer_spk,
            optimizer_scheduler=optimizer_scheduler,
            train_data=self.train_loader_test,
            validation_data=self.test_loader_test,
            callbacks=callbacks_spk,
            start_epoch=start_epoch
        )

        self.sv_trainer_spk = trainer_spk
        self.sv_trainer_lst = trainer_lst

        set_model_saver_saving_mode([self.sv_trainer_spk, self.sv_trainer_lst], mode='only_last')

        return

    def supervised_one_agent_train(self, opts, self_sv_spochs, sv_type='all'):

        sv_order = 'alter' if self.tied else 'old'

        if sv_type == 'all':
            if sv_order == 'alter':
                for i in range(self_sv_spochs):
                    start_epoch = i
                    update_trainer_start_epoch(self.sv_trainer_spk, start_epoch)
                    update_trainer_start_epoch(self.sv_trainer_lst, start_epoch)

                    self.sv_trainer_lst.train(n_epochs=i + 1)
                    if opts.dump_output:
                        arrange_dump_v2(dataset=self.full_dataset, output_dir=self.path_dict['outputs_dir_lst'],
                                        dump_dir=self.path_dict['dump_dir_lst'], mode='lst')

                    self.sv_trainer_spk.train(n_epochs=i + 1)
                    if opts.dump_output:
                        arrange_dump_v2(dataset=self.full_dataset, output_dir=self.path_dict['outputs_dir_spk'],
                                        dump_dir=self.path_dict['dump_dir_spk'], mode='spk')

                    i = i + 1

            elif sv_order == 'old':
                print('sv old fashion: first lst then spk')
                start_epoch = 0
                self.sv_trainer_lst.train(n_epochs=self_sv_spochs)
                if opts.dump_output:
                    arrange_dump_v2(dataset=self.full_dataset, output_dir=self.path_dict['outputs_dir_lst'],
                                    dump_dir=self.path_dict['dump_dir_lst'], mode='lst')

                self.sv_trainer_spk.train(n_epochs=self_sv_spochs)
                if opts.dump_output:
                    arrange_dump_v2(dataset=self.full_dataset, output_dir=self.path_dict['outputs_dir_spk'],
                                    dump_dir=self.path_dict['dump_dir_spk'], mode='spk')

        elif sv_type == 'lst_only':
            print('sv train only listening')
            start_epoch = 0
            self.sv_trainer_lst.train(n_epochs=self_sv_spochs)
            if opts.dump_output:
                arrange_dump_v2(dataset=self.full_dataset, output_dir=self.path_dict['outputs_dir_lst'],
                                dump_dir=self.path_dict['dump_dir_lst'], mode='lst')

            self.sv_trainer_spk.train(n_epochs=0)
            if opts.dump_output:
                arrange_dump_v2(dataset=self.full_dataset, output_dir=self.path_dict['outputs_dir_spk'],
                                dump_dir=self.path_dict['dump_dir_spk'], mode='spk')

        elif sv_type == 'spk_only':
            print('sv train only speaking')
            start_epoch = 0
            self.sv_trainer_spk.train(n_epochs=self_sv_spochs)
            if opts.dump_output:
                arrange_dump_v2(dataset=self.full_dataset, output_dir=self.path_dict['outputs_dir_spk'],
                                dump_dir=self.path_dict['dump_dir_spk'], mode='spk')

            self.sv_trainer_lst.train(n_epochs=0)
            if opts.dump_output:
                arrange_dump_v2(dataset=self.full_dataset, output_dir=self.path_dict['outputs_dir_lst'],
                                dump_dir=self.path_dict['dump_dir_lst'], mode='lst')

        if self.game_spk.model.decoder.embedding.embedding_dim == self.game_lst.model.encoder.embedding.embedding_dim:
            total_embedding_parameters = self.game_spk.model.decoder.embedding.num_embeddings * self.game_spk.model.decoder.embedding.embedding_dim
            p = (
                    self.game_lst.model.encoder.embedding.weight.data == self.game_spk.model.decoder.embedding.weight.data).sum() == total_embedding_parameters

            # print(f'Listener encoder word embedding == Speaker decoder word embedding: {p.item()}')

            total_meaning_embedding_parameters = self.game_spk.model.encoder.embedding.num_embeddings * self.game_spk.model.encoder.embedding.embedding_dim
            p2 = (
                     self.game_lst.model.decoder.embedding.weight.data == self.game_spk.model.encoder.embedding.weight.data).sum() == total_meaning_embedding_parameters
            # print(f'Listener decoder meaning embedding == Speaker encoder meaning embedding: {p2.item()}')

            if not p.item() or not p2.item():
                print('No tying embedding between speaker and listener!!')
            else:
                print('tied embeddings between speaker and listener!!')
        else:
            print('No tying embedding between speaker and listener!! different word embedding dim!!')

        return

    def roll_back(self, stopped_at_epoch, path_dict, load_clever_stop=True):
        folder_name = path_dict['save_model_dir']
        fmodels = os.listdir(folder_name)
        files = [f for f in fmodels if f'_{stopped_at_epoch}' in f and 'Spk_Lst_' in f and '_cleverstop' not in f]

        if load_clever_stop and len(files) == 5:
            f_spk_enc = [f for f in files if 'spk_enc' in f].pop()
            self.speaker.encoder.load_state_dict(torch.load(os.path.join(folder_name, f_spk_enc)))
            f_spk_dec = [f for f in files if 'spk_dec' in f].pop()
            self.speaker.decoder.load_state_dict(torch.load(os.path.join(folder_name, f_spk_dec)))
            f_lst_enc = [f for f in files if 'lst_enc' in f].pop()
            self.listener.encoder.load_state_dict(torch.load(os.path.join(folder_name, f_lst_enc)))
            f_lst_dec = [f for f in files if 'lst_dec' in f].pop()
            self.listener.decoder.load_state_dict(torch.load(os.path.join(folder_name, f_lst_dec)))

            print(f'roll back model to epoch {stopped_at_epoch}')

    def load_model_tied(self, load_clever_stop=True, cleverstop_suffix='', load_teacher=True, load_sv=False):

        speaker_enc = self.speaker.encoder
        rl_speaker_dec = self.speaker.decoder
        listener_enc = self.listener.encoder
        listener_dec = self.listener.decoder

        folder_name = self.path_dict['save_model_dir']

        if load_teacher:

            fmodels = os.listdir(folder_name)
            files = [f for f in fmodels if cleverstop_suffix in f]

            if load_clever_stop and len(files) == 5:
                s = files[0]
                cleverstop_epoch = int(s[len('Spk_Lst_'):s.index('_cleverstop' + cleverstop_suffix)])
                f_spk_enc = [f for f in files if 'spk_enc' in f].pop()
                speaker_enc.load_state_dict(torch.load(os.path.join(folder_name, f_spk_enc)))
                f_spk_dec = [f for f in files if 'spk_dec' in f].pop()
                rl_speaker_dec.load_state_dict(torch.load(os.path.join(folder_name, f_spk_dec)))
                f_lst_enc = [f for f in files if 'lst_enc' in f].pop()
                listener_enc.load_state_dict(torch.load(os.path.join(folder_name, f_lst_enc)))
                f_lst_dec = [f for f in files if 'lst_dec' in f].pop()
                listener_dec.load_state_dict(torch.load(os.path.join(folder_name, f_lst_dec)))

                print(f'load teacher at gen {cleverstop_epoch}')
            else:
                speaker_enc.load_state_dict(torch.load(f'{folder_name}/Spk_Lst_final_spk_enc.pt'))
                rl_speaker_dec.load_state_dict(torch.load(f'{folder_name}/Spk_Lst_final_spk_dec.pt'))
                listener_enc.load_state_dict(torch.load(f'{folder_name}/Spk_Lst_final_lst_enc.pt'))
                listener_dec.load_state_dict(torch.load(f'{folder_name}/Spk_Lst_final_lst_dec.pt'))

                print(f'load teacher at _final_ epoch')

            game_type = 'comm'

        elif load_sv:
            speaker_enc.load_state_dict(torch.load(f'{folder_name}/Speaker_final_enc.pt'))
            rl_speaker_dec.load_state_dict(torch.load(f'{folder_name}/Speaker_final_dec.pt'))
            listener_enc.load_state_dict(torch.load(f'{folder_name}/Listener_final_enc.pt'))
            listener_dec.load_state_dict(torch.load(f'{folder_name}/Listener_final_dec.pt'))
            game_type = 'speak'

        if game_type == 'speak':
            train_mode = 'supervised'
        elif game_type == 'listen':
            train_mode = 'supervised'
        elif game_type == 'comm':
            train_mode = 'reinforce'

        rl_speaker_dec.set_train_mode(train_mode)

        return

    def self_comm(self, opts, self_comm_epochs, eval=False, at_n_epoch=None, inter_comm_turn=-1):

        if self_comm_epochs == 0 or eval:
            print(f'\n**********selfcomm eval {self.name}')

            # self.eval_trainer.start_epoch=at_n_epoch
            self.eval_trainer.eval_customed(epoch=at_n_epoch)

            # validation_loss, validation_interaction = self.eval_trainer.eval()
            #
            # for callback in self.eval_trainer.callbacks:
            #     callback.on_validation_end(
            #         validation_loss, validation_interaction, at_n_epoch
            #     )
            #     if isinstance(callback, ConsoleLogSaver):
            #         aggregate_log = callback.aggregate_log
            #         df = pd.DataFrame.from_dict(aggregate_log)
            #         df.to_csv(callback.log_dir, sep='\t')
            #         print(f'Log file saved to {callback.log_dir}')

        else:
            print(f'\n**********selfcomm train {self.name}')
            update_trainer_start_epoch(self.eval_trainer, at_n_epoch)
            self.eval_trainer.train(n_epochs=self_comm_epochs)

        if opts.dump_output:
            arrange_dump_v2(dataset=self.full_dataset, output_dir=self.self_comm_dict[inter_comm_turn]['outputs_dir_comm'],
                            dump_dir=self.self_comm_dict[inter_comm_turn]['dump_dir_comm'], mode='comm')

        return

    def self_play_trainer_init(self, opts, inter_comm_round=None, inter_comm_turn=None, use_early_stopping=True):
        # self play
        game_selfplay = build_game_after_supervised(
            opts,
            speaker=self.speaker,
            listener=self.listener,
            train_data=self.full_dataset,
            meaning_embedding_size=self.meaning_embedding_dim,
            encoder_hidden_size=self.listener_hidden_size,
            decoder_hidden_size=self.speaker_hidden_size,
            is_distributed=opts.distributed_context.is_distributed,
            game_type='commu'
        )

        print()
        print('self_play_lr:')
        optimizer = core.build_optimizer(game_selfplay.parameters())
        optimizer_scheduler = None

        log_dir = self.path_dict['log_dir_comm']
        outputs_dir = self.path_dict['outputs_dir_comm']
        save_model_dir = self.path_dict['save_model_dir']

        if inter_comm_round is not None:
            new_path_dict = dict()
            for k, v_ in self.path_dict.items():
                if 'comm' in k or 'save_model_dir' == k or 'save_splits_dir' == k:
                    # if self.generation != 0:
                    #     new_v = v_.replace(str(self.rand_seed), str(self.rand_seed) + '+' + str(agent2.rand_seed))
                    # else:
                    #     new_v = v_.replace(str(self.rand_seed), str(self.rand_seed) + '+' + str(agent2.rand_seed))

                    v = v_

                    if 'round' in v:
                        tmp = v[v.find('round'):].find('_') + v.find('round')
                        new_v = v[:v.find('round')] + f'round{inter_comm_round}' + v[tmp:]
                        tmp = new_v[new_v.find('turn'):].find('_') + new_v.find('turn')
                        new_v = new_v[:new_v.find('turn')] + f'turn{inter_comm_turn}' + new_v[tmp:]
                    else:
                        tmp = v[v.find('seed'):].find('_')
                        if tmp != -1:
                            new_v = v[:(tmp + v.find('seed'))] + f'_round{inter_comm_round}_turn{inter_comm_turn}' + v[(tmp + v.find('seed')):]
                        else:
                            new_v = v + f'_round{inter_comm_round}_turn{inter_comm_turn}'

                    new_path_dict.update({k: new_v})
                    if ('log' not in k) and (not os.path.exists(new_v)):
                        os.mkdir(new_v)
            self.self_comm_dict.update({inter_comm_turn: new_path_dict})

            log_dir = new_path_dict['log_dir_comm']
            outputs_dir = new_path_dict['outputs_dir_comm']
            save_model_dir = new_path_dict['save_model_dir']

        callbacks = v5_get_callbacks(log_dir=log_dir, outputs_dir=outputs_dir, save_model_dir=save_model_dir,
                                     using_earlystop=use_early_stopping,
                                     acc_stopper_delayed=opts.acc_stopper_delayed, minimal_epoch=opts.minimal_epoch,
                                     drop_percentage=opts.drop_percentage, clever_stop_suffix=opts.clever_stop_suffix,
                                     acc_threshhold=0.999, patience=opts.patience, dump_output=opts.dump_output,
                                     dump_every=opts.dump_every)

        trainer_selfplay = core.Trainer(
            game=game_selfplay,
            optimizer=optimizer,
            optimizer_scheduler=optimizer_scheduler,
            train_data=self.train_loader_test,
            validation_data=self.test_loader_test,
            callbacks=callbacks,
        )

        self.eval_trainer = trainer_selfplay

    def prepare_interactive_comm_speaking(self, agent2, opts, combo_train_d, test_d, overall_full_dataset, overall_seed,
                                          inter_comm_round=None, inter_comm_turn=None, use_comm_trainset_length=False):
        # begin = datetime.now() + timedelta(hours=9)
        # print(f"| STARTED JOB at {begin}...")
        # interactive play

        new_path_dict = dict()
        for k, v_ in self.path_dict.items():
            if 'comm' in k or 'save_model_dir' == k or 'save_splits_dir' == k:
                if self.generation != 0:
                    new_v = v_.replace(str(self.rand_seed), str(self.rand_seed) + '+' + str(agent2.rand_seed))
                else:
                    new_v = v_.replace(str(self.rand_seed), str(self.rand_seed) + '+' + str(agent2.rand_seed))

                v = new_v

                if inter_comm_round is not None:
                    if 'round' in v:
                        tmp = v[v.find('round'):].find('_') + v.find('round')
                        new_v = v[:v.find('round')] + f'round{inter_comm_round}' + v[tmp:]
                        tmp = new_v[new_v.find('turn'):].find('_') + new_v.find('turn')
                        new_v = new_v[:new_v.find('turn')] + f'turn{inter_comm_turn}' + new_v[tmp:]
                    else:
                        tmp = v[v.find('seed'):].find('_')
                        if tmp != -1:
                            new_v = v[:(tmp + v.find('seed'))] + f'_round{inter_comm_round}_turn{inter_comm_turn}' + v[(tmp + v.find('seed')):]
                        else:
                            new_v = v + f'_round{inter_comm_round}_turn{inter_comm_turn}'

                new_path_dict.update({k: new_v})
                if ('log' not in k) and (not os.path.exists(new_v)):
                    os.mkdir(new_v)
        self.comm_dict.update({agent2.rand_seed: new_path_dict})

        game_interactiveplay = build_game_after_supervised(
            opts,
            speaker=self.speaker,
            listener=agent2.listener,
            train_data=self.full_dataset,
            meaning_embedding_size=self.meaning_embedding_dim,
            encoder_hidden_size=self.listener_hidden_size,
            decoder_hidden_size=self.speaker_hidden_size,
            is_distributed=opts.distributed_context.is_distributed,
            game_type='commu'
        )

        print()
        print('interactive_lr:')
        optimizer = core.build_optimizer(game_interactiveplay.parameters())
        optimizer_scheduler = None

        callbacks = v5_get_callbacks(log_dir=new_path_dict['log_dir_comm'],
                                     outputs_dir=new_path_dict['outputs_dir_comm'],
                                     save_model_dir=new_path_dict['save_model_dir'],
                                     using_earlystop=USING_EARLYSTOP, acc_stopper_delayed=opts.acc_stopper_delayed,
                                     minimal_epoch=opts.minimal_epoch,
                                     drop_percentage=opts.drop_percentage, clever_stop_suffix=opts.clever_stop_suffix,
                                     acc_threshhold=0.999, patience=opts.patience, dump_output=opts.dump_output,
                                     dump_every=opts.dump_every)

        self.resplit_train_combo(combo_train_d, test_d, overall_full_dataset, opts, overall_seed, agent2.rand_seed, use_comm_trainset_length)

        start_epoch = 0
        trainer_interactiveplay = core.Trainer(
            game=game_interactiveplay,
            optimizer=optimizer,
            optimizer_scheduler=optimizer_scheduler,
            train_data=self.train_loader_test,
            validation_data=self.test_loader_test,
            callbacks=callbacks,
            start_epoch=start_epoch
        )

        return trainer_interactiveplay

    def run_interactive_comm_speaking(self, trainer, opts, start_epoch, interactive_comm_epochs, partner, freeze_listener=False, freeze_speaker=False):

        new_path_dict = self.comm_dict[partner.rand_seed]
        update_trainer_start_epoch(trainer, start_epoch)
        if freeze_listener:
            trainer.game.model.listener.requires_grad_(False)
        if freeze_speaker:
            trainer.game.model.speaker.requires_grad_(False)
        trainer.train(n_epochs=interactive_comm_epochs)

        if opts.dump_output:
            arrange_dump_v2(dataset=self.full_dataset, output_dir=new_path_dict['outputs_dir_comm'],
                            dump_dir=new_path_dict['dump_dir_comm'], mode='comm')

        return


class Scenario_setup(object):
    def __init__(self, tying_embedding=True, lst_type='with_embedding', generation=0, group_size=2, phase0_selfcomm=True, phase0_only=False, phase1_selfcomm=True, phase1_onedir=False,
                 gold_lst=False, phase1_lst_freeze=False, gold_spk=False, phase1_spk_freeze=False):
        self.tying_embedding = tying_embedding
        self.lst_type = lst_type
        self.generation = generation
        self.group_size = group_size
        self.gold_lst = gold_lst
        # if use gold_lst: auto-set phase0_svlst_only=T & phase1_lst_freeze=T & phase1_onedir=T
        self.gold_spk = gold_spk
        # if use gold_spk: auto-set phase0_svspk_only=T & phase1_spk_freeze=T & phase1_onedir=T

        self.phase0_selfcomm = phase0_selfcomm
        # during phase0, whether run selfcomm following the SL training
        self.phase1_selfcomm = phase1_selfcomm
        # during phase1, whether run selfcomm following the interactive_comm

        self.phase0_svlst_only = False
        self.phase1_lst_freeze = phase1_lst_freeze
        # during phase1 interactive_comm, whether lst freeze
        self.phase0_svspk_only = False
        self.phase1_spk_freeze = phase1_spk_freeze
        # during phase1 interactive_comm, whether spk freeze

        self.phase1_onedir = phase1_onedir

        self.phase0_only = phase0_only

        if self.group_size != 2:
            self.phase1_onedir = False
            # onedir_speak only test in group of 2 case: only id=0 speak to id=1

        if gold_lst and self.group_size == 2:
            self.phase0_svlst_only = True
            self.phase1_lst_freeze = True
            self.phase1_onedir = True

        if gold_spk and self.group_size == 2:
            self.phase0_svspk_only = True
            self.phase1_spk_freeze = True
            self.phase1_onedir = True

        if self.group_size == 2:
            self.run_selfcomm_threshold = 1 if self.phase1_onedir else 2
            self.round_of_allcomm = 100
        elif self.group_size == 4:
            self.run_selfcomm_threshold = 3 if self.phase1_onedir else 6
            self.round_of_allcomm = 20
        elif self.group_size == 8:
            self.run_selfcomm_threshold = 7 if self.phase1_onedir else 14
            self.round_of_allcomm = 10
        elif self.group_size == 1:
            self.run_selfcomm_threshold = 1
            self.round_of_allcomm = 0
        else:
            self.run_selfcomm_threshold = self.group_size - 1 if self.phase1_onedir else 2 * (1-self.group_size)
            self.round_of_allcomm = 20

        if self.phase0_only:
            self.round_of_allcomm = 0


def main(params: List[str], suffix) -> None:
    begin = datetime.now() + timedelta(hours=9)
    print(f"| STARTED JOB at {begin}...")

    opts = get_opts(params=params)
    # opts.n_epochs = 10
    opts.n_epochs_sv = 60
    opts.n_epochs_comm = 60
    opts.num_workers = 0
    # opts.meaning_embedding_dim = 8
    opts.dump_every = 10
    opts.lr = 0.01
    # opts.speaker_hidden_size = 128
    # opts.listener_hidden_size = 128
    opts.do_padding = True # standard with padding

    opts.rnn = 'gru'

    opts.patience = 10 if 'free' in opts.language else 5
    opts.acc_stopper_delayed = 5
    opts.patience = 3
    opts.minimal_epoch = 20
    opts.drop_percentage = 0.1
    # opts.clever_stop_suffix = f'_minepo{opts.minimal_epoch}_patience{opts.patience}_delay{opts.acc_stopper_delayed}_drop{int(opts.drop_percentage * 100)}'
    opts.clever_stop_suffix = ''

    # opts.generations = 5
    opts.generations = 0
    opts.n_epochs_sv = 60
    opts.n_epochs_phase0_selfcomm = 0
    phase0_early_stopping = False
    opts.n_epochs_phase1_interactivecomm = 1
    opts.n_epochs_phase1_selfcomm = 1
    epoch_step = opts.n_epochs_phase1_interactivecomm

    sl_validation_freq = 10
    rl_validation_freq = 1
    
    rl_lr = 0.01
    sl_lr = 0.01

    if opts.random_seed in [99020000 + 1000 * i for i in range(0, 100)]:
        rl_lr = 0.005
    # lr 7e-3: 0-10, 40-50!!, 70-80
    # lr 5e-3: 10-20, 80-90
    # lr 3e-3: 20-30, 60-70
    # lr 1e-3: 30-40
    # lr 1e-2: 40-60!!mv 40-50 to bk
    # current: 80-90

    opts.num_total_agent = 2

    assinged_lang_types_candidates = [['meaning_phrase_mk50%_osv50%_v1.txt', 'meaning_phrase_mk50%_osv50%_v2.txt', 'meaning_phrase_mk50%_osv50%_v3.txt', 'meaning_phrase_mk50%_osv50%_v4.txt', 'meaning_phrase_mk50%_osv50%.txt'], 
            ['meaning_phrase_mk20%_osv80%_v1.txt', 'meaning_phrase_mk20%_osv80%_v2.txt', 'meaning_phrase_mk20%_osv80%_v3.txt', 'meaning_phrase_mk20%_osv80%_v4.txt', 'meaning_phrase_mk20%_osv80%.txt']]
    assinged_lang_types = []
    for i in range(opts.num_total_agent):
        idx = random.randint(0, len(assinged_lang_types_candidates[i])-1)
        assinged_lang_types.append(assinged_lang_types_candidates[i][idx])

    


    print('pre-assigned languages:')
    print(assinged_lang_types)
    
    opts.spk_max_len = 10
    opts.spk_entropy_coeff = 0.1
    saving_model_interval = 2

    tying_embedding = True
    lst_type = 'with_embedding'
    # ['with_embedding', 'tacl', 'relu'], default='with_embedding'
    phase0_selfcomm = False
    phase0_only = False

    phase1_selfcomm = True
    phase1_onedir = False
    gold_lst = False
    phase1_lst_freeze = False
    gold_spk = False
    phase1_spk_freeze = False

    # default: {generation = 0, group_size = 2, phase0_selfcomm = True, phase1_selfcomm = True, gold_lst = False, phase1_lst_freeze = False, phase1_onedir = False}
    scenario = Scenario_setup(tying_embedding=tying_embedding, lst_type=lst_type, generation=opts.generations,
                              group_size=opts.num_total_agent, phase0_selfcomm=phase0_selfcomm, phase0_only=phase0_only,
                              phase1_selfcomm=phase1_selfcomm, phase1_onedir=phase1_onedir,
                              gold_lst=gold_lst, phase1_lst_freeze=phase1_lst_freeze,
                              gold_spk=gold_spk, phase1_spk_freeze=phase1_spk_freeze)

    # # test default scenatio
    # scenario = Scenario_setup()
    

    # trainset_proprtion = 0.667
    # opts.trainset_proportion = 0.334
    # assinged in main (suffix split0.667 vs split0.334);
    # !!! remember to change ana_final files when generating acc/count/types
    # !!! remember to run trainset_proprtion = 0.667 for 990000, 991000 !!!

    opts.trainset_proportion = 0.667
    trainset_proprtion = opts.trainset_proportion
    opts.comm_trainset_length = 32 * 10 
    # if opts.random_seed in [11420000 + 1000 * i for i in range(10, 20)]:
    scenario.run_selfcomm_threshold = 5 * scenario.run_selfcomm_threshold
    print(f'run_selfcomm_threshhold: {scenario.run_selfcomm_threshold}')

    # # TEST HYPERPARAMETER
    # opts.n_epochs_sv = 2
    # opts.n_epochs_phase0_selfcomm = 3
    # sl_validation_freq = 1
    # rl_validation_freq = 1
    # # opts.n_epochs_phase1_interactivecomm = 10
    # # opts.n_epochs_phase1_selfcomm = 10
    # # epoch_step = opts.n_epochs_phase1_interactivecomm
    # # scenario.round_of_allcomm = 5

    selfcomm_use_earlystopping_interactively = False
    # for selfcomm during interactive
    group_id = opts.random_seed
    num_total_agent = opts.num_total_agent
    print(f'group seed: {group_id}')
    # set total agent number in the group
    corresponse_seeds = [group_id + i for i in range(num_total_agent)]

    print(f"{opts}\n")
    if not opts.distributed_context.is_distributed and opts.pdb:
        breakpoint()

    print(f'round_of_allcomm: {scenario.round_of_allcomm}')
    total_connections = list(itertools.permutations(range(num_total_agent), 2))
    if scenario.group_size == 2 and scenario.phase1_onedir:
        total_connections = [(0, 1)]
    num_communications = len(total_connections) * scenario.round_of_allcomm

    # sample = True
    # # Option 1: randomly choose
    # selected_communications = random.choices(
    #     total_connections, k=num_communications) if sample else total_connections
    #
    # if num_communications <= len(total_connections):
    #     random.seed(group_id)
    #     selected_communications = total_connections.copy()
    #     random.shuffle(selected_communications)
    #     selected_communications = selected_communications[:num_communications]
    #
    # print(selected_communications)

    # Option 2:
    # repeat(shuffle(total_connections))
    i_ = 0
    selected_communications = []
    random.seed(group_id)
    if scenario.round_of_allcomm != 0:
        while i_ <= math.ceil(num_communications/len(total_connections)):
            i_ = i_ + 1
            selected_ = total_connections.copy()
            random.shuffle(selected_)
            selected_communications.extend(selected_)
        selected_communications = selected_communications[:num_communications]
    print(selected_communications)

    # option 1: spk -> lst only
    # option 2: both direction
    comm_option = 'one_dir'
    # comm_option = 'both_dir'

    # assign group languages, agents can be assgined with various languges
    opts.dataset_folder = 'data_expand'
    dataset_files = os.listdir(os.path.join(opts.dataset_folder, opts.language))
    find_string = lambda x, y: int(x[x.find(y) + len(y): x[x.find(y):].find('%') + x.find(y)]) if x.find(
        y) != -1 else None
    dataset_df = pd.DataFrame({'ds_name': dataset_files})
    dataset_df['mk'] = dataset_df.apply(lambda x: find_string(x['ds_name'], 'mk'), axis=1)
    dataset_df['osv'] = dataset_df.apply(lambda x: find_string(x['ds_name'], 'osv'), axis=1)

    i_ = 0
    selected_languages = []
    random.seed(group_id)
    while i_ <= math.ceil(num_total_agent / len(dataset_files)):
        i_ = i_ + 1
        selected_ = dataset_files.copy()
        random.shuffle(selected_)
        selected_languages.extend(selected_)
    selected_languages = selected_languages[:num_total_agent]
    # print(selected_languages)

    overall_seed = int(str(group_id)[:4])

    new_splits_proportions = [trainset_proprtion, 0.2]
    # train_proportion, test_proportion
    default_dataset_dir = os.path.join('data', opts.language, opts.dataset_filename)
    overall_full_dataset, combo_train_d, test_d = create_basic_split(default_dataset_dir, opts, group_id, splits_proportions=new_splits_proportions)

    agents = pd.DataFrame()
    # SL+RL AGENTS

    sprint = '****************** Phase 1: Agent Self SL+RL only ********************'
    print('\n' + '*' * len(sprint))
    print(sprint)
    print('*' * len(sprint))

    for id, seed in zip(range(num_total_agent), corresponse_seeds):
        a_name = f'agent_{id}'
        a_dataset = assinged_lang_types[id]
        a_suffix = suffix.replace(str(group_id), str(seed))
        print(seed)

        agent = My_Agent_tied(rand_seed=seed, suffix=a_suffix, word_embedding_dim=16, meaning_embedding_dim=8,
                               name=a_name, dataset_filename=a_dataset, opts=opts, tying_embeddings=scenario.tying_embedding, listener_type=scenario.lst_type, lst_word_embedding_dim=16)
        print(f'\n {agent.name}: seed = {agent.rand_seed} \n')

        # resplit a1 dataset
        agent.resplit_train_combo(combo_train_d, test_d, overall_full_dataset, opts, use_comm_trainset_length=False)

        agent.new_init_model(opts)

        print('model size:')
        print(agent.game_lst.model)
        print(agent.game_spk.model)
        print(f'****************** Agent {id} Self_train ********************')

        # # load instead of sv train
        # agent1.load_model_tied(load_teacher=False, load_sv=True, load_clever_stop=False)

        opts.lr = sl_lr
        opts.validation_freq = sl_validation_freq

        agent.supervised_one_agent_trainer_init(opts)

        if id == 0 and scenario.gold_spk and scenario.group_size == 2:
            sv_type = 'spk_only'
        elif id == 1 and scenario.gold_lst and scenario.group_size == 2:
            sv_type = 'lst_only'
        else:
            sv_type = 'all'
        agent.supervised_one_agent_train(opts, self_sv_spochs=opts.n_epochs_sv, sv_type=sv_type)

        # 1st round of self_comm: use_early_stopping always = True
        opts.lr = rl_lr
        opts.validation_freq = rl_validation_freq

        agent.self_play_trainer_init(opts, inter_comm_round=None, inter_comm_turn=None, use_early_stopping=phase0_early_stopping)
        start_epoch = 0
        if scenario.phase0_selfcomm:
            agent.self_comm(opts, self_comm_epochs=start_epoch+opts.n_epochs_phase0_selfcomm, eval=False, at_n_epoch=start_epoch, inter_comm_turn=-1)
            if agent.eval_trainer.should_stop:
                stopped_at_epoch = agent.eval_trainer.callbacks[2].stopped_at_epoch
                agent.roll_back(stopped_at_epoch, agent.path_dict)
        else:
            agent.self_comm(opts, self_comm_epochs=0, eval=True, at_n_epoch=start_epoch, inter_comm_turn=-1)

        agents = agents.append({'id': id, 'seed': seed, 'agent': agent}, ignore_index=True)

    agents.set_index('id')
    sprint = '****************** Phase 2: Interactive Agents Communication ********************'
    print('\n'+'*'*len(sprint))
    print(sprint)
    print('*'*len(sprint))

    opts.lr = rl_lr
    opts.validation_freq = rl_validation_freq

    comm_historys = []
    comm_id = 0

    comm_round = 0
    comm_turn = 0
    # saving_model_rounds = [0, round_of_allcomm-1]
    # saving_model_interval = 5
    saving_model_rounds = [i * saving_model_interval for i in range(scenario.round_of_allcomm // saving_model_interval + 1)] if scenario.round_of_allcomm > saving_model_interval else [0, scenario.round_of_allcomm-1]

    print(f'\n****************** Interactive_Comm Round {comm_round} ******************** \n')
    for comm_pair in selected_communications:
        agent1_id = comm_pair[0]   # agent1: speaker
        agent2_id = comm_pair[1]   # agent2: listener

        agent1 = agents.loc[agents['id']==agent1_id]['agent'].item()
        agent2 = agents.loc[agents['id']==agent2_id]['agent'].item()

        print(f'\n******************Interactive_Comm Turn {comm_turn}: {agent1.name} speak to {agent2.name} ******************** \n')

        # embed resplit into prepare_function
        agent1_comm_trainer = agent1.prepare_interactive_comm_speaking(agent2, opts, combo_train_d, test_d,
                                                                       overall_full_dataset, overall_seed + comm_turn * 10 + agent1_id * 100, inter_comm_round=comm_round, inter_comm_turn=comm_turn, use_comm_trainset_length=True)
        agent2_comm_trainer = agent2.prepare_interactive_comm_speaking(agent1, opts, combo_train_d, test_d,
                                                                       overall_full_dataset, overall_seed + comm_turn * 10 + agent2_id * 100, inter_comm_round=comm_round, inter_comm_turn=comm_turn, use_comm_trainset_length=True)

        agent1.self_play_trainer_init(opts, inter_comm_round=comm_round, inter_comm_turn=comm_turn, use_early_stopping=selfcomm_use_earlystopping_interactively)
        agent2.self_play_trainer_init(opts, inter_comm_round=comm_round, inter_comm_turn=comm_turn, use_early_stopping=selfcomm_use_earlystopping_interactively)

        if comm_round not in saving_model_rounds:
            set_model_saver_saving_mode([agent1_comm_trainer, agent2_comm_trainer, agent1.eval_trainer, agent2.eval_trainer], mode='None')

        start_epoch = 0
        while start_epoch < opts.n_epochs_phase1_interactivecomm:
            if opts.n_epochs_phase1_interactivecomm != 1:
                print(f'*********** Interactive Comm Epoch: {start_epoch + 1} *************')
            else:
                print(f'total interactive_comm_epochs = 1 *************')

            # new run_interactive: reuse predefined trainer
            agent1.run_interactive_comm_speaking(opts=opts, trainer=agent1_comm_trainer, start_epoch=start_epoch,
                                                 interactive_comm_epochs=start_epoch + epoch_step, partner=agent2,
                                                 freeze_listener=scenario.phase1_lst_freeze,
                                                 freeze_speaker=scenario.phase1_spk_freeze)

            if comm_option == 'both_dir':
                agent2.run_interactive_comm_speaking(opts=opts, trainer=agent2_comm_trainer, start_epoch=start_epoch,
                                                     interactive_comm_epochs=start_epoch + epoch_step, partner=agent1)

            if agent1_comm_trainer.should_stop or agent2_comm_trainer.should_stop:
                stopped_at_epoch = agent1_comm_trainer.callbacks[2].stopped_at_epoch if agent1_comm_trainer.callbacks[
                                                                                            2].stopped_at_epoch != -1 \
                    else agent2_comm_trainer.callbacks[2].stopped_at_epoch

                # option 1: stop when interactive_comm.should_stop == True, no self_comm followed
                agent1.roll_back(stopped_at_epoch, agent1.comm_dict[agent2.rand_seed])
                agent2.roll_back(stopped_at_epoch, agent2.comm_dict[agent1.rand_seed])

                # # option 2: after interactive_comm.should_stop == True, do one more turn of self_comm
                # if scenario.phase1_selfcomm:
                #     agent1.self_comm(opts, self_comm_epochs=stopped_at_epoch+epoch_step, eval=False, at_n_epoch=stopped_at_epoch)
                #     agent2.self_comm(opts, self_comm_epochs=stopped_at_epoch+epoch_step, eval=False, at_n_epoch=stopped_at_epoch)
                #     # agent1.roll_back(stopped_at_epoch, agent1.path_dict)
                #     # agent2.roll_back(stopped_at_epoch, agent2.path_dict)
                # else:
                #     agent1.roll_back(stopped_at_epoch, agent1.comm_dict[agent2.rand_seed])
                #     agent2.roll_back(stopped_at_epoch, agent2.comm_dict[agent1.rand_seed])

                break

            start_epoch = start_epoch + epoch_step

        start_epoch = 0
        if not scenario.gold_spk:
            agent1.activate_round_counter = agent1.activate_round_counter + 1
        if not scenario.gold_lst:
            agent2.activate_round_counter = agent2.activate_round_counter + 1
        if scenario.phase1_selfcomm:
            if agent1.activate_round_counter >= scenario.run_selfcomm_threshold:
                print(f'\n{agent1.name} activate_round_counter: {agent1.activate_round_counter} ')
                agent1.self_comm(opts, self_comm_epochs=opts.n_epochs_phase1_selfcomm + start_epoch, eval=False,
                                 at_n_epoch=start_epoch, inter_comm_turn=comm_turn)
                agent1.total_selfcomm_during_interactive = agent1.total_selfcomm_during_interactive + 1
                print(f'{agent1.name} total_selfcomm_during_interactive: {agent1.total_selfcomm_during_interactive}\n')
                agent1.activate_round_counter = 0
            else:
                agent1.self_comm(opts, self_comm_epochs=0, eval=True, at_n_epoch=start_epoch,
                                 inter_comm_turn=comm_turn)

            if agent2.activate_round_counter >= scenario.run_selfcomm_threshold:
                print(f'{agent2.name} activate_round_counter: {agent2.activate_round_counter} ')
                agent2.self_comm(opts, self_comm_epochs=opts.n_epochs_phase1_selfcomm + start_epoch, eval=False,
                                 at_n_epoch=start_epoch, inter_comm_turn=comm_turn)
                agent2.total_selfcomm_during_interactive = agent2.total_selfcomm_during_interactive + 1
                print(f'{agent2.name} total_selfcomm_during_interactive: {agent2.total_selfcomm_during_interactive}\n')
                agent2.activate_round_counter = 0
            else:
                agent2.self_comm(opts, self_comm_epochs=0, eval=True, at_n_epoch=start_epoch,
                                 inter_comm_turn=comm_turn)

        else:
            agent1.self_comm(opts, self_comm_epochs=0, eval=True, at_n_epoch=start_epoch, inter_comm_turn=comm_turn)
            agent2.self_comm(opts, self_comm_epochs=0, eval=True, at_n_epoch=start_epoch, inter_comm_turn=comm_turn)

        history = {comm_id: (agent1.name, agent2.name)}
        comm_id = comm_id + 1
        comm_historys.append(history)

        agents.loc[agent1_id, 'agent'] = agent1
        agents.loc[agent2_id, 'agent'] = agent2

        comm_turn = comm_turn + 1
        if comm_turn // len(total_connections) > comm_round:
            print(f'\n****************** Interactive_Comm Round {comm_turn // len(total_connections)} ******************** \n')
        comm_round = comm_turn // len(total_connections)

    return


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    import sys

    # 'free_op'
    # 'fix_op'

    # seed = abcdef
    # ab: different setup of experiment -> 99
    # c:  different "seed" in one setup i.e., 1 specific group  -> 1, 2
    # d:  generation    -> 0
    # ef: agents id in one group    -> assign in running

    s = [99020000 + 1000 * i for i in range(0, 50)]
    # 3020000: mk50%_osv50%

    # s = [999999]
    # s = [8820000]

    hyperdict = {'language': ['free_op'], 'trainset_proportion': [0.667], 'speaker_hidden_size': [128],
                 'meaning_embedding_dim': [8], 'batch_size': [32], 'random_seed': s}
                 # ['fix_mk2', 'fix_op', 'free_mk', 'free_op']

    rename = {'language': 'lang', 'trainset_proportion': 'split', 'speaker_hidden_size': 'hidden',
              'meaning_embedding_dim': 'emb', 'batch_size': 'batch', 'random_seed': 'seed'}

    multi_params, suffix = multi_run(hyperdict, rename, test_suffix='')

    for param_set, suf in zip(multi_params, suffix):
        main(param_set, suf)



