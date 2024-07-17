
# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import sys

sys.path.insert(0, './pytorch-seq2seq/')

import os
from pathlib import Path

ROOT_DIR = Path(__file__).absolute().parent.parent.parent.parent
sys.path.insert(1, str(ROOT_DIR))

from itertools import product
from IPython.display import display, HTML
from typing import Any, Iterable, List, Optional, Tuple, Union


from datetime import datetime, timedelta
import random
from itertools import product
from IPython.display import display, HTML
from typing import Any, Iterable, List, Optional, Tuple, Union
import pickle as pkl
import wandb

# wandb.init(project="my-test-project", entity="ylian")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from captum.attr import LayerIntegratedGradients, TokenReferenceBase, visualization

from egg.core.batch import Batch
from egg.core import Interaction
from egg.core.interaction import LoggingStrategy
from IPython.core.display import HTML

from egg.zoo.nellcom_x.archs import RLSpeaker_decoder, SpeakerListener, my_padding
from egg.zoo.nellcom_x.archs_spk import Speaker
from egg.zoo.nellcom_x.archs_lst import Listener
from egg.zoo.nellcom_x.archs_spk import Speaker_encoder
from egg.zoo.nellcom_x.archs_lst import Listener_encoder, Listener_decoder

from egg.zoo.nellcom_x.game_callbacks import get_callbacks, v2_get_callbacks, \
    v3_get_callbacks_no_earlystop, v4_get_callbacks_cleverstopper
from egg.zoo.nellcom_x.games import build_game_after_supervised
from egg.zoo.nellcom_x.games_comm import build_game_comm_spk, build_game_comm_lst, v2_build_game_comm_spk
from egg.zoo.nellcom_x.data import get_dataloader, MyDataset, my_selected_split
from egg.zoo.nellcom_x.utils import get_opts, set_seed



def multi_run(hyperdict, rename, test_suffix=''):
    grid = list(product(*hyperdict.values()))
    multi_params = []
    suffix = []
    for value_set in grid:
        v_list = list(zip(hyperdict.keys(), value_set))
        v_str = [value2params(v) for v in v_list]
        multi_params.append(v_str)
        v_rename = '_'.join([value2str(v, rename) for v in v_list])
        # just_test
        v_rename = test_suffix + v_rename
        suffix.append(v_rename)

    return multi_params, suffix


def value2params(pair):
    para, v = pair
    return '--' + para + '=' + str(v)


def value2str(pair, name):
    para, v = pair
    return name[para] + str(v)


def load_f_df(log_path='./training_log', dump_path='./dump'):
    # training_log
    # log_path = './training_log'
    files = os.listdir(log_path)
    files = [f for f in files if 'free_op' in f]
    f_df_log = general_files2df(log_path, files, fn=lambda x: os.path.splitext(x)[1] == '.txt')

    # dump
    # dump_path = './dump'
    files = os.listdir(dump_path)
    files = [f for f in files if 'free_op' in f]
    f_df_dump = general_files2df(dump_path, files, fn=lambda x: 'lang' in x)

    new_f_df = check_file_df(f_df_log, f_df_dump)

    return new_f_df


def get_dump(f_df, suf, param_set, dump_type='comm', dump_path='dump', epoch=60, classified=pd.DataFrame()):

    f_df_suf = f_df.loc[f_df.fpath_dump.str.contains(suf, regex=False)]

    _, dump_, _ = get_df_anaoutput_final(f_df_suf, dump_path, sv=(dump_type == 'spk'), specific_epoch=epoch)

    if classified.empty:
        dump_clf = classified
        print('no dump_clf loaded!')
    else:
        lang, seed = get_params(['lang', 'seed'], param_set)
        dump_clf = classified.loc[(classified['lang'] == lang.replace('_', '-'))
                                  & (classified['seed'] == int(seed))
                                  & (classified['epoch'] == epoch)]

    return dump_, dump_clf


def get_params(params, params_set):
    set = []
    for p in params:
        for ps in params_set:
            if p in ps:
                set.append(ps[ps.index('=')+1 :])
    return set


def substr_rindex(s, subs):
    if subs in s:
        ind = s.index(subs) + len(subs)
        if s[ind:].find('_') == -1:
            next_ = None
        else:
            next_ = s[ind:].index('_')+ind
        return s.index(subs) + len(subs), next_
    else:
        return False


def myloading_tensor(dir, f_list):
    tensor_dict = {}
    if len(f_list) > 0:
        for f in f_list:
            t = torch.load(os.path.join(dir, f))
            x, y = substr_rindex(f, 'epoch')
            epoch = int(f[x:y])
            tensor_dict[epoch] = t
    return tensor_dict


def additional_dump_cleverstop(dataset, output_dir, dump_dir, mode, cleverstop_suffix):
    files = os.listdir(output_dir)
    files = [f for f in files if cleverstop_suffix in f]

    f_uttr = [f for f in files if 'uttr' in f]
    uttr_dict = myloading_tensor(output_dir, f_uttr)

    f_mean = [f for f in files if 'mean' in f]
    mean_dict = myloading_tensor(output_dir, f_mean)

    f_msg = [f for f in files if 'msg' in f]
    msg_dict = myloading_tensor(output_dir, f_msg)

    f_lstpred = [f for f in files if 'lstpred' in f]
    lstpred_dict = myloading_tensor(output_dir, f_lstpred)

    if uttr_dict.keys() == mean_dict.keys() == msg_dict.keys() == lstpred_dict.keys():
        for k in msg_dict.keys():
            uttr = uttr_dict[k]
            mean = mean_dict[k]
            msg = msg_dict[k]
            lstpred = lstpred_dict[k]
            total_steps = uttr.shape[1]-1

            if not(uttr.size(0) == mean.size(0)):
                break

            msg_token, uttr_token, mean_token, lstpred_token = [], [], [], []
            for i in range(uttr.size(0)):
                uttr_t = ' '.join(dataset.vocab_utterance.lookup_tokens(uttr[i].tolist()))
                mean_t = ' '.join(dataset.vocab_meaning.lookup_tokens(mean[i].tolist()))
                msg_t = ' '.join(dataset.vocab_utterance.lookup_tokens(msg[i].tolist())) if mode in ['comm', 'spk'] else ''
                lstpred_t = ' '.join(dataset.vocab_meaning.lookup_tokens(lstpred[i].tolist())) if mode in ['comm', 'lst'] else ''
                msg_token.append(msg_t)
                uttr_token.append(uttr_t)
                mean_token.append(mean_t)
                lstpred_token.append(lstpred_t)

            df = pd.DataFrame({'meaning': mean_token, 'utterance': uttr_token,
                               'message': msg_token, 'listener_prediction': lstpred_token})
            df.to_csv(os.path.join(dump_dir, f'{cleverstop_suffix}_dump_epoch{k}.txt'), sep='\t')
    return


def arrange_dump_v2(dataset, output_dir, dump_dir, mode):
    files = os.listdir(output_dir)

    f_uttr = [f for f in files if 'uttr' in f]
    uttr_dict = myloading_tensor(output_dir, f_uttr)

    f_mean = [f for f in files if 'mean' in f]
    mean_dict = myloading_tensor(output_dir, f_mean)

    f_msg = [f for f in files if 'msg' in f]
    msg_dict = myloading_tensor(output_dir, f_msg)

    f_lstpred = [f for f in files if 'lstpred' in f]
    lstpred_dict = myloading_tensor(output_dir, f_lstpred)

    if uttr_dict.keys() == mean_dict.keys() == msg_dict.keys() == lstpred_dict.keys():
        for k in msg_dict.keys():
            uttr = uttr_dict[k]
            mean = mean_dict[k]
            msg = msg_dict[k]
            lstpred = lstpred_dict[k]
            total_steps = uttr.shape[1]-1

            if not(uttr.size(0) == mean.size(0)):
                break

            msg_token, uttr_token, mean_token, lstpred_token = [], [], [], []
            for i in range(uttr.size(0)):
                uttr_t = ' '.join(dataset.vocab_utterance.lookup_tokens(uttr[i].tolist()))
                mean_t = ' '.join(dataset.vocab_meaning.lookup_tokens(mean[i].tolist()))
                msg_t = ' '.join(dataset.vocab_utterance.lookup_tokens(msg[i].tolist())) if mode in ['comm', 'spk'] else ''
                lstpred_t = ' '.join(dataset.vocab_meaning.lookup_tokens(lstpred[i].tolist())) if mode in ['comm', 'lst'] else ''
                msg_token.append(msg_t)
                uttr_token.append(uttr_t)
                mean_token.append(mean_t)
                lstpred_token.append(lstpred_t)

            df = pd.DataFrame({'meaning': mean_token, 'utterance': uttr_token,
                               'message': msg_token, 'listener_prediction': lstpred_token})
            df.to_csv(os.path.join(dump_dir, f'dump_epoch{k}.txt'), sep='\t')
    return


def iter_dump(dataset, output_dir, dump_dir, mode, generation=0):
    files = os.listdir(output_dir)

    f_uttr = [f for f in files if 'uttr' in f]
    uttr_dict = myloading_tensor(output_dir, f_uttr)

    f_mean = [f for f in files if 'mean' in f]
    mean_dict = myloading_tensor(output_dir, f_mean)

    f_msg = [f for f in files if 'msg' in f]
    msg_dict = myloading_tensor(output_dir, f_msg)

    f_lstpred = [f for f in files if 'lstpred' in f]
    lstpred_dict = myloading_tensor(output_dir, f_lstpred)

    if uttr_dict.keys() == mean_dict.keys() == msg_dict.keys() == lstpred_dict.keys():
        for k in msg_dict.keys():
            uttr = uttr_dict[k]
            mean = mean_dict[k]
            msg = msg_dict[k]
            lstpred = lstpred_dict[k]
            total_steps = uttr.shape[1]-1

            if not(uttr.size(0) == mean.size(0)):
                break

            msg_token, uttr_token, mean_token, lstpred_token = [], [], [], []
            for i in range(uttr.size(0)):
                uttr_t = dataset.vocab_utterance.lookup_tokens(uttr[i][1:-1].tolist())
                uttr_t = cutt_util_eos(uttr_t)
                mean_t = ' '.join(dataset.vocab_meaning.lookup_tokens(mean[i].tolist()))
                msg_t = dataset.vocab_utterance.lookup_tokens(msg[i][:-2].tolist()) if mode in ['comm', 'spk'] else ''
                msg_t = cutt_util_eos(msg_t)
                lstpred_t = ' '.join(dataset.vocab_meaning.lookup_tokens(lstpred[i].tolist())) if mode in ['comm', 'lst'] else ''
                msg_token.append(msg_t)
                uttr_token.append(uttr_t)
                mean_token.append(mean_t)
                lstpred_token.append(lstpred_t)

            df = pd.DataFrame({'meaning': mean_token, 'utterance': uttr_token,
                               'message': msg_token, 'listener_prediction': lstpred_token})
            df.to_csv(os.path.join(dump_dir, f'dump_epoch{k}.txt'), sep='\t')

            df_spk_iter_data = pd.DataFrame({'meaning': mean_token, 'message': msg_token})
            df_spk_iter_data.to_csv(os.path.join(dump_dir, f'spk_epoch{k}_iter{generation}.txt'), sep='\t', index=False, header=False)

    return


def cutt_util_eos(msg):
    msg_ = ''
    if len(msg) > 0:
        if len(msg) == 10:
            print(msg)
        for m in msg:
            if m == '<EOS>':
                break
            elif m == '<SOS>' or m == '<PAD>':
                msg_ = msg_
            else:
                msg_ = ' '.join([msg_, m])
    return msg_


# useless funcs from iter_train
def dump_previous_dataset(model, dataset, full_dataset, generation, epoch, teacher_outputs_dir, teacher_generated_dataset_dir, train_mode=None):
    interactions = []

    with torch.no_grad():
        for batch in dataset:
            if not isinstance(batch, Batch):
                batch = Batch(*batch)
            batch = batch.to("cpu")
            interaction = game_comm(model, *batch)
            interaction = interaction.to("cpu")
            interactions.append(interaction)

    if not os.path.exists(teacher_outputs_dir):
        os.mkdir(teacher_outputs_dir)

    teacher_outputs_dir = os.path.join(teacher_outputs_dir, train_mode)
    if not os.path.exists(teacher_outputs_dir):
        os.mkdir(teacher_outputs_dir)

    iters = Interaction.from_iterable(interactions)

    dump_dir_mean = os.path.join(teacher_outputs_dir, f'mean_epoch{epoch}')
    dump_dir_uttr = os.path.join(teacher_outputs_dir, f'uttr_epoch{epoch}')
    dump_dir_msg = os.path.join(teacher_outputs_dir, f'msg_epoch{epoch}')
    dump_dir_lstpredict = os.path.join(teacher_outputs_dir, f'lstpred_epoch{epoch}')
    torch.save(iters.sender_input, dump_dir_mean)
    torch.save(iters.message, dump_dir_msg)
    torch.save(iters.labels, dump_dir_uttr)
    torch.save(iters.receiver_output, dump_dir_lstpredict)

    if not os.path.exists(teacher_generated_dataset_dir):
        os.mkdir(teacher_generated_dataset_dir)

    teacher_generated_dataset_dir = os.path.join(teacher_generated_dataset_dir, train_mode)
    if not os.path.exists(teacher_generated_dataset_dir):
        os.mkdir(teacher_generated_dataset_dir)

    iter_dump(dataset=full_dataset, output_dir=teacher_outputs_dir,
              dump_dir=teacher_generated_dataset_dir, mode='comm', generation=generation)


def load_iter_train_datasplits(teacher, train_loader_dataset, test_loader_dataset, full_dataset, opts, suffix, t_epoch, t_generation, new_seed):

    teacher_generated_dataset_dir = os.path.join('iter_teacher_generated_dataset', suffix + '_comm')
    next_train_dataset_dir = os.path.join(teacher_generated_dataset_dir, 'train', f'spk_epoch{t_epoch}_iter{t_generation}.txt')
    next_test_dataset_dir = os.path.join(teacher_generated_dataset_dir, 'test', f'spk_epoch{t_epoch}_iter{t_generation}.txt')

    if not os.path.exists(next_train_dataset_dir) or not os.path.exists(next_test_dataset_dir):
        teacher_outputs = os.path.join('iter_teacher_outputs', suffix)
        dump_previous_dataset(teacher, train_loader_dataset, full_dataset, t_generation, t_epoch, teacher_outputs,
                              teacher_generated_dataset_dir, train_mode='train')
        dump_previous_dataset(teacher, test_loader_dataset, full_dataset, t_generation, t_epoch, teacher_outputs,
                              teacher_generated_dataset_dir, train_mode='test')

    next_train_dataset = MyDataset(next_train_dataset_dir, opts.language)
    train_d = my_selected_split(next_train_dataset, [len(next_train_dataset)], selected=False)

    next_test_dataset = MyDataset(next_test_dataset_dir, opts.language)
    test_d = my_selected_split(next_test_dataset, [len(next_test_dataset)], selected=False)

    train_loader_for_iter = get_dataloader(
        train_dataset=train_d,
        batch_size=opts.batch_size,
        num_workers=opts.num_workers,
        is_distributed=opts.distributed_context.is_distributed,
        seed=new_seed,
    )
    test_loader_for_iter = get_dataloader(
        train_dataset=test_d,
        batch_size=len(test_d),
        num_workers=opts.num_workers,
        is_distributed=opts.distributed_context.is_distributed,
        seed=new_seed,
        drop_last=False,
        shuffle=False,
    )

    return train_loader_for_iter, test_loader_for_iter


def create_new_datasplits(new_seed, opts, save_splits_dir):
    dataset_dir = os.path.join(opts.dataset_folder, opts.language, opts.dataset_filename)

    full_dataset = MyDataset(dataset_dir, opts.language)

    training_data_size = opts.trainset_proportion

    language = opts.language

    train_size = int(training_data_size * len(full_dataset))
    test_size = len(full_dataset) - train_size

    # train_d, test_d = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    train_d, test_d = my_selected_split(full_dataset, [train_size, test_size], selected=True)
    # selected=True:    all elements appear in the train set,
    #                   train_size MUST sent first!!!

    # to save datasplits
    if not os.path.exists(save_splits_dir):
        os.mkdir(save_splits_dir)
    torch.save(train_d, f"{save_splits_dir}/train.pkl")
    torch.save(test_d, f"{save_splits_dir}/test.pkl")
    torch.save(full_dataset, f"{save_splits_dir}/full.pkl")

    train_loader = get_dataloader(
        train_dataset=train_d,
        batch_size=opts.batch_size,
        num_workers=opts.num_workers,
        is_distributed=opts.distributed_context.is_distributed,
        seed=new_seed,
    )
    test_loader = get_dataloader(
        train_dataset=test_d,
        batch_size=len(test_d),
        num_workers=opts.num_workers,
        is_distributed=opts.distributed_context.is_distributed,
        seed=new_seed,
        drop_last=False,
        shuffle=False,
    )

    return train_loader, test_loader, full_dataset


def load_init_datasplits(splits_dir, opts):
    train_d = torch.load(f"{splits_dir}/train.pkl")
    test_d = torch.load(f"{splits_dir}/test.pkl")
    full_dataset = torch.load(f"{splits_dir}/full.pkl")

    train_loader_test = get_dataloader(
        train_dataset=train_d,
        batch_size=opts.batch_size,
        num_workers=opts.num_workers,
        is_distributed=opts.distributed_context.is_distributed,
        seed=opts.random_seed,
    )
    test_loader_test = get_dataloader(
        train_dataset=test_d,
        batch_size=len(test_d),
        num_workers=opts.num_workers,
        is_distributed=opts.distributed_context.is_distributed,
        seed=opts.random_seed,
        drop_last=False,
        shuffle=False,
    )

    return train_loader_test, test_loader_test, full_dataset


def init_comm_agents(
        train_data,
        meaning_embedding_size: int = 32,
        speaker_hidden_size: int = 128,
        listener_hidden_size: int = 128,
        rnn_cell: str = 'gru',
        spk_max_len: int = 10,
) -> nn.Module:
    meaning_vocab_size, uttr_vocab_size = train_data.get_vocab_size()
    sos_id, eos_id, pad_id = train_data.get_special_index()
    meaning_max_len, uttr_max_len = train_data.get_max_len()
    uttr_max_len = spk_max_len

    speaker_enc = Speaker_encoder(vocab_size=meaning_vocab_size, embedding_size=meaning_embedding_size,
                                  max_len=meaning_max_len, output_size=speaker_hidden_size)
    rl_speaker_dec = RLSpeaker_decoder(vocab_size=uttr_vocab_size, max_len=uttr_max_len,
                                       hidden_size=speaker_hidden_size,
                                       sos_id=sos_id, eos_id=eos_id, rnn_cell=rnn_cell, use_attention=False,
                                       pad_id=pad_id)
    # speaker = Speaker(speaker_enc, rl_speaker_dec)

    listener_enc = Listener_encoder(vocab_size=uttr_vocab_size, max_len=meaning_max_len,
                                    hidden_size=listener_hidden_size,
                                    rnn_cell=rnn_cell, variable_lengths=True, sos_id=sos_id, eos_id=eos_id,
                                    pad_id=pad_id)
    listener_dec = Listener_decoder(vocab_size=meaning_vocab_size, meaning_len=meaning_max_len,
                                    input_size=listener_hidden_size)
    # listener = Listener(listener_enc, listener_dec)

    return speaker_enc, rl_speaker_dec, listener_enc, listener_dec


def load_model(dataset, opts, load_clever_stop=True, cleverstop_suffix='', load_teacher=True, folder_name=None, load_sv=False):
    speaker_enc, rl_speaker_dec, listener_enc, listener_dec = init_comm_agents(
        train_data=dataset,
        meaning_embedding_size=opts.meaning_embedding_dim,
        speaker_hidden_size=opts.speaker_hidden_size,
        listener_hidden_size=opts.listener_hidden_size,
        rnn_cell=opts.rnn,
        spk_max_len=opts.spk_max_len
    )

    if load_teacher:

        fmodels = os.listdir(folder_name)
        files = [f for f in fmodels if cleverstop_suffix in f]

        if load_clever_stop and len(files) == 5:
            s = files[0]
            cleverstop_epoch = int(s[len('Spk_Lst_'):s.index('_cleverstop'+cleverstop_suffix)])
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
    spk = Speaker(speaker_enc, rl_speaker_dec)
    lst = Listener(listener_enc, listener_dec)

    return spk, lst


def game_comm(spk_lst,
              sender_input: torch.Tensor,
              labels: torch.Tensor,
              receiver_input: torch.Tensor = None,
              aux_input=None, ):
    receiver_input = receiver_input.items() if any(receiver_input) else None

    speaker_output, message, message_length, entropy_spk = game_comm_speaking(spk_lst, sender_input, labels=labels,
                                                                              receiver_input=receiver_input,
                                                                              aux_input=aux_input)

    listener_output, listener_prediction, logits_lst, entropy_lst = game_comm_listening(spk_lst, sender_input, message)

    aux_info = dict()

    logging_strategy = LoggingStrategy()

    interaction = logging_strategy.filtered_interaction(
        sender_input=sender_input,
        receiver_input=receiver_input,
        labels=labels,
        aux_input=aux_input,
        receiver_output=listener_prediction,
        message=message,
        message_length=torch.ones(message[0].size(0)),
        aux=aux_info,
    )

    return interaction


def game_comm_speaking(spk_lst, sender_input: torch.Tensor, labels: torch.Tensor, receiver_input: torch.Tensor = None,
                       aux_input=None, ):
    # # SPEAKING
    spk_lst.speaker.training = False
    speaker_output, message, message_length, entropy_spk = spk_lst.speaker(sender_input, labels=labels,
                                                                           receiver_input=receiver_input,
                                                                           aux_input=aux_input)
    speaker_output = torch.stack(speaker_output).permute(1, 0)

    # # attach [sos] to message
    # sos_ = torch.stack([torch.tensor(self.speaker.decoder.sos_id)] * message.size(0))
    # message = torch.cat((sos_.unsqueeze(-1), message), dim=1)
    padded_message, message_length = my_padding(message, spk_lst.speaker.eos_id, spk_lst.speaker.pad_id,
                                                do_padding=spk_lst.do_padding)

    return speaker_output, padded_message, message_length, entropy_spk


def game_comm_listening(spk_lst, sender_input: torch.Tensor, padded_message):
    # # LISTENING
    # # switch input&lable
    listener_output, listener_prediction, logits_lst, entropy_lst = spk_lst.listener(sender_input=padded_message,
                                                                                     labels=sender_input,
                                                                                     receiver_input=None,
                                                                                     aux_input=None)

    return listener_output, listener_prediction, logits_lst, entropy_lst



def v2_dump_previous_dataset(model, dataset, full_dataset, generation, epoch, teacher_outputs_dir, teacher_generated_dataset_dir):
    interactions = []

    with torch.no_grad():
        for batch in dataset:
            if not isinstance(batch, Batch):
                batch = Batch(*batch)
            batch = batch.to("cpu")
            interaction = game_comm(model, *batch)
            interaction = interaction.to("cpu")
            interactions.append(interaction)

    if not os.path.exists(teacher_outputs_dir):
        os.mkdir(teacher_outputs_dir)

    iters = Interaction.from_iterable(interactions)

    dump_dir_mean = os.path.join(teacher_outputs_dir, f'mean_epoch{epoch}')
    dump_dir_uttr = os.path.join(teacher_outputs_dir, f'uttr_epoch{epoch}')
    dump_dir_msg = os.path.join(teacher_outputs_dir, f'msg_epoch{epoch}')
    dump_dir_lstpredict = os.path.join(teacher_outputs_dir, f'lstpred_epoch{epoch}')
    torch.save(iters.sender_input, dump_dir_mean)
    torch.save(iters.message, dump_dir_msg)
    torch.save(iters.labels, dump_dir_uttr)
    torch.save(iters.receiver_output, dump_dir_lstpredict)

    if not os.path.exists(teacher_generated_dataset_dir):
        os.mkdir(teacher_generated_dataset_dir)

    iter_dump(dataset=full_dataset, output_dir=teacher_outputs_dir,
              dump_dir=teacher_generated_dataset_dir, mode='comm', generation=generation)


def teacher_dataset_dump(teacher, full_dataset, opts, new_split_sizes, suffix, t_epoch, t_generation, shared_seed, shared_gen):

    # step 1: generate teacher_full_dataset_csv
    # no child related
    teacher_generated_dataset_dir = os.path.join('iter_teacher_generated_dataset', suffix + '_comm')
    next_dataset_dir = os.path.join(teacher_generated_dataset_dir, f'spk_epoch{t_epoch}_iter{t_generation}.txt')

    full_data_loader = get_dataloader(
        train_dataset=full_dataset,
        batch_size=len(full_dataset),
        num_workers=opts.num_workers,
        is_distributed=opts.distributed_context.is_distributed,
        seed=shared_seed,
    )

    # if not os.path.exists(next_dataset_dir):
    teacher_outputs_dir = os.path.join('iter_teacher_outputs', suffix)
    v2_dump_previous_dataset(teacher, full_data_loader, full_dataset, t_generation, t_epoch, teacher_outputs_dir,
                             teacher_generated_dataset_dir)

    overall_full_dataset = MyDataset(csv_file=next_dataset_dir, language=opts.language, gen=shared_gen)

    train_size, test_size, dev_size = new_split_sizes

    combo_train_d, test_d = my_selected_split(overall_full_dataset, [train_size + dev_size, test_size], selected=True)

    # to save datasplits
    save_splits_dir = os.path.join('generated_splits', suffix)
    if not os.path.exists(save_splits_dir):
        os.mkdir(save_splits_dir)
    torch.save(combo_train_d, f"{save_splits_dir}/train.pkl")
    torch.save(test_d, f"{save_splits_dir}/test.pkl")
    torch.save(overall_full_dataset, f"{save_splits_dir}/full.pkl")

    return combo_train_d, test_d


def load_iter_datasets(teacher, full_dataset, opts, suffix, t_epoch, t_generation, new_seed, new_gen):

    teacher_generated_dataset_dir = os.path.join('iter_teacher_generated_dataset', suffix + '_comm')
    next_dataset_dir = os.path.join(teacher_generated_dataset_dir, f'spk_epoch{t_epoch}_iter{t_generation}.txt')

    full_data_loader = get_dataloader(
        train_dataset=full_dataset,
        batch_size=len(full_dataset),
        num_workers=opts.num_workers,
        is_distributed=opts.distributed_context.is_distributed,
        seed=new_seed,
    )

    # if not os.path.exists(next_dataset_dir):
    teacher_outputs_dir = os.path.join('iter_teacher_outputs', suffix)
    v2_dump_previous_dataset(teacher, full_data_loader, full_dataset, t_generation, t_epoch, teacher_outputs_dir,
                          teacher_generated_dataset_dir)

    next_full_dataset = MyDataset(csv_file=next_dataset_dir, language=opts.language, gen=new_gen)

    train_size = int(opts.trainset_proportion * len(next_full_dataset))
    test_size = len(next_full_dataset) - train_size

    train_d, test_d = my_selected_split(next_full_dataset, [train_size, test_size], selected=True)

    # to save datasplits
    save_splits_dir = os.path.join('generated_splits', suffix)
    if not os.path.exists(save_splits_dir):
        os.mkdir(save_splits_dir)
    torch.save(train_d, f"{save_splits_dir}/train.pkl")
    torch.save(test_d, f"{save_splits_dir}/test.pkl")
    torch.save(full_dataset, f"{save_splits_dir}/full.pkl")

    train_loader_for_iter = get_dataloader(
        train_dataset=train_d,
        batch_size=opts.batch_size,
        num_workers=opts.num_workers,
        is_distributed=opts.distributed_context.is_distributed,
        seed=new_seed,
    )
    test_loader_for_iter = get_dataloader(
        train_dataset=test_d,
        batch_size=len(test_d),
        num_workers=opts.num_workers,
        is_distributed=opts.distributed_context.is_distributed,
        seed=new_seed,
        drop_last=False,
        shuffle=False,
    )

    return train_loader_for_iter, test_loader_for_iter, next_full_dataset


def load_normal_datasets(full_dataset, opts, suffix, new_seed, group_comm=False):
    train_size = int(opts.trainset_proportion * len(full_dataset))
    test_size = len(full_dataset) - train_size

    train_d, test_d = my_selected_split(full_dataset, [train_size, test_size], selected=True)

    # to save datasplits
    comm_suff = 'group_' if group_comm else ''

    save_splits_dir = os.path.join(f'{comm_suff}splits', suffix)
    if not os.path.exists(save_splits_dir):
        os.mkdir(save_splits_dir)
    torch.save(train_d, f"{save_splits_dir}/train.pkl")
    torch.save(test_d, f"{save_splits_dir}/test.pkl")
    torch.save(full_dataset, f"{save_splits_dir}/full.pkl")

    train_loader_for_iter = get_dataloader(
        train_dataset=train_d,
        batch_size=opts.batch_size,
        num_workers=opts.num_workers,
        is_distributed=opts.distributed_context.is_distributed,
        seed=new_seed,
    )
    test_loader_for_iter = get_dataloader(
        train_dataset=test_d,
        batch_size=len(test_d),
        num_workers=opts.num_workers,
        is_distributed=opts.distributed_context.is_distributed,
        seed=new_seed,
        drop_last=False,
        shuffle=False,
    )

    return train_loader_for_iter, test_loader_for_iter, full_dataset


def create_basic_split(dataset_dir, opts, overall_seed, splits_proportions=[0.667, 0.2]):
    # dataset_dir = os.path.join(opts.dataset_folder, opts.language, opts.dataset_filename)
    overall_full_dataset = MyDataset(csv_file=dataset_dir, language=opts.language, gen=0)

    # # changed from 240/480 to 144/dev/480
    # new_trainset_proportion = 0.667
    # new_testset_proportion = 0.2
    new_trainset_proportion, new_testset_proportion = splits_proportions

    train_size = int(new_trainset_proportion * len(overall_full_dataset))
    test_size = int(new_testset_proportion * len(overall_full_dataset))
    dev_size = len(overall_full_dataset) - train_size - test_size

    set_seed(overall_seed)
    combo_train_d, test_d = my_selected_split(overall_full_dataset, [train_size + dev_size, test_size],
                                              selected=True)
    # selected=True:    all elements appear in the train set,
    #                   train_size MUST sent first!!!

    return overall_full_dataset, combo_train_d, test_d


def set_model_saver_saving_mode(trainers, mode):
    for trainer in trainers:
        for callback in trainer.callbacks:
            if callback.callback_name == 'GeneralModelSaver':
                callback.set_saving_mode(mode)
    return

