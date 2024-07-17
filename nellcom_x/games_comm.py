# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn

from egg.core.interaction import LoggingStrategy
from egg.zoo.nellcom_x.archs import RLSpeaker_decoder, RLSpeaker_decoder_tied
from egg.zoo.nellcom_x.archs_spk import Speaker
from egg.zoo.nellcom_x.archs_lst import Listener
from egg.zoo.nellcom_x.archs_spk import Speaker_Game, Speaker_encoder, Speaker_encoder_tied
from egg.zoo.nellcom_x.losses_spk import MyLoss_spk, MyLoss_spk_v2
from egg.zoo.nellcom_x.archs_lst import Listener_Game, Listener_encoder, Listener_decoder, \
    Listener_encoder_tied, Listener_decoder_tied
from egg.zoo.nellcom_x.losses_lst import MyLoss_lst


def build_game_comm_spk(
        train_data,
        meaning_embedding_size: int = 32,
        decoder_hidden_size: int = 128,
        is_distributed: bool = False,
        rnn_cell: str = 'gru',
        spk_max_len: int = 10,
) -> nn.Module:

    train_logging_strategy = LoggingStrategy.minimal()
    train_logging_strategy.store_aux_input =True

    meaning_vocab_size, uttr_vocab_size = train_data.get_vocab_size()
    sos_id, eos_id, pad_id = train_data.get_special_index()
    meaning_max_len, uttr_max_len = train_data.get_max_len()
    uttr_max_len = spk_max_len

    loss = MyLoss_spk(nn.NLLLoss(ignore_index=pad_id))

    speaker_enc = Speaker_encoder(vocab_size=meaning_vocab_size, embedding_size=meaning_embedding_size, max_len=meaning_max_len, output_size=decoder_hidden_size)
    rl_speaker_dec = RLSpeaker_decoder(vocab_size=uttr_vocab_size, max_len=uttr_max_len, hidden_size=decoder_hidden_size, sos_id=sos_id, eos_id=eos_id, rnn_cell=rnn_cell, use_attention=False, pad_id=pad_id)
    speaker = Speaker(speaker_enc, rl_speaker_dec)
    speaker.decoder.set_train_mode('supervised')

    game = Speaker_Game(speaker, loss, train_logging_strategy)
    if is_distributed:
        game = nn.SyncBatchNorm.convert_sync_batchnorm(game)

    return game


def v2_build_game_comm_spk(
        train_data,
        meaning_embedding_size: int = 32,
        decoder_hidden_size: int = 128,
        is_distributed: bool = False,
        rnn_cell: str = 'gru',
        spk_max_len: int = 10,
) -> nn.Module:

    train_logging_strategy = LoggingStrategy.minimal()
    train_logging_strategy.store_aux_input =True

    meaning_vocab_size, uttr_vocab_size = train_data.get_vocab_size()
    sos_id, eos_id, pad_id = train_data.get_special_index()
    meaning_max_len, uttr_max_len = train_data.get_max_len()
    uttr_max_len = spk_max_len

    loss = MyLoss_spk_v2(nn.NLLLoss(ignore_index=pad_id))

    speaker_enc = Speaker_encoder(vocab_size=meaning_vocab_size, embedding_size=meaning_embedding_size, max_len=meaning_max_len, output_size=decoder_hidden_size)
    rl_speaker_dec = RLSpeaker_decoder(vocab_size=uttr_vocab_size, max_len=uttr_max_len, hidden_size=decoder_hidden_size, sos_id=sos_id, eos_id=eos_id, rnn_cell=rnn_cell, use_attention=False, pad_id=pad_id)
    speaker = Speaker(speaker_enc, rl_speaker_dec)
    speaker.decoder.set_train_mode('supervised')

    game = Speaker_Game(speaker, loss, train_logging_strategy)
    if is_distributed:
        game = nn.SyncBatchNorm.convert_sync_batchnorm(game)

    return game


def tied_v2_build_game_comm_spk(
        train_data,
        meaning_embedding_size: int = 32,
        decoder_hidden_size: int = 128,
        is_distributed: bool = False,
        rnn_cell: str = 'gru',
        spk_max_len: int = 10,
        shared_word_embedding = None,
        shared_meaning_embedding = None
) -> nn.Module:

    train_logging_strategy = LoggingStrategy.minimal()
    train_logging_strategy.store_aux_input =True

    meaning_vocab_size, uttr_vocab_size = train_data.get_vocab_size()
    sos_id, eos_id, pad_id = train_data.get_special_index()
    meaning_max_len, uttr_max_len = train_data.get_max_len()
    uttr_max_len = spk_max_len

    loss = MyLoss_spk_v2(nn.NLLLoss(ignore_index=pad_id))

    # speaker_enc = Speaker_encoder(vocab_size=meaning_vocab_size, embedding_size=meaning_embedding_size, max_len=meaning_max_len, output_size=decoder_hidden_size)
    tied_speaker_enc = Speaker_encoder_tied(shared_meaning_embedding=shared_meaning_embedding, vocab_size=meaning_vocab_size, embedding_size=meaning_embedding_size, max_len=meaning_max_len, output_size=decoder_hidden_size)
    # rl_speaker_dec = RLSpeaker_decoder(vocab_size=uttr_vocab_size, max_len=uttr_max_len, hidden_size=decoder_hidden_size, sos_id=sos_id, eos_id=eos_id, rnn_cell=rnn_cell, use_attention=False, pad_id=pad_id)
    tied_rl_speaker_dec = RLSpeaker_decoder_tied(shared_word_embedding=shared_word_embedding, vocab_size=uttr_vocab_size, max_len=uttr_max_len, hidden_size=decoder_hidden_size, sos_id=sos_id, eos_id=eos_id, rnn_cell=rnn_cell, use_attention=False, pad_id=pad_id)
    # speaker = Speaker(speaker_enc, rl_speaker_dec)
    speaker = Speaker(tied_speaker_enc, tied_rl_speaker_dec)
    speaker.decoder.set_train_mode('supervised')

    game = Speaker_Game(speaker, loss, train_logging_strategy)
    if is_distributed:
        game = nn.SyncBatchNorm.convert_sync_batchnorm(game)

    return game


def build_game_comm_lst(
        train_data,
        encoder_hidden_size,
        is_distributed: bool = False,
        rnn_cell: str = 'gru',
) -> nn.Module:

    train_logging_strategy = LoggingStrategy.minimal()
    train_logging_strategy.store_aux_input =True

    loss = MyLoss_lst(nn.NLLLoss())

    meaning_vocab_size, uttr_vocab_size = train_data.get_vocab_size()
    meaning_max_len, uttr_max_len = train_data.get_max_len()
    sos_id, eos_id, pad_id = train_data.get_special_index()

    listener_enc = Listener_encoder(vocab_size=uttr_vocab_size, max_len=meaning_max_len, hidden_size=encoder_hidden_size, rnn_cell=rnn_cell, variable_lengths=True, sos_id=sos_id, eos_id=eos_id, pad_id=pad_id)
    listener_dec = Listener_decoder(vocab_size=meaning_vocab_size, meaning_len=meaning_max_len, input_size=encoder_hidden_size)
    listener = Listener(listener_enc, listener_dec)
    game = Listener_Game(listener, loss, train_logging_strategy)
    if is_distributed:
        game = nn.SyncBatchNorm.convert_sync_batchnorm(game)

    return game


def tied_build_game_comm_lst(
        train_data,
        encoder_hidden_size,
        meaning_embedding_dim: int = 32,
        is_distributed: bool = False,
        rnn_cell: str = 'gru',
        shared_word_embedding = None,
        shared_meaning_embedding = None,
        lst_type = 'with_embedding'
) -> nn.Module:

    train_logging_strategy = LoggingStrategy.minimal()
    train_logging_strategy.store_aux_input =True

    loss = MyLoss_lst(nn.NLLLoss())

    meaning_vocab_size, uttr_vocab_size = train_data.get_vocab_size()
    meaning_max_len, uttr_max_len = train_data.get_max_len()
    sos_id, eos_id, pad_id = train_data.get_special_index()

    # listener_enc = Listener_encoder(vocab_size=uttr_vocab_size, max_len=meaning_max_len, hidden_size=encoder_hidden_size, rnn_cell=rnn_cell, variable_lengths=True, sos_id=sos_id, eos_id=eos_id, pad_id=pad_id)
    tied_listener_enc = Listener_encoder_tied(shared_word_embedding=shared_word_embedding, vocab_size=uttr_vocab_size, max_len=meaning_max_len, hidden_size=encoder_hidden_size, rnn_cell=rnn_cell, variable_lengths=True, sos_id=sos_id, eos_id=eos_id, pad_id=pad_id)
    # listener_dec = Listener_decoder(vocab_size=meaning_vocab_size, meaning_len=meaning_max_len, input_size=encoder_hidden_size)
    tied_listener_dec = Listener_decoder_tied(shared_meaning_embedding=shared_meaning_embedding, meaning_embedding_dim=meaning_embedding_dim, vocab_size=meaning_vocab_size,
                                              meaning_len=meaning_max_len, input_size=encoder_hidden_size, listener_type=lst_type)
    # listener = Listener(listener_enc, listener_dec)
    listener = Listener(tied_listener_enc, tied_listener_dec)
    game = Listener_Game(listener, loss, train_logging_strategy)
    if is_distributed:
        game = nn.SyncBatchNorm.convert_sync_batchnorm(game)

    return game