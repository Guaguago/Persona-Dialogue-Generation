#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Train model for ppl metric with pre-selected parameters.
These parameters have some variance in their final perplexity, but they were
used to achieve the pre-trained model.
"""
import os
import random
import torch
from agents.transmitter.transmitter import ARCH_CHOICE
from parlai.scripts.train_model import setup_args as setup_dict_args, TrainLoop

# Parameters
MODEL = 'transmitter'
NAME = "pegg-o"
IS_ORIGINAL = True

GEN, GATE, CLS = 1., 1., 1.
MIDDLE_POOL_SIZE = None
NEXT_POOL_SIZE = None
PERSONA_POOL_R = 1.0
PERSONA_POOL_SIZE = None
USE_TO_PERSONA_POOL = True
USE_CONTEXT_POOL = False
DROP_LITERAL_PERSONA = False
PERSONA_LOWER_BOUND = 3
CONTEXT_LOWER_BOUND = 3
USE_ATTENTION = False
BEAM_SIZE = 2

MODEL_DIR = '/apdcephfs/share_916081/chencxu/pegg/AAAI/train-o-25'
DATA_DIR = '/apdcephfs/share_916081/chencxu/pegg/data'


def setup_task():
    if IS_ORIGINAL:
        task_name = 'tasks.convai2transmitter.agents:SelfOriginalTeacher'
    else:
        task_name = 'tasks.convai2transmitter.agents:SelfRevisedTeacher'
    return task_name


def setup_seed(seed=1706123):
    # random seed, to evaluate the performance
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)


def gpt_setting():
    return 10, 1e-4, 'gpt_custom', 1.0


def lstm_setting():
    return 64, 3, 'sgd', 0.1


def setup_args():
    """
    Use create test env setting
    :return: opt
    """
    parser = setup_dict_args()
    exp_name = NAME
    n_epoches = 100
    beam_size = BEAM_SIZE
    encode_layers = 2
    decode_layers = 2
    embedding_size = 256
    turn_emed_size = 50
    encoder_turn_use = False
    encoder_dis_use = False
    encoder_hidden_size = 1024
    decoder_hidden_size = 1024
    encode_max_seq_len = 256
    decode_max_seq_len = 32
    smoothing = 0.05
    dropout = 0.1
    embedding_type = 'glove'
    momentum = 0.9
    persona_append_strategy = 'concat'
    history_append_strategy = -1
    select_persona = False
    shuffle_persona = True
    share_decoder_input_output_embed = False
    num_train_epochs = 4

    if ARCH_CHOICE == 'gpt':
        batchsize, lr, optimizer, gradient_clip = gpt_setting()
    else:
        batchsize, lr, optimizer, gradient_clip = lstm_setting()

    task_name = setup_task()
    parser.set_defaults(
        task=task_name,
        rank_candidates=False,
        # task='tasks.convai2transmitter.agents:SelfRevisedTeacher:no_cands',
        model='agents.transmitter.transmitter:TransformerAgent',
        datapath=DATA_DIR,
        # =====================
        gen_weight=GEN,
        gate_weight=GATE,
        cls_weight=CLS,
        middle_pool_size=MIDDLE_POOL_SIZE,
        persona_pool_size=PERSONA_POOL_SIZE,
        next_pool_size=NEXT_POOL_SIZE,
        use_context_pool=USE_CONTEXT_POOL,
        use_to_persona_pool=USE_TO_PERSONA_POOL,
        drop_literal_persona=DROP_LITERAL_PERSONA,
        persona_lower_bound=PERSONA_LOWER_BOUND,
        context_lower_bound=CONTEXT_LOWER_BOUND,
        use_attention=USE_ATTENTION,
        persona_pool_r=PERSONA_POOL_R,
        # ======================
        model_file='{}/{}/{}.model'.format(MODEL_DIR, MODEL, exp_name),
        dict_tokenizer='split',
        datatype='train',
        gpt_lr=6.25e-5,
        n_epoches=n_epoches,
        num_epochs=num_train_epochs,
        batchsize=batchsize,
        beam_size=beam_size,
        encoder_layers=encode_layers,
        decoder_layers=decode_layers,
        encoder_embed_dim=embedding_size,
        encoder_turn_dim=turn_emed_size,
        encoder_turn_use=encoder_turn_use,
        encoder_dis_use=encoder_dis_use,
        decoder_embed_dim=embedding_size,
        encode_max_seq_len=encode_max_seq_len,
        decode_max_seq_len=decode_max_seq_len,
        select_persona=select_persona,
        shuffle_persona=shuffle_persona,
        persona_append_strategy=persona_append_strategy,
        history_append_strategy=history_append_strategy,
        encoder_bidirectional=False,
        encoder_hidden_size=encoder_hidden_size,
        decoder_hidden_size=decoder_hidden_size,
        smoothing=smoothing,
        lr=lr,
        dropout=dropout,
        encoder_dropout_in=dropout,
        encoder_dropout_out=0,
        decoder_dropout_in=dropout,
        decoder_dropout_out=0,
        share_decoder_input_output_embed=share_decoder_input_output_embed,
        gradient_clip=gradient_clip,
        lookuptable='enc_dec',
        optimizer=optimizer,
        embedding_type=embedding_type,
        momentum=momentum,
        # rough enough
        validation_max_exs=-1,
        validation_every_n_secs=3600,
        validation_metric='ppl',
        validation_metric_mode='min',
        validation_patience=10,
        log_every_n_secs=30,
        gpu=0,
        # logging configuration
        exp=exp_name,
        tensorboard_log=True,
        tensorboard_tag='exp',
        train_report_metrics='ppl,f1,hits@1',
        tensorboard_metrics='ppl,f1,hits@1',
        visualization=False
    )
    return parser


if __name__ == '__main__':
    opt = setup_args()
    setup_seed()
    TrainLoop(opt).train()
