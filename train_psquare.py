"""
Double Dual Training
1 Policy Gradient Training with Pre-trained Transmitter & Receiver
- Training in Self-play World
- Evaluate in convai2:self
"""
import os
import random
import torch

from scripts.train_model_selfplay import setup_args as setup_args_dict, TrainLoop

# TODO: must at least two GPU as the receiver & transmitter cannot be run in the same GPU card
#  within less than 24GB memory.
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

IS_ORIGINAL = True

GEN, GATE, CLS = 1., 1., 1.
MIDDLE_POOL_SIZE = None
NEXT_POOL_SIZE = None
PERSONA_POOL_R = 1.4
USE_TO_PERSONA_POOL = False
USE_CONTEXT_POOL = False
DROP_LITERAL_PERSONA = False
PERSONA_LOWER_BOUND = 0
CONTEXT_LOWER_BOUND = 0
USE_ATTENTION = False
BEAM_SIZE = 2

MODEL_DIR = '/apdcephfs/share_916081/chencxu/pegg/AAAI/train-o-18'
DATA_DIR = '/apdcephfs/share_916081/chencxu/pegg/data'


def set_seed(seed=1):
    torch.manual_seed(seed)
    random.seed(seed)


def setup_args():
    parser = setup_args_dict()
    encode_max_seq_len = 256
    decode_max_seq_len = 32

    if IS_ORIGINAL:
        # receiver_basic = 'receiver_revised'
        transmitter_basic = 'pegg-o'
        exp_task = 'tasks.convai2.agents:OriginalTeacher,tasks.convai2.agents:OriginalPersonaTeacher'
        exp_eval_task = 'tasks.convai2transmitter.agents:SelfOriginalTeacher:no_cands'
    else:
        # receiver_basic = 'receiver_original'
        transmitter_basic = 'pegg-r'
        exp_task = 'tasks.convai2.agents:RevisedTeacher,tasks.convai2.agents:RevisedPersonaTeacher'
        exp_eval_task = 'tasks.convai2transmitter.agents:SelfRevisedTeacher:no_cands'

    exp_name = 'psquare_bot'
    validation_max = -1
    train_display = 300

    # exp_name = 'DEBUG'
    parser.set_defaults(
        # idea add ================
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
        # =======================
        download_path='{}/downloads'.format(DATA_DIR),
        datapath=DATA_DIR,
        exp=exp_name,  # name for experiment
        task=exp_task,
        personapath=MODEL_DIR,
        max_turn=3,
        evaltask=exp_eval_task,
        # TODO: now we must keep batch size equal because the world share the same agent
        evalbatchsize=6,
        batchsize=6,
        dict_lower=True,
        dict_include_valid=True,
        # not load from checkpoint
        dict_maxexs=1,
        dict_tokenizer='split',  # Important!
        datatype='train',
        history_replies='label_else_model',
        # model configuration
        model='agents.psquare.psquare:PSquareAgent',
        model_file='{}/psquare/{}.model'.format(MODEL_DIR, exp_name),
        lr=1e-6,
        gpt_lr=1e-6,
        # world sample configuration
        top_k=20,
        beam_size=2,
        gradient_clip=1.0,
        encode_max_seq_len=encode_max_seq_len,
        decode_max_seq_len=decode_max_seq_len,
        # transmitter initialization
        init_model_transmitter='{}/transmitter/{}.model'.format(MODEL_DIR, transmitter_basic),
        rnn_class_transmitter='lstm',
        lookuptable_transmitter='enc_dec',
        embedding_type_transmitter='glove_fixed',
        optimizer_step=-1,
        # receiver configuration
        # model_receiver='agents.receiver.receiver:ReceiverAgent',
        # init_model_receiver='./tmp/receiver/{}.model'.format(receiver_basic),
        # language model configuration
        init_model_coherent='{}/transmitter/{}.model'.format(MODEL_DIR, transmitter_basic),
        # validation configuration
        validation_max_exs=validation_max,  # -1
        validation_every_n_secs=3600,  # 90
        train_display_every_n_secs=train_display,
        validation_metric='f1',
        validation_metric_mode='max',
        validation_patience=10,
        log_every_n_secs=30,
        # logging configuration
        display_examples=False,
        tensorboard_log=True,
        tensorboard_tag='exp',
        train_report_metrics='num_selfplay_turns,num_selfplay_episode,total_reward,reward,reward_var',
        tensorboard_metrics='ppl,reward,total_reward,f1,hits@1,reward_var,bleu'
    )
    return parser


if __name__ == '__main__':
    opt = setup_args()
    set_seed()
    TrainLoop(opt).train()
