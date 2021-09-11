from parlai.scripts.eval_model import eval_model, setup_args as base_setup_args

IS_ORIGINAL = True
MIDDLE_POOL_SIZE = 100
NEXT_POOL_SIZE = None
PERSONA_POOL_R = None
USE_TO_PERSONA_POOL = False
USE_CONTEXT_POOL = False
DROP_LITERAL_PERSONA = False
PERSONA_LOWER_BOUND = 0
CONTEXT_LOWER_BOUND = 0
BEAM_SIZE = 2


MODEL_DIR = '/apdcephfs/share_916081/chencxu/pegg/AAAI/train-o-42/psquare/rl-o-7.model'
DATA_DIR = '/apdcephfs/share_916081/chencxu/pegg/data'


def setup_task():
    if IS_ORIGINAL:
        task_name = 'tasks.convai2transmitter.agents:SelfOriginalTeacher'
    else:
        task_name = 'tasks.convai2transmitter.agents:SelfRevisedTeacher'
    return task_name


def setup_trained_weights():
    if IS_ORIGINAL:
        weights_name = '{}'.format(MODEL_DIR)
    else:
        weights_name = '{}'.format(MODEL_DIR)
    return weights_name


def setup_args(parser=None):
    parser = base_setup_args(parser)
    task_name = setup_task()
    parser.set_defaults(
        task=task_name,
        datatype='valid',
        hide_labels=False,
        metrics='hits@1',
    )
    return parser


def eval_hits(opt, print_parser):
    report = eval_model(opt, print_parser)
    print('============================')
    print('FINAL Hits@1: ' + str(report['hits@1']))


if __name__ == '__main__':
    parser = setup_args()
    model_name = setup_trained_weights()
    parser.set_params(
        datapath=DATA_DIR,
        model_file='{}'.format(MODEL_DIR),
        model='agents.transmitter.transmitter:TransformerAgent',
        init_model_transmitter=model_name,
        gpu=0,
        batchsize=16,
        beam_size=2,
        rank_candidates=True,
        report_freq=0.0001,
    )
    print("Model: {}".format(model_name))
    opt = parser.parse_args(print_args=False)
    eval_hits(opt, print_parser=parser)
