from parlai.scripts.eval_model import eval_model, setup_args as base_setup_args

IS_ORIGINAL = True
MODEL_DIR = 'train-o-18'
MIDDLE_POOL_SIZE = None
NEXT_POOL_SIZE = None
PERSONA_POOL_R = None
PERSONA_POOL_SIZE = 50
USE_TO_PERSONA_POOL = False
USE_CONTEXT_POOL = False
DROP_LITERAL_PERSONA = False
PERSONA_LOWER_BOUND = 0
CONTEXT_LOWER_BOUND = 0
USE_ATTENTION = False
BEAM_SIZE = 2

def setup_task():
    if IS_ORIGINAL:
        task_name = 'tasks.convai2cosplay.agents:SelfOriginalTeacher'
    else:
        task_name = 'tasks.convai2cosplay.agents:SelfRevisedTeacher'
    return task_name


def setup_trained_weights():
    if IS_ORIGINAL:
        weights_name = '/apdcephfs/share_916081/chencxu/pegg/AAAI/{}/transmitter/pegg-o.model'.format(MODEL_DIR)
    else:
        weights_name = '/apdcephfs/share_916081/chencxu/pegg/AAAI/{}/transmitter/pegg-r.model'.format(MODEL_DIR)
    return weights_name


def setup_args(parser=None):
    parser = base_setup_args(parser)
    task_name = setup_task()
    parser.set_defaults(
        task=task_name,
        datatype='valid',
        hide_labels=False,
        metrics='f1,bleu,c_recall',
    )
    return parser


def eval_f1(opt, print_parser):
    report = eval_model(opt, print_parser)
    print('============================')
    print('Final F1@1: {}, BLEU:  {}, C_Recall:  {}'.format(report['f1'], report['bleu'], report['c_recall']))


if __name__ == '__main__':
    parser = setup_args()
    model_name = setup_trained_weights()
    parser.set_params(
        datapath='/apdcephfs/share_916081/chencxu/pegg/data',
        model='agents.transmitter.transmitter:TransformerAgent',
        model_file=model_name,
        gpu=0,
        batchsize=10,
        beam_size=BEAM_SIZE,
        display_examples=False,
        eval_c_recall=True,
        # ===================
        middle_pool_size=MIDDLE_POOL_SIZE,
        persona_pool_size=PERSONA_POOL_SIZE,
        next_pool_size=NEXT_POOL_SIZE,
        use_context_pool=USE_CONTEXT_POOL,
        use_to_persona_pool=USE_TO_PERSONA_POOL,
        drop_literal_persona=DROP_LITERAL_PERSONA,
        persona_lower_bound=PERSONA_LOWER_BOUND,
        context_lower_bound=CONTEXT_LOWER_BOUND,
        use_attention=USE_ATTENTION,
        persona_pool_r=PERSONA_POOL_R
        # ===================
    )
    opt = parser.parse_args(print_args=False)
    eval_f1(opt, print_parser=parser)
