from nltk.util import ngrams
import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import torch
import torch.nn as nn

from kw_model import KW_GNN

_lemmatizer = WordNetLemmatizer()
name_list = ['keyword2id', 'id2keyword', 'node2id', 'word2id', 'CN_hopk_graph_dict']

pkl_list = []
for name in name_list:
    with open('/apdcephfs/private_chencxu/p2/data/KWModel/{}.pkl'.format(name),
              "rb") as f:
        pkl_list.append(pickle.load(f))

keyword2id, id2keyword, node2id, word2id, CN_hopk_graph_dict = pkl_list


# idea interface
def hybrid_kw_and_lm_probs(gate, lm_mask, kw_probs, lm_probs):
    # for gate only optimize examples with keywords in the response
    hybrid_probs = lm_probs * (1 - gate * lm_mask.unsqueeze(1)) + gate * lm_mask.unsqueeze(
        1) * kw_probs
    return hybrid_probs


def kw_word_map(dict):
    map = [0] * 40516
    tokenizer = dict.tokenizer
    keys = tokenizer.decoder.keys()
    count = 0
    for idx in keys:
        word = tokenizer.decode([idx])
        basic_form_word = kw_format([word])[0]
        if basic_form_word in keyword2id:
            map[idx] = keyword2id[basic_form_word]
            count += 1
        else:
            map[idx] = 0
    return map


## gate
def gate(hidden_states):
    sigmoid = nn.Sigmoid()
    gate_linear = nn.Linear(hidden_states.shape[-1], 1)
    gate = sigmoid(gate_linear(hidden_states))
    return gate


def load_kw_model(load_kw_prediction_path, device, use_keywords=True):
    device = torch.device(device)
    if use_keywords:
        kw_model = load_kw_prediction_path.split("/")[-1][:-3]  # keyword prediction model name
        if "GNN" in kw_model:
            kw_model = "KW_GNN"
            use_last_k_utterances = 2

        # load pretrained model
        print("Loading weights from ", load_kw_prediction_path)
        kw_model_checkpoint = torch.load(load_kw_prediction_path, map_location=device)
        if "word2id" in kw_model_checkpoint:
            word2id = 100
            word2id = kw_model_checkpoint.pop("word2id")

        if "model_kwargs" in kw_model_checkpoint:
            kw_model_kwargs = kw_model_checkpoint.pop("model_kwargs")
            kw_model = globals()[kw_model](**kw_model_kwargs)
        kw_model.load_state_dict(kw_model_checkpoint)
        kw_model.to(device)
        kw_model.eval()  # set to evaluation mode, no training required
        return kw_model


## load kw model
kw_model = load_kw_model('saved_model/convai2/KW_GNN_Commonsense.pt', 'cpu')


## kw model forward
def cal_kw_logits(inputs_for_kw_model, keyword_mask_matrix):
    batch_context = inputs_for_kw_model['batch_context']
    batch_context_keywords = inputs_for_kw_model['batch_context_keywords']
    batch_context_concepts = inputs_for_kw_model['batch_context_concepts']
    CN_hopk_edge_index = inputs_for_kw_model['CN_hopk_edge_index']
    with torch.no_grad():
        kw_logits = kw_model(CN_hopk_edge_index, batch_context_keywords,
                             x_utter=batch_context,
                             x_concept=batch_context_concepts)  # (batch_size, keyword_vocab_size)

        if keyword_mask_matrix is not None:
            batch_vocab_mask = keyword_mask_matrix[batch_context_keywords].sum(dim=1).clamp(min=0,
                                                                                            max=1)  # (batch_size, keyword_vocab_size)
            kw_logits = (1 - batch_vocab_mask) * (
                -5e4) + batch_vocab_mask * kw_logits  # (batch, vocab_size), masked logits

    # top_kws = kw_logits.topk(3, dim=-1)[1]
    # (batch_size, 3), need to convert to vocab token id based on word2id
    return kw_logits


## one example for kw model
def inputs_for_KW_model(history, text, dict):
    context, last_two_utters = process_context(history, text, dict)
    last_two_utters_keywords = extract_keywords(context, keyword2id, 20)
    last_two_utters_concepts = extract_concepts(context, node2id, 30, dict)
    return last_two_utters, last_two_utters_keywords, last_two_utters_concepts


## one batch for kw model
def vectorize(obs):
    inputs_for_kw_model = {}
    itr = zip(*[x['kw_model'] for x in obs])
    batch_context = torch.LongTensor(next(itr))
    batch_context_keywords = torch.LongTensor(next(itr))
    batch_context_concepts = torch.LongTensor(next(itr))
    CN_hopk_edge_index = torch.LongTensor(CN_hopk_graph_dict["edge_index"]).transpose(0, 1)  # (2, num_edges)

    inputs_for_kw_model['batch_context'] = batch_context
    inputs_for_kw_model['batch_context_keywords'] = batch_context_keywords
    inputs_for_kw_model['batch_context_concepts'] = batch_context_concepts
    inputs_for_kw_model['CN_hopk_edge_index'] = CN_hopk_edge_index
    return inputs_for_kw_model


def inputs_for_gate_module(src_seq, tgt_seq, dict):
    # len_gate_label = len(src) + len(tgt)
    def app_func(x):
        word = dict.tokenizer.decode([x])
        keyword = kw_format([word])[0]
        if keyword in keyword2id:
            return 1
        else:
            return 0

    gate_label = tgt_seq.clone()
    gate_label[gate_label == 0] = -1
    gate_label[gate_label != -1] = gate_label[gate_label != -1].apply_(app_func)
    # cls_mask = torch.full_like(tgt_seq[:, 0:1], -1)
    # gate_label = torch.cat([torch.full_like(src_seq, -1), gate_label, cls_mask], 1)
    # gate_label = torch.cat([gate_label, cls_mask], 1)

    gate_mask = (gate_label != -1) + 0
    gate_label.masked_fill_(gate_label == -1, 0)

    lm_mask = (gate_label.sum(1) != 0).float().unsqueeze(1)
    gate_mask = lm_mask.expand_as(gate_label) * gate_mask

    gate = {
        'lm_mask': lm_mask,
        'gate_label': gate_label,
        'gate_mask': gate_mask
    }
    return gate


# Others
def get_keyword_mask_matrix():
    keyword_mask_matrix = torch.from_numpy(
        CN_hopk_graph_dict["edge_mask"]).float()  # numpy array of (keyword_vocab_size, keyword_vocab_size)
    print("building keyword mask matrix...")
    keyword_vocab_size = len(keyword2id)
    keyword_mask_matrix[torch.arange(keyword_vocab_size), torch.arange(keyword_vocab_size)] = 0  # remove self loop
    return keyword_mask_matrix


def process_context(history, text, dict):
    context = ''
    if text != '__silence__':
        context += text
        minus_one = dict.split_tokenize(context)
        minus_one = [word2id[w] if w in word2id else word2id["<unk>"] for w in minus_one]
        minus_one = pad_sentence(minus_one, 30, word2id["<pad>"])
    else:
        minus_one = [0] * 30

    if history:
        history = history['labels'][0]
        if history != '__silence__':
            context = history + ' ' + context
            minus_two = dict.split_tokenize(history)
            minus_two = [word2id[w] if w in word2id else word2id["<unk>"] for w in minus_two]
            minus_two = pad_sentence(minus_two, 30, word2id["<pad>"])
        else:
            minus_two = [0] * 30
    else:
        minus_two = [0] * 30

    return context, [minus_two, minus_one]


def extract_keywords(context, keyword2id, max_sent_len):
    simple_tokens = kw_tokenize(context)
    utter_keywords = [keyword2id[w] for w in simple_tokens if w in keyword2id]
    utter_keywords = pad_sentence(utter_keywords, max_sent_len, keyword2id["<pad>"])
    return utter_keywords


def extract_concepts(context, node2id, max_sent_len, dict):
    context = dict.split_tokenize(context)
    utter_concepts = []
    all_utter_ngrams = []
    for n in range(5, 0, -1):
        all_utter_ngrams.extend(ngrams(context, n))
    for w in all_utter_ngrams:
        w = "_".join(w)
        if w in node2id and not any([w in ngram for ngram in utter_concepts]):
            utter_concepts.append(w)
    utter_concepts = [node2id[w] for w in utter_concepts]
    utter_concepts = pad_sentence(utter_concepts, max_sent_len, node2id["<pad>"])
    return utter_concepts


def pad_sentence(sent, max_sent_len, pad_token):
    if len(sent) >= max_sent_len:
        return sent[:max_sent_len]
    else:
        return sent + (max_sent_len - len(sent)) * [pad_token]


def kw_tokenize(string):
    return tokenize(string, [nltk_tokenize, lower, pos_tag, to_basic_form])


def kw_format(string):
    return tokenize(string, [pos_tag, to_basic_form])


def tokenize(example, ppln):
    for fn in ppln:
        example = fn(example)
    return example


def nltk_tokenize(string):
    return nltk.word_tokenize(string)


def lower(tokens):
    if not isinstance(tokens, str):
        return [lower(token) for token in tokens]
    return tokens.lower()


def pos_tag(tokens):
    return nltk.pos_tag(tokens)


def to_basic_form(tokens):
    if not isinstance(tokens, tuple):
        return [to_basic_form(token) for token in tokens]
    word, tag = tokens
    if tag.startswith('NN'):
        pos = 'n'
    elif tag.startswith('VB'):
        pos = 'v'
    elif tag.startswith('JJ'):
        pos = 'a'
    else:
        return word
    return _lemmatizer.lemmatize(word, pos)
