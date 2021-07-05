from nltk.util import ngrams
import nltk

nltk.data.path.append('/apdcephfs/share_916081/chencxu/pegg/data/nltk_data')
from nltk.stem import WordNetLemmatizer
import pickle
import torch
import torch.nn as nn
import numpy as np
from kw_model import KW_GNN

_lemmatizer = WordNetLemmatizer()
name_list = ['keyword2id', 'id2keyword', 'node2id', 'word2id', 'CN_hopk_graph_dict']

pkl_list = []
for name in name_list:
    with open('/apdcephfs/share_916081/chencxu/pegg/data/kw_model/{}.pkl'.format(name),
              "rb") as f:
        pkl_list.append(pickle.load(f))

keyword2id, id2keyword, node2id, word2id, CN_hopk_graph_dict = pkl_list


# idea interface
def prepare_example_persona_kws(history, persona_str):
    if history:
        return history['persona_kws']
    else:
        return extract_keywords(persona_str, keyword2id, 30)


def prepare_batch_persona_kw_mask(obs, device):
    batch_persona_kws = torch.tensor([o['persona_kws'] for o in obs if len(o['text2vec']) > 0]).to(device)
    mask = torch.zeros(len(keyword2id)).to(device).unsqueeze(0).expand(len(batch_persona_kws), -1)
    batch_persona_kws_mask = mask.scatter(dim=1, index=batch_persona_kws, src=torch.ones_like(mask))
    batch_persona_kws_mask[:, 0:2] = 0
    return batch_persona_kws_mask


def get_kw_graph_distance_matrix(path, device):
    kw_graph_distance_matrix = np.ones((len(keyword2id), len(keyword2id))) * 1000
    kw_graph_distance_dict = load_pickle(path)
    for node1, node2 in kw_graph_distance_dict.keys():
        kw_graph_distance_matrix[keyword2id[node1], keyword2id[node2]] = kw_graph_distance_dict[(node1, node2)]

    kw_graph_distance_matrix[np.isinf(kw_graph_distance_matrix)] = 1000
    return torch.from_numpy(kw_graph_distance_matrix).to(device)


## kw model forward
def cal_kw_logits(inputs_for_kw_model, keyword_mask_matrix, kw_model):
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


def cal_walk_probs(kw_logits, kw_mask_matrix, context_kws, softmax):
    neighbors = kw_mask_matrix[context_kws].sum(dim=1).clamp(min=0, max=1)  # (keyword_vocab_size)
    # kw_logits: (vocab, )
    num_neighbors = neighbors.sum(1).long()
    has_neighbors = num_neighbors.clamp(0, 1).unsqueeze(1).expand(-1, kw_logits.size(-1))
    neighbor_filter = kw_logits * ((1 - has_neighbors) + neighbors)
    return softmax(neighbor_filter)


def cal_jump_probs(kw_graph_distance_matrix, persona_kws, softmax):
    num_persona_kw = persona_kws.sum(-1)
    has_persona_kws = num_persona_kw.clamp(0, 1).unsqueeze(1).expand(-1, len(keyword2id))
    sum_logits = (kw_graph_distance_matrix * (persona_kws.unsqueeze(1))).sum(-1)
    mean_logits = sum_logits / num_persona_kw.unsqueeze(1)
    logits = (mean_logits + (1 - has_persona_kws)).reciprocal()
    return softmax(logits/0.2)


def hybrid_kw_and_lm_probs(gate, lm_mask, kw_probs, lm_probs):
    # for gate only optimize examples with keywords in the response
    hybrid_probs = lm_probs * (1 - gate * lm_mask.unsqueeze(1)) + gate * lm_mask.unsqueeze(
        1) * kw_probs
    return hybrid_probs


def kw_word_map(dict, device):
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
    return torch.tensor(map).to(device)


def load_kw_model(load_kw_prediction_path, device, use_keywords=True):
    if use_keywords:
        kw_model = load_kw_prediction_path.split("/")[-1][:-3]  # keyword prediction model name
        if "GNN" in kw_model:
            kw_model = "KW_GNN"
        print("Loading weights from ", load_kw_prediction_path)
        kw_model_checkpoint = torch.load(load_kw_prediction_path, map_location=device)
        if "word2id" in kw_model_checkpoint:
            kw_model_checkpoint.pop("word2id")

        if "model_kwargs" in kw_model_checkpoint:
            kw_model_kwargs = kw_model_checkpoint.pop("model_kwargs")
            kw_model = globals()[kw_model](**kw_model_kwargs)
        kw_model.load_state_dict(kw_model_checkpoint)
        kw_model.train()
        return kw_model.to(device)


## one example for kw model
def prepare_example_for_kw_model(history, text, dict):
    context, last_two_utters = process_context(history, text, dict)
    last_two_utters_keywords = extract_keywords(context, keyword2id, 20)
    last_two_utters_concepts = extract_concepts(context, node2id, 30, dict)
    return last_two_utters, last_two_utters_keywords, last_two_utters_concepts


## one batch for kw model
def prepare_batch_for_kw_model(obs, device):
    inputs_for_kw_model = {}

    for_kw_models = [x['kw_model'] for x in obs if len(x['text2vec']) > 0]
    itr = zip(*for_kw_models)
    try:
        batch_context = torch.tensor(next(itr)).to(device)
        batch_context_keywords = torch.tensor(next(itr)).to(device)
        batch_context_concepts = torch.tensor(next(itr)).to(device)
        CN_hopk_edge_index = torch.tensor(CN_hopk_graph_dict["edge_index"]).transpose(0, 1).to(device)  # (2, num_edges)

        inputs_for_kw_model['batch_context'] = batch_context
        inputs_for_kw_model['batch_context_keywords'] = batch_context_keywords
        inputs_for_kw_model['batch_context_concepts'] = batch_context_concepts
        inputs_for_kw_model['CN_hopk_edge_index'] = CN_hopk_edge_index
    except:
        inputs_for_kw_model = None
    return inputs_for_kw_model


def inputs_for_gate_module(tgt_seq, vocab_map):
    # len_gate_label = len(src) + len(tgt)

    gate_label = tgt_seq.clone()
    gate_label[gate_label == 0] = -1
    gate_label[gate_label != -1] = (vocab_map.gather(0, gate_label[gate_label != -1]) != 0) + 0

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
def get_keyword_mask_matrix(device):
    keyword_mask_matrix = torch.from_numpy(
        CN_hopk_graph_dict["edge_mask"]).float().to(device)  # numpy array of (keyword_vocab_size, keyword_vocab_size)
    print("building keyword mask matrix...")
    keyword_vocab_size = len(keyword2id)
    keyword_mask_matrix[torch.arange(keyword_vocab_size).to(device), torch.arange(keyword_vocab_size).to(
        device)] = 0  # remove self loop
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


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)
