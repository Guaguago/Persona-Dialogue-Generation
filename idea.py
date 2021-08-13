from nltk.util import ngrams
import nltk

from agents.common.gpt_dictionary import recover_bpe_encoding

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
    if history and 'persona_kws' in history:
        return history['persona_kws']
    else:
        return extract_keywords(persona_str, 30)


def prepare_batch_persona_kw_mask(obs, device):
    batch_persona_kws = torch.tensor([o['persona_kws'] for o in obs if len(o['text2vec']) > 0]).to(device)
    mask = torch.zeros(len(keyword2id)).to(device).unsqueeze(0).expand(len(batch_persona_kws), -1)
    batch_persona_kws_mask = mask.scatter(dim=1, index=batch_persona_kws, src=torch.ones_like(mask))
    batch_persona_kws_mask[:, 0:2] = 0
    return batch_persona_kws_mask


def get_kw_graph_distance_matrix(path, device):
    kw_graph_distance_matrix = torch.ones((len(keyword2id), len(keyword2id))).to(device) * -1
    kw_graph_distance_dict = load_pickle(path)
    for node1, node2 in kw_graph_distance_dict.keys():
        kw_graph_distance_matrix[keyword2id[node1], keyword2id[node2]] = kw_graph_distance_dict[(node1, node2)]
    kw_graph_distance_matrix[torch.isinf(kw_graph_distance_matrix)] = -1.
    max_distance = kw_graph_distance_matrix.max().item()
    kw_graph_distance_matrix = torch.where(kw_graph_distance_matrix == -1,
                                           torch.ones_like(kw_graph_distance_matrix) * max_distance,
                                           kw_graph_distance_matrix)
    min_distance = kw_graph_distance_matrix.view(-1).topk(2680, largest=False)[0].unique()[1].item()
    kw_graph_distance_matrix = torch.where(0 == kw_graph_distance_matrix,
                                           torch.ones_like(kw_graph_distance_matrix) * min_distance,
                                           kw_graph_distance_matrix)
    return {'matrix': kw_graph_distance_matrix, 'max': max_distance, 'min': min_distance}


## kw model forward
def cal_kw_logits(inputs_for_kw_model, keyword_mask_matrix, kw_model):
    batch_context = inputs_for_kw_model['batch_context']
    batch_context_keywords = inputs_for_kw_model['batch_context_keywords']
    batch_context_concepts = inputs_for_kw_model['batch_context_concepts']
    CN_hopk_edge_index = inputs_for_kw_model['CN_hopk_edge_index']
    with torch.no_grad():
        kw_logits, kw_hidden_states = kw_model(CN_hopk_edge_index, batch_context_keywords,
                                               x_utter=batch_context,
                                               x_concept=batch_context_concepts)  # (batch_size, keyword_vocab_size)

        if keyword_mask_matrix is not None:
            batch_vocab_mask = keyword_mask_matrix[batch_context_keywords].sum(dim=1).clamp(min=0,
                                                                                            max=1)  # (batch_size, keyword_vocab_size)
            kw_logits = (1 - batch_vocab_mask) * (
                -5e4) + batch_vocab_mask * kw_logits  # (batch, vocab_size), masked logits

    # top_kws = kw_logits.topk(3, dim=-1)[1]
    # (batch_size, 3), need to convert to vocab token id based on word2id
    return kw_logits, kw_hidden_states


def visualize_samples(data_for_visualization, dict, valid_inds, batch_reply,
                      observations, dictionary, end_idx, report_freq=0.1,
                      labels=None, answers=None, ys=None):
    """Predictions are mapped back to appropriate indices in the batch_reply
       using valid_inds.
       report_freq -- how often we report predictions
    """

    predictions = data_for_visualization['prediction']
    from_context_prob = data_for_visualization['from_context_prob']
    to_persona_prob = data_for_visualization['to_persona_prob']
    concept_probs = (from_context_prob + to_persona_prob) / 2
    final_concept_probs = data_for_visualization['final_concept_probs']
    hybrid_prob = data_for_visualization['hybrid_probs']

    lm_prob = data_for_visualization['lm_probs']

    gate = data_for_visualization['gate'].squeeze().tolist()
    gate.append(0)
    # gate_str = ' '.join(['{:>6.2f}'.format(i) for i in gate.squeeze().tolist()])

    for i in range(len(predictions)):
        # map the predictions back to non-empty examples in the batch
        # we join with spaces since we produce tokens one at a timelab
        curr = batch_reply[valid_inds[i]]
        output_tokens = []
        j = 0
        for c in predictions[i]:
            if c == end_idx and j != 0:
                break
            else:
                output_tokens.append(c)
            j += 1
        if len(output_tokens) == 0:
            output_tokens.append(3)
        if dictionary.default_tok == 'bpe':
            # TODO: judge whether the dictionary has method including parameter recover_bpe
            try:
                curr_pred = dictionary.vec2txt(output_tokens, recover_bpe=True)
            except Exception as e:
                print("You are using BPE decoding but do not recover them in prediction. "
                      "You should support the interface `vec2txt(tensors, recover_bpe)` in your custom dictionary.")
                print("Error Message:\n {}".format(e))
                curr_pred = dictionary.vec2txt(output_tokens)
        else:
            curr_pred = dictionary.vec2txt(output_tokens)

        if curr_pred == '':
            print("Got Error: \n")
            print("output tokens {} length: {}".format(i, len(output_tokens)))
            print("predictions: {} ".format(predictions))
            curr_pred = 'hello how are you today'

        curr['text'] = curr_pred

        if labels is not None and answers is not None and ys is not None:
            y = []
            for c in ys[i]:
                if c == end_idx:
                    break
                else:
                    y.append(c)
            answers[valid_inds[i]] = y
        elif answers is not None:
            answers[valid_inds[i]] = curr_pred

        display_outputs = [i.item() for i in output_tokens]
        line_outputs = recover_bpe_encoding(dict.tokenizer.convert_ids_to_tokens(display_outputs))
        output_str = ' '.join(['{:>7}'.format(i) for i in line_outputs])

        visualize_from_context_probs = visualize_topk_nodes_with_values(from_context_prob, id2keyword, k=10,
                                                                        concept=True)
        visualize_to_persona_probs = visualize_topk_nodes_with_values(to_persona_prob, id2keyword, k=10, concept=True)
        visualize_concept_probs = visualize_topk_nodes_with_values(concept_probs, id2keyword, k=10, concept=True)

        visualize_lm_probs = visualize_topk_nodes_with_values(lm_prob[i], dict, k=5, concept=False)
        visualize_final_concept_probs = visualize_topk_nodes_with_values(final_concept_probs[i], dict, k=5,
                                                                         concept=False)
        visualize_hb_probs = visualize_topk_nodes_with_values(hybrid_prob[i], dict, k=5, concept=False)

        print('=' * 150)
        print('TEXT: ', observations[valid_inds[i]]['text'])
        print('FROM: {}'.format(visualize_from_context_probs))
        print('TOPE: {}'.format(visualize_to_persona_probs))
        print('CONC: {}'.format(visualize_concept_probs))
        print('PRED: {}'.format(output_str))
        gate_str = ' '.join(['{:>7}'.format(w) + '(' + str('{:.4f}'.format(g)) + ')' for w, g in
                             zip(line_outputs, gate)])

        print('GATE: {}'.format(gate_str))
        print('LM & HYBR:')
        print('{}'.format(visualize_lm_probs))
        print('-' * 150)
        print('{}'.format(visualize_final_concept_probs))
        print('-' * 150)
        print('{}'.format(visualize_hb_probs))

    # print("predictions shape: {}".format(predictions.size()))
    # print("batch reply: {}".format(batch_reply))
    return


def visualize_topk_nodes_with_values(tensor, vocab, k=10, concept=False):
    visualization = ''
    tensor = tensor.squeeze()
    if len(tensor.size()) == 1:
        idx = tensor.topk(k)[1].tolist()
        if concept:
            concepts = [vocab[i] for i in idx]
            values = tensor.topk(10)[0].tolist()
            visualization += ' '.join(
                ['{:>6}'.format(c) + '(' + str('{:.3f}'.format(v)) + ')' for c, v in zip(concepts, values)])
    else:
        idx = tensor.topk(5)[1].transpose(0, 1).tolist()
        values = tensor.topk(5)[0].transpose(0, 1).tolist()
        for i, v in zip(idx, values):
            words = recover_bpe_encoding(vocab.tokenizer.convert_ids_to_tokens(i))
            line = ' '.join(['{:>9}'.format(word) + '(' + str('{:.6f}'.format(prob)) + ')'
                             for word, prob in zip(words, v)])
            visualization += (line + '\n')
    return visualization


def cal_walk_probs(kw_logits, kw_mask_matrix, context_kws, softmax, topk=50):
    # neighbors = kw_mask_matrix[context_kws].sum(dim=1).clamp(min=0, max=1)  # (keyword_vocab_size)
    # kw_logits: (vocab, )
    # num_neighbors = neighbors.sum(1).long()
    # has_neighbors = num_neighbors.clamp(0, 1).unsqueeze(1).expand(-1, kw_logits.size(-1))
    # neighbor_filter = kw_logits * ((1 - has_neighbors) + neighbors)
    # logits = walk_logits(neighbor_filter, 10)

    logits = top_k_logits(kw_logits, topk)
    probs = softmax(logits / 3.0)
    return probs


def cal_final_reward(fcg_score, agent_a_coherent_reward, agent_a_language_reward):
    reward_a_list = fcg_score + agent_a_coherent_reward + agent_a_language_reward
    reward_a_baseline = reward_a_list.mean(axis=0, keepdims=True)
    reward_a_list = reward_a_list - reward_a_baseline
    return reward_a_list


def cal_finding_common_ground_score(send_messages_list, receive_messages_list,
                                    trainer_persona, partner_persona, kw_graph_distance_matrix, device):
    # calulate persona ground
    both_persona_str = trainer_persona + ' ' + partner_persona
    persona_concepts = extract_keywords(both_persona_str, 50)
    persona_ground = torch.scatter(input=torch.zeros(2680).to(device), dim=-1,
                                   index=torch.tensor(persona_concepts).to(device),
                                   src=torch.ones_like(torch.tensor(persona_concepts, dtype=torch.float).to(device)))
    persona_ground[0] = 0
    # num_persona_ground_concepts = persona_ground.sum().item()

    batch_size = len(send_messages_list[0])
    num_turn = len(send_messages_list)

    # calculate common ground
    # common_grounds = [[[] for _ in range(num_turn)] for _ in range(batch_size)]
    # num_common_ground_concepts = [[0 for _ in range(num_turn)] for _ in range(batch_size)]
    fcg_scores = [[0 for _ in range(num_turn)] for _ in range(batch_size)]
    common_ground_history = [torch.zeros(2680).to(device) for _ in range(batch_size)]
    for idx_turn, receive_messages, send_messages in zip(
            reversed(range(num_turn)), reversed(receive_messages_list), reversed(send_messages_list)):
        for idx_batch, receive_message, send_message in zip(range(batch_size), receive_messages, send_messages):
            concepts = extract_keywords(send_message + ' ' + receive_message, 50)
            common_ground_current = torch.scatter(input=torch.zeros(2680).to(device), dim=-1,
                                                  index=torch.tensor(concepts).to(device),
                                                  src=torch.ones_like(
                                                      torch.tensor(concepts, dtype=torch.float).to(device)))
            if have_concepts_in(common_ground_current):
                common_ground_current[0] = 0
            common_ground = (common_ground_current + common_ground_history[idx_batch]).clamp(0, 1)
            common_ground_history[idx_batch] = common_ground
            # if no concept, then the common_ground_one_turn[0] will be scattered by 1.

            # num_common_ground_concepts[idx_batch][idx_turn] += common_ground.sum().item()
            precision_score = fcg_precision_score(persona_ground, common_ground, kw_graph_distance_matrix)
            recall_score = fcg_recall_score(persona_ground, common_ground, kw_graph_distance_matrix, 0.6)
            fcg_score = (precision_score + recall_score) / 2
            fcg_scores[idx_batch][idx_turn] += fcg_score / (num_turn - idx_turn + 1)
            # common_grounds[idx_batch][idx_turn] += common_ground.tolist()

    # common_grounds = torch.tensor(common_grounds, dtype=torch.bool).to(device)
    # num_common_ground_concepts = torch.tensor(num_common_ground_concepts).to(device)
    # concepts2persona_ground = (kw_graph_distance_matrix * persona_ground).sum(-1) / num_persona_ground_concepts
    # fcg_precision = (common_grounds * concepts2persona_ground).sum(-1) / num_common_ground_concepts
    return fcg_scores


def fcg_precision_score(persona_ground, common_ground, distance_matrix):
    persona_ground = persona_ground.type(torch.bool)
    common_ground = common_ground.type(torch.bool)
    score = distance_matrix['matrix'][common_ground][:, persona_ground].mean()
    if score.isnan():
        score = distance_matrix['max']
    else:
        score = distance_matrix['max'] - score.item()
    score = score / distance_matrix['max']
    return score


def fcg_recall_score(persona_ground, common_ground, distance_matrix, threshold=0.8):
    persona_ground = persona_ground.type(torch.bool)
    common_ground = common_ground.type(torch.bool)
    scores = distance_matrix['matrix'][common_ground][:, persona_ground]
    if 0 == scores.size(0) * scores.size(1):
        score = 0
    else:
        overlap_ground = torch.where(scores < threshold, torch.ones_like(scores), torch.zeros_like(scores)).sum(
            0).clamp(0, 1)
        score = overlap_ground.sum() / len(overlap_ground)
    return score.item()


def have_concepts_in(common_ground_one_turn):
    return common_ground_one_turn.sum() > 1


def cal_jump_probs(kw_graph_distance_matrix, persona_kws, softmax, topk=50):
    num_persona_kw = persona_kws.sum(-1)
    has_persona_kws = num_persona_kw.clamp(0, 1).unsqueeze(1).expand(-1, len(keyword2id))
    sum_logits = (kw_graph_distance_matrix['matrix'] * (persona_kws.unsqueeze(1))).sum(-1)
    mean_logits = sum_logits / num_persona_kw.unsqueeze(1).clamp(min=1e-6)
    # logits = mean_logits.clamp(min=1e-6).reciprocal()
    # logits = (mean_logits + (1 - has_persona_kws)).reciprocal()
    logits = - mean_logits

    # hardly select topk concepts
    logits = top_k_logits(logits, topk)
    probs = softmax(logits / 0.25)

    nan = probs.isnan().sum()
    if nan > 0:
        print('persona_kws topk: {}'.format(persona_kws.topk(10)[0]))
        print('logits topk: {}'.format(logits.topk(10)[0]))
        print('probs topk: {}'.format(probs.topk(10)[0]))
        print('mean_logits topk: {}'.format(mean_logits.topk(10)[0]))
        print('num_persona_kw: {}'.format(num_persona_kw))

    return probs


def cal_hybrid_probs(walk_probs, jump_probs, hybrid_weights, word2concept_map, concept2words_map,
                     lm_logits, gate, softmax, lm_mask=None):
    # jump or walk [10, 2680]
    assert len(gate.size()) == 3
    assert len(lm_logits.size()) == 3

    batch_size = lm_logits.size(0)
    output_len = lm_logits.size(1)

    lm_probs = softmax(lm_logits / 3.0)
    prob_concept = (jump_probs * hybrid_weights['jump'] + walk_probs * hybrid_weights['walk'])

    concept2words_mask = concept2words_map.ne(0)
    num = concept2words_mask.sum()
    logits = torch.gather(lm_logits.unsqueeze(-2).expand(batch_size, output_len, 2680, -1), dim=-1,
                          index=concept2words_map.type(torch.int64).unsqueeze(0).unsqueeze(0).expand(batch_size,
                                                                                                     output_len, -1,
                                                                                                     -1))

    logits = logits * concept2words_mask
    num_2 = logits.ne(0).sum()

    logits = torch.where(logits.eq(0), torch.ones_like(logits) * -1e5, logits)
    prob_words_given_concept = softmax(logits)
    prob_words_given_concept[:, :, 0:2] = 0
    num_3 = (prob_words_given_concept > 0.00000001).sum()

    prob_words_concept = prob_words_given_concept * (prob_concept.unsqueeze(-1).unsqueeze(1))

    idx = concept2words_map.view(-1).unsqueeze(0).unsqueeze(0).expand(batch_size, output_len, -1).type(torch.int64)
    src = prob_words_concept.view(batch_size, output_len, -1)
    tgt = torch.zeros_like(lm_probs)

    final_probs = tgt.scatter(dim=-1, index=idx, src=src)

    hybrid_probs = cal_hybrid_probs_each_timestep(gate, tgt, lm_probs, lm_mask)
    return hybrid_probs, final_probs


def cal_hybrid_probs_each_timestep(gate, concept_probs, lm_probs, lm_mask):
    # for gate only optimize examples with keywords in the response
    if lm_mask is not None:
        hybrid_probs = lm_probs * (1 - gate * lm_mask.unsqueeze(1)) + gate * lm_mask.unsqueeze(1) * concept_probs
    else:
        hybrid_probs = lm_probs * (1 - gate) + gate * concept_probs
    return hybrid_probs


def cal_word2concept_map(dict, device):
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


def cal_concept2word_map(word_concept_map, device):
    lists = [[0.] * 7]
    for i in range(1, 2680):
        concept2words_idx = torch.where(word_concept_map.eq(i))[0]
        lists.append(torch.cat([concept2words_idx, torch.zeros(7 - len(concept2words_idx)).to(device)]).tolist())
    concept2word_map = torch.tensor(lists)
    return concept2word_map.to(device)
    # # list = [word_concept_map.eq(i).sum().item() for i in range(2680)]
    # # max = torch.tensor(list).topk(10)[0]
    # # concept_word_mask = torch.cat(
    # #     [word_concept_map.eq(concept_id).unsqueeze(0) + 0 for concept_id in range(len(keyword2id))], dim=0)
    # concept_word_mask[0] = 0
    # assert (word_concept_map > 0).sum() == concept_word_mask.sum()
    # return concept_word_mask


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
        kw_model.eval()
        return kw_model.to(device)


## one example for kw model
def prepare_example_for_kw_model(history, text, dict):
    context, last_two_utters = process_context(history, text, dict)
    last_two_utters_keywords = extract_keywords(context, 20)
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


def inputs_for_gate_module(tgt_seq, word2concept_map):
    # len_gate_label = len(src) + len(tgt)

    gate_label = tgt_seq.clone()
    gate_label[gate_label == 0] = -1
    gate_label[gate_label != -1] = (word2concept_map.gather(0, gate_label[gate_label != -1]) != 0) + 0

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
    if text is None or text == '__silence__':
        minus_one = [0] * 30
    else:
        context += text
        minus_one = dict.split_tokenize(context)
        minus_one = [word2id[w] if w in word2id else word2id["<unk>"] for w in minus_one]
        minus_one = pad_sentence(minus_one, 30, word2id["<pad>"])

    if history is None or history == '__silence__':
        # if history['labels']:
        #     history_text = history['labels'][0]
        # else:
        #     history_text = ''
        # dialog = np.array(history['dialog'])
        # idx_beg = np.where(dialog == 40478)[0][-1].item()
        # idx_end = np.where(dialog == 40479)[0][-1].item()
        # history_text = dict.tokenizer.decode(dialog[idx_beg + 1: idx_end])
        minus_two = [0] * 30
    else:
        context = history + ' ' + context
        minus_two = dict.split_tokenize(history)
        minus_two = [word2id[w] if w in word2id else word2id["<unk>"] for w in minus_two]
        minus_two = pad_sentence(minus_two, 30, word2id["<pad>"])

    return context, [minus_two, minus_one]


def extract_keywords(context, max_sent_len):
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


def walk_logits(logits):
    """
    Masks everything but the neighbors of the context concepts
    Used to mask logits such that e^-infinity -> 0 won't contribute to the
    sum of the denominator.
    """
    logits = torch.where(logits == 0.0, torch.ones_like(logits) * -1e10, logits)
    return logits


def top_k_logits(logits, k):
    """
    Masks everything but the k top entries as -infinity (1e10).
    Used to mask logits such that e^-infinity -> 0 won't contribute to the
    sum of the denominator.
    """
    if k == 0:
        return logits
    else:
        values = torch.topk(logits, k)[0]
        batch_mins = values[..., -1:]
        return torch.where(logits < batch_mins, torch.ones_like(logits) * -1e10, logits)
