from nltk.util import ngrams
import nltk
import random

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
        return extract_concepts(persona_str, 30)


def prepare_batch_persona_kw_mask(obs, device):
    batch_persona_kws = torch.tensor([o['persona_kws'] for o in obs if len(o['text2vec']) > 0]).to(device)
    mask = torch.zeros(len(keyword2id)).to(device).unsqueeze(0).expand(len(batch_persona_kws), -1)
    batch_persona_kws_mask = mask.scatter(dim=1, index=batch_persona_kws, src=torch.ones_like(mask))
    batch_persona_kws_mask[:, 0:2] = 0
    return batch_persona_kws_mask


def get_transition_matrix(path, device):
    MAX = 100
    kw_graph_distance_matrix = torch.ones((len(keyword2id), len(keyword2id))).to(device) * -1
    kw_graph_distance_dict = load_pickle(path)
    for node1, node2 in kw_graph_distance_dict.keys():
        kw_graph_distance_matrix[keyword2id[node1], keyword2id[node2]] = kw_graph_distance_dict[(node1, node2)]
    kw_graph_distance_matrix[torch.isinf(kw_graph_distance_matrix)] = -1.
    max_distance = kw_graph_distance_matrix.max().item()
    kw_graph_distance_matrix = torch.where(kw_graph_distance_matrix.eq(-1),
                                           torch.ones_like(kw_graph_distance_matrix) * MAX,
                                           kw_graph_distance_matrix)
    min_distance = kw_graph_distance_matrix.view(-1).topk(2680, largest=False)[0].unique()[1].item()
    kw_graph_distance_matrix = torch.where(kw_graph_distance_matrix.eq(0),
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

        # if keyword_mask_matrix is not None:
        #     batch_vocab_mask = keyword_mask_matrix[batch_context_keywords].sum(dim=1).clamp(min=0,
        #                                                                                     max=1)  # (batch_size, keyword_vocab_size)
        #     kw_logits = (1 - batch_vocab_mask) * (
        #         -5e4) + batch_vocab_mask * kw_logits  # (batch, vocab_size), masked logits

    # if 0 != kw_logits.max() - kw_logits.min:
    #     print()

    # top_kws = kw_logits.topk(3, dim=-1)[1]
    # (batch_size, 3), need to convert to vocab token id based on word2id
    return kw_logits, kw_hidden_states


def visualize_samples(data_for_visualization, dict, valid_inds, observations):
    i = random.randint(0, len(data_for_visualization) - 1)
    prediction = data_for_visualization[i]['prediction']
    final_pool = data_for_visualization[i]['final_pool']
    # from_context_probs = data_for_visualization[i]['from_context_probs']
    # to_persona_probs = data_for_visualization[i]['to_persona_probs']
    # concept_probs = (to_persona_probs * hybrid_weights['jump'] + from_context_probs * hybrid_weights['walk'])
    concept_word_probs = data_for_visualization[i]['concept_word_probs']
    hybrid_word_probs = data_for_visualization[i]['hybrid_word_probs']
    lm_word_probs = data_for_visualization[i]['lm_word_probs']
    gate = data_for_visualization[i]['gate'].squeeze(-1).tolist()

    #  construct visulization strings
    line_outputs = dict.vec2words(prediction.tolist())
    vis_prediction = ' '.join(['{:>5}'.format(i) for i in line_outputs])
    # vis_from_context_probs = visualize_topk_nodes_with_values(from_context_probs, id2keyword, k=8, concept=True)
    # vis_to_persona_probs = visualize_topk_nodes_with_values(to_persona_probs, id2keyword, k=8, concept=True)
    # vis_concept_probs = visualize_topk_nodes_with_values(concept_probs, id2keyword, k=8, concept=True)
    vis_lm_word_probs = visualize_topk_nodes_with_values(lm_word_probs, dict, k=5, concept=False, matrix=True)
    vis_concept_word_probs = visualize_topk_nodes_with_values(concept_word_probs, dict, k=5, concept=False,
                                                              matrix=True)
    vis_hybrid_word_probs = visualize_topk_nodes_with_values(hybrid_word_probs, dict, k=5, concept=False,
                                                             matrix=True)
    print('=' * 150)
    print('【TEXT】{}'.format(observations[valid_inds[i]]['text']))
    # print('【FROM】{}'.format(vis_from_context_probs))
    # print('【TOPE】{}'.format(vis_to_persona_probs))
    # print('【CONC】{}'.format(vis_concept_probs))
    print('【POOL】{}'.format(len(final_pool)))
    print('【PRED】{}'.format(vis_prediction))
    gate_str = ' '.join(['{:>7}'.format(w) + '(' + str('{:.4f}'.format(g)) + ')' for w, g in
                         zip(line_outputs, gate)])
    print('【GATE】{}'.format(gate_str))
    print('------------------ 【LM Word Probs】 -----------------------')
    print('{}'.format(vis_lm_word_probs))
    print('------------------ 【Concept Word Probs】 ------------------')
    print('{}'.format(vis_concept_word_probs))
    print('------------------ 【Hybrid Word Probs】 -------------------')
    print('{}'.format(vis_hybrid_word_probs))

    return


def visualize_topk_nodes_with_values(tensor, vocab, k=10, concept=False, matrix=False):
    visualization = ''
    if matrix is False:
        idx = tensor.topk(k)[1].tolist()
        if concept:
            concepts = [vocab[i] for i in idx]
            values = tensor.topk(10)[0].tolist()
            visualization += ' '.join(
                ['{:>6}'.format(c) + '(' + str('{:.3f}'.format(v)) + ')' for c, v in zip(concepts, values)])
    else:
        idx = torch.topk(tensor, 5, dim=-1)[1].transpose(0, 1).tolist()
        values = tensor.topk(5)[0].transpose(0, 1).tolist()
        for i, v in zip(idx, values):
            words = vocab.vec2words(i)
            line = ' '.join(['{:>8}'.format(word) + '(' + str('{:.3f}'.format(prob)) + ')'
                             for word, prob in zip(words, v)])
            visualization += (line + '\n')
    return visualization


def cal_final_reward(fcg_score, recall_score, coherent_score, language_score, weights):
    reward_a_list = weights[0] * fcg_score + \
                    weights[1] * recall_score + \
                    weights[2] * coherent_score + \
                    weights[3] * language_score
    reward_a_baseline = reward_a_list.mean(axis=0, keepdims=True)
    reward_a_list = reward_a_list - reward_a_baseline
    return reward_a_list


def cal_finding_common_ground_score(send_messages_list, receive_messages_list,
                                    trainer_persona, partner_persona, kw_graph_distance_matrix, device, r=None):
    # calulate persona ground
    both_persona_str = trainer_persona + ' ' + partner_persona
    persona_concepts = extract_concepts(both_persona_str, 50)
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
    recall_scores = [[0 for _ in range(num_turn)] for _ in range(batch_size)]
    common_ground_history = [torch.zeros(2680).to(device) for _ in range(batch_size)]
    for idx_turn, receive_messages, send_messages in zip(
            reversed(range(num_turn)), reversed(receive_messages_list), reversed(send_messages_list)):
        for idx_batch, receive_message, send_message in zip(range(batch_size), receive_messages, send_messages):
            concepts = extract_concepts(send_message + ' ' + receive_message, 50)
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
            fcg_scores[idx_batch][idx_turn] += precision_score

            recall_score = fcg_recall_score(persona_ground, common_ground, kw_graph_distance_matrix, r)
            recall_scores[idx_batch][idx_turn] += recall_score / (num_turn - idx_turn + 1)
            # common_grounds[idx_batch][idx_turn] += common_ground.tolist()

    # common_grounds = torch.tensor(common_grounds, dtype=torch.bool).to(device)
    # num_common_ground_concepts = torch.tensor(num_common_ground_concepts).to(device)
    # concepts2persona_ground = (kw_graph_distance_matrix * persona_ground).sum(-1) / num_persona_ground_concepts
    # fcg_precision = (common_grounds * concepts2persona_ground).sum(-1) / num_common_ground_concepts
    return fcg_scores, recall_scores


def fcg_precision_score(persona_ground, common_ground, distance_matrix):
    persona_ground = persona_ground.type(torch.bool)
    common_ground = common_ground.type(torch.bool)
    score = distance_matrix['matrix'][common_ground][:, persona_ground].mean()
    if score.isnan():
        score = distance_matrix['max']
    else:
        score = - score.item()
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


def cal_next_pool(logits, context_pool, softmax, topk=300):
    # neighbors = kw_mask_matrix[context_kws].sum(dim=1).clamp(min=0, max=1)  # (keyword_vocab_size)
    # kw_logits: (vocab, )
    # num_neighbors = neighbors.sum(1).long()
    # has_neighbors = num_neighbors.clamp(0, 1).unsqueeze(1).expand(-1, kw_logits.size(-1))
    # neighbor_filter = kw_logits * ((1 - has_neighbors) + neighbors)
    # logits = walk_logits(neighbor_filter, 10)
    has_context = context_pool.eq(1).sum(dim=-1).clamp(0, 1).unsqueeze(-1)  # [bs, 1]
    if topk is not None:
        logits = top_k_logits(logits, topk)
    probs = softmax(logits / 1.0)
    pool = probs > 1e-5
    pool = pool * has_context + (1 - has_context)
    return pool, probs


def cal_concept_pool(concept_logits, distance_matrix, context_concepts, persona_concept_mask, max_pool_size, softmax):
    batch_size = concept_logits.size(0)
    max = distance_matrix['max']
    context_concept_mask = torch.scatter(input=torch.zeros_like(concept_logits, dtype=torch.bool),
                                         src=torch.ones_like(context_concepts, dtype=torch.bool),
                                         index=context_concepts, dim=-1)  # [bs, 2680]
    context_concept_mask[:, 0:2] = 0

    masked_matrix = context_concept_mask.unsqueeze(-1) * \
                    distance_matrix['matrix'].unsqueeze(0).expand(batch_size, -1, -1) * \
                    persona_concept_mask.unsqueeze(1)

    d_c2p = torch.where(masked_matrix.eq(0), torch.ones_like(masked_matrix) * max,
                        masked_matrix).view(batch_size, -1).min(dim=-1)[0]

    masked_matrix = distance_matrix['matrix'] * persona_concept_mask.unsqueeze(1)
    d_n2p = torch.where(masked_matrix.eq(0), torch.ones_like(masked_matrix) * max, masked_matrix).min(dim=-1)[0]

    mask = (d_n2p - d_c2p.unsqueeze(-1)) < 0

    masked_concept_logits = torch.where((concept_logits * mask).eq(0), torch.ones_like(concept_logits) * -1e10,
                                        concept_logits)

    logits = top_k_logits(masked_concept_logits, max_pool_size)
    concept_probs = softmax(logits)
    concept_pool = concept_probs > 1e-5

    return concept_pool, concept_probs


def cal_context_pool(context_concepts, device, lower_bound=0):
    batch_size = context_concepts.size(0)
    context_concept_pool = torch.scatter(input=torch.zeros([batch_size, 2680], dtype=torch.bool, device=device),
                                         src=torch.ones_like(context_concepts, dtype=torch.bool),
                                         index=context_concepts, dim=-1)  # [bs, 2680]
    context_concept_pool[:, 0:2] = 0

    if lower_bound is not None:
        exceed_lower_bound = (context_concept_pool.sum(-1) >= lower_bound).unsqueeze(-1)
        context_concept_pool = context_concept_pool * exceed_lower_bound

    return context_concept_pool


def cal_to_persona_pool(distance_matrix, context_pool, persona_concept_mask, softmax, use_min=False,
                        concept2words_map=None):
    batch_size = context_pool.size(0)
    matrix = distance_matrix['matrix']
    max = distance_matrix['max']

    has_concept = (context_pool + persona_concept_mask).sum(dim=-1).clamp(0, 1).unsqueeze(-1)  # [bs, 1]

    masked_matrix = context_pool.unsqueeze(-1) * matrix.unsqueeze(0).expand(batch_size, -1,
                                                                            -1) * persona_concept_mask.unsqueeze(
        1)  # [bs, 2680, 2680]

    if use_min:
        d_c2p = \
            torch.where(masked_matrix.eq(0), torch.ones_like(masked_matrix) * max, masked_matrix).view(batch_size,
                                                                                                       -1).min(
                dim=-1)[0]
    else:
        d_c2p = masked_matrix.view(batch_size, -1).sum(dim=-1) / ((masked_matrix.view(batch_size, -1) > 0) + 0.).sum(
            -1).clamp(1e-5)
    # d_c2p = masked_matrix.view(batch_size, -1).max(dim=-1)[0]

    masked_matrix = matrix * persona_concept_mask.unsqueeze(1)
    if use_min:
        d_n2p = torch.where(masked_matrix.eq(0), torch.ones_like(masked_matrix) * max, masked_matrix).min(dim=-1)[0]
    else:
        d_n2p = masked_matrix.sum(-1) / (((persona_concept_mask) > 0) + 0.).sum(-1).unsqueeze(-1).clamp(1e-5)
    to_persona_pool = (d_n2p - d_c2p.unsqueeze(-1)) < 0

    # No concepts then this pool = 1
    to_persona_pool = to_persona_pool * has_concept + (1 - has_concept)
    # 去掉 word vocab 没有的
    to_persona_pool = to_persona_pool * concept2words_map.sum(-1).ne(0)

    return to_persona_pool


def cal_expanded_ground(opt, context_concepts, persona_kw_mask, kw_graph_distance_matrix, device,
                        data_for_kw_model, concept2words_map, softmax, kw_mask_matrix, kw_model):
    middle_pool_size = opt.get('middle_pool_size')
    next_pool_size = opt.get('next_pool_size')
    persona_pool_r = opt.get('persona_pool_r')
    persona_pool_size = opt.get('persona_pool_size')
    use_context_pool = opt.get('use_context_pool')
    use_to_persona_pool = opt.get('use_to_persona_pool')
    drop_literal_persona = opt.get('drop_literal_persona')
    context_lower_bound = opt.get('context_lower_bound')
    persona_lower_bound = opt.get('persona_lower_bound')

    # if size of pool < lower_bound, then this pool = 0
    context_ground = cal_context_pool(context_concepts=context_concepts,
                                      lower_bound=context_lower_bound, device=device)

    ground = torch.logical_or(context_ground, persona_kw_mask)

    expanded_ground = expansion_transition(ground=ground, topk=persona_pool_size,
                                           transition_matrix=kw_graph_distance_matrix,
                                           concept2words_map=concept2words_map)

    return expanded_ground


def cal_final_pool(opt, context_concepts, persona_kw_mask, kw_graph_distance_matrix, device,
                   data_for_kw_model, concept2words_map, softmax, kw_mask_matrix, kw_model):
    middle_pool_size = opt.get('middle_pool_size')
    next_pool_size = opt.get('next_pool_size')
    persona_pool_r = opt.get('persona_pool_r')
    persona_pool_size = opt.get('persona_pool_size')
    use_context_pool = opt.get('use_context_pool')
    use_to_persona_pool = opt.get('use_to_persona_pool')
    drop_literal_persona = opt.get('drop_literal_persona')
    context_lower_bound = opt.get('context_lower_bound')
    persona_lower_bound = opt.get('persona_lower_bound')

    # if size of pool < lower_bound, then this pool = 0
    context_pool = cal_context_pool(context_concepts=context_concepts,
                                    lower_bound=context_lower_bound, device=device)

    if middle_pool_size is not None:
        middle_pool = cal_middle_pool(distance_matrix=kw_graph_distance_matrix,
                                      context_pool=context_pool, topk=middle_pool_size,
                                      persona_concept_mask=persona_kw_mask,
                                      softmax=softmax,
                                      concept2words_map=concept2words_map)
        final_pool = middle_pool
    elif next_pool_size is not None:
        kw_logits, kw_hidden_states = cal_kw_logits(data_for_kw_model, kw_mask_matrix, kw_model)
        # if context_pool = 0, then this pool = 1
        next_pool, next_probs = cal_next_pool(logits=kw_logits, topk=next_pool_size,
                                              context_pool=context_pool,
                                              softmax=softmax)
        final_pool = next_pool
    elif persona_pool_r is not None or persona_pool_size is not None:
        # if size of pool < lower_bound, then this pool = 0
        # persona_pool = (context_pool + persona_kw_mask).clamp(0, 1)
        persona_pool, jump_probs = cal_persona_pool(kw_graph_distance_matrix, persona_kw_mask,
                                                    softmax, topk=persona_pool_size,
                                                    r=persona_pool_r, lower_bound=persona_lower_bound,
                                                    concept2words_map=concept2words_map)
        final_pool = persona_pool
        if drop_literal_persona:
            final_pool = (persona_pool - persona_kw_mask).clamp(0, 1)
    else:
        all_concept_pool = torch.ones_like(context_pool)
        final_pool = all_concept_pool

    if use_to_persona_pool:
        # if concept_pool or persona_pool = 0 then this pool = 1
        to_persona_pool = cal_to_persona_pool(distance_matrix=kw_graph_distance_matrix,
                                              context_pool=context_pool,
                                              persona_concept_mask=persona_kw_mask,
                                              softmax=softmax,
                                              concept2words_map=concept2words_map)
        final_pool = (final_pool + to_persona_pool).clamp(0, 1)

    if use_context_pool:
        # if persona_pool=0, then this pool=0
        context_pool = context_pool * ((final_pool.eq(0).sum(-1).clamp(0, 1)).unsqueeze(-1))
        final_pool = (final_pool + context_pool).clamp(0, 1)

    has_concept = final_pool.sum(-1).clamp(0, 1).unsqueeze(-1)
    final_pool = final_pool * has_concept + (1 - has_concept)

    return final_pool


def cal_middle_pool(distance_matrix, context_pool, persona_concept_mask, softmax, topk, concept2words_map):
    max = distance_matrix['max']
    matrix = distance_matrix['matrix']
    end_pool = context_pool + persona_concept_mask

    num_end = end_pool.sum(-1).unsqueeze(-1)

    masked_matrix = matrix * end_pool.unsqueeze(1)
    logits = max - (masked_matrix.sum(-1) / (num_end.clamp(1e-5)))

    # 去掉 GPT 词典中没有的 concept for attention calculation
    logits = logits * concept2words_map.sum(-1).ne(0)

    probs = softmax(top_k_logits(logits, topk))

    # 精确 topk 个数，for attention calculation
    pool = torch.scatter(input=torch.zeros_like(probs),
                         index=probs.topk(topk)[1],
                         src=torch.ones_like(probs),
                         dim=-1)

    return pool


def expansion_transition(ground, topk, transition_matrix, concept2words_map):
    matrix = transition_matrix['matrix']
    max = 100
    masked_matrix = matrix * ground.unsqueeze(1)
    masked_matrix = torch.where(masked_matrix.eq(0), torch.ones_like(masked_matrix) * 100, masked_matrix)
    distance = masked_matrix.min(dim=-1)[0]

    # drop words not in GPT vocab
    distance = distance * concept2words_map.sum(-1).ne(0)
    distance = torch.where(distance.eq(0), torch.ones_like(distance) * max, distance)

    idx = distance.topk(k=topk, largest=False)[1]
    expanded_ground = torch.scatter(input=torch.zeros_like(ground),
                                    src=torch.ones_like(ground),
                                    index=idx, dim=-1)
    return expanded_ground


def cal_persona_pool(kw_graph_distance_matrix, persona_kws, softmax, r=None, lower_bound=0, topk=None,
                     concept2words_map=None):
    probs = None

    expansion_transition(ground=persona_kws, topk=topk, transition_matrix=kw_graph_distance_matrix,
                         concept2words_map=concept2words_map)

    # has_persona = persona_kws.sum(-1).clamp(0, 1).unsqueeze(-1)
    matrix = kw_graph_distance_matrix['matrix']
    max = kw_graph_distance_matrix['max']
    # print([id2keyword[i] for i in torch.where(persona_kws[0].eq(1))[0].tolist()])
    to_persona_matrix = matrix * persona_kws.unsqueeze(1)  # [bs, 2680]
    # mask 后产生 0， 需要将其替换为 max
    to_persona_matrix = torch.where(to_persona_matrix.eq(0), torch.ones_like(to_persona_matrix) * 100,
                                    to_persona_matrix)

    logits = max - to_persona_matrix.min(dim=-1)[0]

    if r is not None:
        pool = (to_persona_matrix.min(dim=-1)[0] < r) + 0.
    else:
        # 去掉 GPT 词典中没有的 concept for attention calculation
        logits = logits * concept2words_map.sum(-1).ne(0)
        logits = torch.where(logits.eq(0), torch.ones_like(logits) * -1e10, logits)

        logits = top_k_logits(logits, topk)
        probs = softmax(logits)
        # 精确 topk 个数，for attention calculation
        pool = torch.scatter(input=torch.zeros_like(probs),
                             index=probs.topk(topk)[1],
                             src=torch.ones_like(probs),
                             dim=-1)

    # print([id2keyword[i] for i in probs[0].topk(topk)[1].tolist()])
    # print([id2keyword[i] for i inpersona_pool.tolist()])
    # persona_pool = probs > 1e-5

    if lower_bound is not None:
        exceed_lower_bound = (persona_kws.sum(-1) >= lower_bound).unsqueeze(-1)
        pool = pool * exceed_lower_bound
    # print([id2keyword[i] for i, j in enumerate(persona_pool[0].tolist()) if j == 1])
    return pool, probs


def cal_lm_word_probs(logits, softmax, temperature=1.0):
    probs = softmax(logits / temperature)
    return probs


def cal_concept_word_probs(logits, final_pool, concept2words_map, softmax, temperature=1.0):
    assert len(logits.size()) == 3
    assert len(final_pool.size()) == 2
    batch_size = logits.size(0)
    output_len = logits.size(1)
    # concept_probs = (jump_probs * hybrid_weights['jump'] + walk_probs * hybrid_weights['walk'])

    concept_word_probs, concept_word_mask = None, None
    if final_pool is not None:
        # [bs, 2680]
        topk_concept2words_map = (final_pool.unsqueeze(-1) * concept2words_map).view(batch_size, -1)
        # assert topk_concept2words_map.size() == (batch_size, 2680 * 7)

        # map to the word vocab
        idx = topk_concept2words_map.unsqueeze(1).expand(-1, output_len, -1).type(torch.int64)
        concept_word_logits_mask = torch.scatter(input=torch.zeros_like(logits, dtype=torch.int64), index=idx,
                                                 src=torch.ones_like(idx), dim=-1)
        concept_word_logits_mask[:, :, 0] = 0
        concept_word_logits = logits * concept_word_logits_mask

        concept_word_logits = torch.where(concept_word_logits.eq(0), torch.ones_like(concept_word_logits) * -1e10,
                                          concept_word_logits)
        concept_word_probs = softmax(concept_word_logits / temperature)

    # topk_concept_idx = concept_probs.topk(topk)[1]
    # topk_concept_probs = concept_probs.topk(topk)[0]
    #
    # #  [bs, topk, 7]
    # topk_concept2words_map = torch.gather(input=concept2words_map.unsqueeze(0).expand(batch_size, -1, -1), dim=1,
    #                                       index=topk_concept_idx.unsqueeze(-1).expand(-1, -1, 7))
    #
    # # topk_concept_probs = torch.gather(input=concept_probs, dim=1, index=topk_concept_idx)
    # topk_concept2words_mask = topk_concept2words_map.ne(0)
    #
    # #  [bs, len, topk, 7]
    # concept_word_logits = torch.gather(lm_logits.unsqueeze(-2).expand(batch_size, output_len, topk, -1), dim=-1,
    #                                    index=topk_concept2words_map.type(torch.int64).unsqueeze(1).expand(
    #                                        batch_size, output_len, topk, -1))
    # concept_word_logits2 = concept_word_logits * topk_concept2words_mask.unsqueeze(1).expand(-1, output_len, -1, -1)

    # if use_lm_logits:
    #     # map to the word vocab
    #     idx = topk_concept2words_map.unsqueeze(1).expand(-1, output_len, -1, -1).view(batch_size, output_len, -1).type(
    #         torch.int64)
    #     src = concept_word_logits2.view(batch_size, output_len, -1)
    #     tgt = torch.zeros_like(lm_logits)
    #     final_logits = tgt.scatter(dim=-1, index=idx, src=src)
    #     final_logits = torch.where(final_logits.eq(0), torch.ones_like(final_logits) * -1e10, final_logits)
    #     final_probs = softmax(final_logits)
    #
    # else:
    #     concept_word_logits3 = torch.where(concept_word_logits2.eq(0), torch.ones_like(concept_word_logits2) * -1e10,
    #                                        concept_word_logits2)
    #     word_probs_given_concept = softmax(concept_word_logits3)
    #     # word_probs_given_concept[:, :, 0:2] = 0
    #
    #     concept_word_probs = word_probs_given_concept * (topk_concept_probs.unsqueeze(-1).unsqueeze(1))
    #
    #     # map to the word vocab
    #     idx = topk_concept2words_map.unsqueeze(1).expand(-1, output_len, -1, -1).view(batch_size, output_len, -1).type(
    #         torch.int64)
    #     src = concept_word_probs.view(batch_size, output_len, -1)
    #     tgt = torch.zeros_like(lm_logits)
    #     final_probs = tgt.scatter(dim=-1, index=idx, src=src)

    return concept_word_probs


def cal_concept_word_probs_attention(embed, hidden, final_pool, concept2words_map, lm_word_probs, softmax, model):
    assert len(hidden.size()) == 3
    assert len(final_pool.size()) == 2

    batch_size = hidden.size(0)
    output_len = hidden.size(1)

    # [bs, 2680, 7]
    concept_words = final_pool.unsqueeze(-1) * concept2words_map

    # [bs, topk, 7]

    # print('concept2words_map: {}'.format(concept2words_map.size()))
    # print('concept_words: {}'.format(concept_words.size()))

    top_concept2word_map = concept2words_map.unsqueeze(0).expand(batch_size, -1, -1)[
        concept_words.sum(-1).gt(0).type(torch.bool)].view(
        batch_size, -1, 7)

    tempt_probs = torch.tensor(lm_word_probs)
    topk = top_concept2word_map.size(1)

    tempt_probs[:, :, 0] = 0
    top_concept_word_p = torch.gather(dim=-1,
                                      input=tempt_probs.unsqueeze(-2).expand(-1, -1, top_concept2word_map.size(-2),
                                                                             -1),
                                      index=top_concept2word_map.unsqueeze(1).expand(-1, output_len, -1, -1).type(
                                          torch.int64))
    # [bs, len, top]
    max_idx = top_concept_word_p.max(-1)[1]

    # [bs, len, top, 1] after lm prob to choose in each concept group
    concept2word_idx = torch.gather(input=top_concept2word_map.unsqueeze(1).expand(-1, output_len, -1, -1),
                                    index=max_idx.unsqueeze(-1),
                                    dim=-1).squeeze(-1)

    concept_embed = embed[concept2word_idx.type(torch.long)]

    # scores = model.linear_ec(concept_embed) + model.linear_hl(hidden).unsqueeze(-1)
    # scores = torch.bmm(concept_embed.view(-1, topk, 768), hidden.unsqueeze(-1).contiguous().view(-1, 768, 1)).view(
    #     batch_size, output_len, topk, -1).squeeze(-1)
    shifted_hidden = torch.cat([hidden[:, 0:1, :], hidden[:, :-1, :]], dim=1)
    scores = torch.matmul(concept_embed, shifted_hidden.unsqueeze(-1)).squeeze(-1)
    # torch.matmul(concept_embed, hidden.unsqueeze(-1))
    # scores = torch.matmul(model.w(concept_embed), hidden.unsqueeze(-1)).squeeze(-1)
    # weighted_sum_concept_embed = (softmax(scores).unsqueeze(-1) * concept_embed).sum(dim=-2)
    weighted_sum_concept_embed = torch.matmul(softmax(scores).unsqueeze(-2), concept_embed).squeeze(-2)
    # weighted_sum_concept_embed = (softmax(scores).unsqueeze(-1) * concept_embed).sum(dim=-2)

    probs = torch.scatter(input=torch.zeros_like(lm_word_probs), src=softmax(scores), index=concept2word_idx, dim=-1)
    return probs, weighted_sum_concept_embed


def cal_hybrid_word_probs(lm_word_probs, concept_word_probs, gate, lm_mask, ablation=False):
    # jump or walk [10, 2680]
    assert len(gate.size()) == 3
    assert len(lm_word_probs.size()) == 3

    # for gate only optimize examples with concepts in the response
    if ablation:
        hybrid_probs = lm_word_probs * (1 - torch.zeros_like(gate)) + torch.zeros_like(gate) * concept_word_probs
    elif lm_mask is not None:
        hybrid_probs = lm_word_probs * (1 - gate * lm_mask.unsqueeze(1)) + gate * lm_mask.unsqueeze(
            1) * concept_word_probs
    else:
        hybrid_probs = lm_word_probs * (1 - gate) + gate * concept_word_probs
    return hybrid_probs


def cal_word2concept_map(dict, device):
    map = [0] * 40516
    tokenizer = dict.tokenizer
    keys = tokenizer.decoder.keys()
    count = 0
    for idx in keys:
        word = tokenizer.decode([idx])
        if word in keyword2id:
            map[idx] = keyword2id[word]
            count += 1
        else:
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
    concept2word_map = torch.tensor(lists, dtype=torch.int64)
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
    last_two_utters_keywords = extract_concepts(context, 20)
    last_two_utters_concepts = extract_concepts_grams(context, node2id, 30, dict)
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


def inputs_for_gate_module(tgt_seq, word2concept_map, pool):
    # len_gate_label = len(src) + len(tgt)
    bs = tgt_seq.size(0)
    pool = torch.gather(input=pool, index=word2concept_map.unsqueeze(0).expand(bs, -1), dim=-1)

    gate_label = tgt_seq.clone()
    gate_label[gate_label == 0] = -1

    tail = gate_label * gate_label.eq(-1)

    gate_label = torch.gather(input=pool, index=gate_label * gate_label.ne(-1), dim=-1) + 0
    gate_label = gate_label + tail
    # gate_label[gate_label != -1] = (pool.gather(-1, gate_label[gate_label != -1])) + 0

    # gate_label[gate_label != -1] = (word2concept_map.gather(0, gate_label[gate_label != -1]) != 0) + 0

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


def extract_concepts(context, max_sent_len):
    rest = []
    concept_id = []
    tokenized = concept_tokenize(context)

    for w in tokenized:
        if w in keyword2id:
            concept_id.append(keyword2id[w])
            continue
        rest.append(w)

    basic = concept_to_basic(rest)

    for w in basic:
        if w in keyword2id:
            concept_id.append(keyword2id[w])

    concept_id = pad_sentence(concept_id, max_sent_len, keyword2id["<pad>"])

    return concept_id


def extract_concepts_grams(context, node2id, max_sent_len, dict):
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


def concept_tokenize(string):
    return tokenize(string, [nltk_tokenize, lower])


def concept_to_basic(string):
    return tokenize(string, [pos_tag, to_basic_form])


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


def walk_logits(logits, k):
    """
    Masks everything but the neighbors of the context concepts
    Used to mask logits such that e^-infinity -> 0 won't contribute to the
    sum of the denominator.
    """
    # d = 2 / ((1 + k) * k)
    # p = [i * d for i in range(1, k + 1)]
    # pp = torch.tensor(p).unsqueeze(0).expand(logits.size(0), -1)
    # idx = torch.topk(logits, k)[1]
    # probs = torch.scatter(input=torch.zeros_like(logits), dim=-1, index=idx, src=pp)
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
