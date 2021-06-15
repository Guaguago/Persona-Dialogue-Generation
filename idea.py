from nltk.util import ngrams
import nltk
from nltk.stem import WordNetLemmatizer
import pickle

_lemmatizer = WordNetLemmatizer()

name_list = ['keyword2id', 'id2keyword', 'node2id']

pkl_list = []
for name in name_list:
    with open('/Users/xuchen/core/pycharm/project/Persona-Dialogue-Generation/data/KWModel/{}.pkl'.format(name),
              "rb") as f:
        pkl_list.append(pickle.load(f))

keyword2id, id2keyword, node2id, = pkl_list


# idea interface
def inputs_for_KW_model(history, text, dict):
    context = concat_last_2_utterances(history, text)
    utter_keywords = extract_keywords(context, keyword2id, 20)
    utter_concepts = extract_concepts(context, node2id, 30, dict)
    pass


def concat_last_2_utterances(history, text):
    context = ''
    if text != '__silence__':
        context += text
    if history:
        history = history['labels'][0]
        if history != '__silence__':
            context = history + ' ' + context
    return context


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
