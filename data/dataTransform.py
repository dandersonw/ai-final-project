import re
import os

from pathlib import Path


def _load_intern_dict():
    resource_path = Path(os.environ['AI_RESOURCE_PATH'])
    word_embedding_path = resource_path / 'glove.840B.300d.txt'
    intern_dict = {}
    with open(word_embedding_path, mode='r') as f:
        for l in f:
            tokens = l.split(' ')
            word = tokens[0]
            idx = len(intern_dict)
            intern_dict[word] = idx
    return intern_dict


TOKENIZER = re.compile('-| ')
intern_dict = _load_intern_dict()


def transform_name(datum):
    name = datum[1]
    tokens = []
    for c in name:
        tokens.append(ord(c))
    word_tokens = []
    for w in TOKENIZER.split(name):
        word_tokens.append(intern_dict.get(w, intern_dict['unk']))
    uncased_word_tokens = []
    for w in TOKENIZER.split(name):
        uncased_word_tokens.append(intern_dict.get(w.lower(), intern_dict['unk']))
    return (tokens, word_tokens, uncased_word_tokens)


# Entity = 1, Spell = 2, Enchantment = 3, Artifact = 4
label_mapping = {
    "Creature": 1,
    "Planeswalker": 1,
    "Instant": 2,
    "Sorcery": 2,
    "Enchantment": 3,
    "Artifact": 4
}


def transform_data(datum):
    record = {}
    tokens, word_tokens, uncased_word_tokens = transform_name(datum)
    record['tokens'] = tokens
    record['word_tokens'] = word_tokens
    record['uncased_word_tokens'] = uncased_word_tokens
    record['length'] = len(datum[1])
    record['image'] = datum[4]
    record['label'] = label_mapping[datum[6]]
    return record
