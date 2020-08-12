import time
import json
import os
import functools
from collections import defaultdict

from gensim.models import Word2Vec
from nltk import bigrams
from nltk.corpus import wordnet as wn
from nltk.tag import pos_tag
from .extract_info import init_tagger
from nltk.tokenize import word_tokenize
from .common import *
from numpy import log, sqrt
specials = {"cloudy": "cloud"}

COMMON_PATH = os.getenv("COMMON_PATH")
LSC_PATH = os.getenv("LSC_PATH")

model = Word2Vec.load(f"{COMMON_PATH}/word2vec.model")
map2deeplab = json.load(open(f"{COMMON_PATH}/map2deeplab.json"))
deeplab2simple = json.load(open(f"{COMMON_PATH}/deeplab2simple.json"))
simples = json.load(open(f"{COMMON_PATH}/simples.json"))
synsets = json.load(open(f"{LSC_PATH}/word2vec/wn.txt"))
freq = json.load(open(f"{COMMON_PATH}/stats/all_tags.json"))
freq = {term: log(191404/freq[term]) for term in freq}
specials = {"cloudy": "cloud"}
conceptnet = json.load(open(f"{LSC_PATH}/word2vec/conceptnet/Synonym.json"))

@cache
def morphy(word, sense):
    return wn.morphy(word, sense)


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' %
                  (method.__name__, (te - ts) * 1000))
        return result

    return timed


def get_most_similar(model, word, vocabulary):
    word = word.replace(' ', '_')
    if word in model.wv.vocab:
        vocabulary = [w for w in vocabulary if w in model.wv.vocab]
        if vocabulary:
            return sorted(zip(vocabulary, model.wv.distances(word, vocabulary)), key=lambda x: x[1])
    # else:
    #     print(f"{word} not in word2vec model")
    return []


def hyper(s): return s.hypernyms()


def hypo(s): return s.hyponyms()


def holo(s): return s.member_holonyms() + \
    s.substance_holonyms() + s.part_holonyms()


def mero(s): return s.member_meronyms() + \
    s.substance_meronyms() + s.part_meronyms()


def list_lemmas(synsets_list):
    for synset in synsets_list:
        for lemma in synset.lemmas():
            yield lemma.name()


def update_lemmas(lemma_dicts, synsets_list, depth):
    for lemma in list_lemmas(synsets_list):
        if lemma not in lemma_dicts:
            lemma_dicts[lemma] = depth
        else:
            lemma_dicts[lemma] = min(lemma_dicts[lemma], depth)
    return lemma_dicts


@cache
def get_hypernyms(word):
    syns = wn.synsets(word, pos='n')
    results = {}
    for syn in syns:
        for depth in range(10):
            if depth == 0:
                for lemma in syn.lemmas():
                    results[lemma.name()] = 0
            else:
                results = update_lemmas(
                    results, syn.closure(hyper, depth=depth), depth)
    return results


@cache
def inspect(syns, max_depth):
    result = {"lemmas": {}, "hypernyms": {},
              "hyponyms": {}, "holonyms": {}, "meronyms": {}}
    for syn in syns:
        syn = wn.synset(syn)
        for lemma in syn.lemmas():
            result["lemmas"][lemma.name()] = 0
        for depth in range(max_depth):
            result["hypernyms"] = update_lemmas(
                result["hypernyms"], syn.closure(hyper, depth=depth), depth)
            result["hyponyms"] = update_lemmas(
                result["hyponyms"], syn.closure(hypo, depth=depth), depth)
            result["holonyms"] = update_lemmas(
                result["holonyms"], syn.closure(holo, depth=depth), depth)
            result["meronyms"] = update_lemmas(
                result["meronyms"], syn.closure(mero, depth=depth), depth)
    return result


@cache
def get_similar(word):
    # kws = [kw.keyword for kw in KEYWORDS]
    # if word in kws:
    #     return [word]
    similars = defaultdict(lambda: 10)

    for kw in KEYWORDS:
        is_sim, new_depth = kw.is_similar(word)
        if is_sim:
            similars[kw.keyword] = min(new_depth, similars[kw.keyword])
    return similars


class Keyword:
    def __init__(self, word):
        word = word.replace(' ', '_')
        self.words = None
        if word in ["waiting_in_line", "using_tools"]:
            self.keyword = word
        else:
            if word == "bakery/shop":
                self.keyword = "bakery"
            else:
                self.keyword = word
            depth = 0 if self.keyword in [
                "animal", "person", "food", "color"] else 3
            if self.keyword.replace("_", " ") in synsets:
                syns = synsets[self.keyword.replace("_", " ")].split(', ')[
                    0].split()
            else:
                syns = [syn.name() for syn in wn.synsets(self.keyword)]
            if syns and syns != ["None"]:
                self.words = inspect(syns, depth)

    def is_similar(self, word):
        word = word.replace(' ', '_')
        if word == self.keyword:
            return True, -1
        if self.keyword in ["waiting_in_line", "using_tools"] or self.words is None:
            return False, 10
        if self.words:
            if word in self.words["lemmas"]:
                return True, 0

        for nyms in self.words:
            if word in self.words[nyms]:
                return True, self.words[nyms][word]

        hypernyms = get_hypernyms(word)
        if self.keyword in hypernyms:
            return True, hypernyms[self.keyword]
        else:
            return False, 10


KEYWORDS = [Keyword(kw) for kw in all_keywords]


def to_deeplab(word):
    for kw in map2deeplab:
        if word in map2deeplab[kw][1]:
            yield deeplab2simple[kw]


def get_all_similar(words, keywords, must_not_terms):
    expansion = defaultdict(lambda: [])
    exacts = set([w for t in keywords for w in keywords[t]["exact"]])
    musts = set([w for t in keywords for w in keywords[t]
                 ["exact"] + keywords[t]["expanded"]])
    for word in words:
        word = word.replace('_', ' ')
        possible_words = []
        if word in all_keywords:
            possible_words = [word]
        else:
            for w in to_deeplab(word):
                possible_words.append(w)
        if possible_words:
            for w in possible_words:
                if w in microsoft:
                    keywords["microsoft"]["expanded"].append(w)
                keywords["descriptions"]["expanded"].append(w)
            musts.update(possible_words)
        similars = get_similar(word)
        if similars:
            if word in similars:
                musts.add(word)
            for w in similars:
                expansion[w.replace('_', ' ')].append(0.95 / sqrt(similars[w] if similars[w] > 0 else 1))
        # print('-' * 80)
        # print(word)
        for w, dist in get_most_similar(model, word, all_keywords)[:20]:
            expansion[w.replace('_', ' ')].append(1-dist)
            # if dist < 0.2:
            # musts.add(w)
            # elif dist < 0.5:
            # shoulds.add(w)
            # print(w.ljust(20), round(dist, 2))
    # print(keywords)
    for keyword in keywords["descriptions"]["expanded"]:
        # print('-' * 80)
        # print(keyword)
        for w, dist in get_most_similar(model, keyword, all_keywords)[:20]:
            expansion[w.replace('_', ' ')].append(1-dist)
            # print(w.ljust(20), round(dist, 2))

    final_expansions = []
    score = {}
    for w, dist in expansion.items():
        if w not in must_not_terms:
            mean_dist = sum(dist) / len(dist)
            max_dist = max(dist)
            if mean_dist > 0.5 and max_dist > 0.75:
                musts.add(w)
                score[w] = max_dist
            if mean_dist > 0.5 or max_dist > 0.6:
                final_expansions.append(w)
                score[w] = max_dist

    for w in exacts:
        if w in conceptnet:
            for sym in conceptnet[w]:
                musts.add(sym)
                score[sym] = 1.5

    score = dict(sorted(score.items(), key=lambda x: -x[1])[:10])
    musts = musts.difference(["airplane", "plane"])
    return list(exacts), list(musts), list(score.keys()), score


categories = ["animal", "object", "location", "plant", "person", "food", "room", 'device',
              'communication', 'body_of_water', 'artifact', 'action', 'sky', 'stone',
              'color']


@cache
def check_category(word):
    syns = wn.synsets(word, 'n')
    results = {}
    for syn in syns:
        for depth in range(10):
            if depth == 0:
                for lemma in syn.lemmas():
                    results[lemma.name()] = 0
            else:
                results = update_lemmas(
                    results, syn.closure(hyper, depth=depth), depth)

    return set(categories).intersection(results.keys())


def process_string(info, keywords, must_not_terms):
    pos = pos_tag(info)
    s = []
    # print(pos)
    for w, t in pos:
        if w == "be":
            continue
        if t in ["NN", "VB"]:
            s.append(w)
        elif t in ["JJ", "NNS"]:
            w = morphy(w, 'n')
            if w:
                s.append(w)
        elif t in ["VBG", "VBP", "VBZ", "VBD", "VBN"]:
            w = morphy(w, 'v')
            if w and w != 'be':
                s.append(w)

    def to_take(w):
        category = check_category(w)
        return category and 'color' not in category

    s = [w for w in s if to_take(w)]
    return get_all_similar(s, keywords, must_not_terms)