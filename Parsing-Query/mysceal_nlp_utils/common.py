import json
import re

import nltk
from nltk.corpus import stopwords
import os
import shelve

stop_words = stopwords.words('english')
stop_words += [',', '.']

COMMON_PATH = os.getenv("COMMON_PATH")
simpletime = ['at', 'around', 'about', 'on']
period = ['while', "along", "as"]

preceeding = ['before', "afore"]
following = ['after']
location = ['across', 'along', 'around', 'at', 'behind', 'beside', 'near', 'by', 'nearby', 'close to',
            'next to', 'from', 'in front of', 'inside', 'in', 'into', 'off', 'on',
            'opposite', 'out of', 'outside', 'past', 'through', 'to', 'towards']

all_words = period + preceeding + following
all_prep = simpletime + period + preceeding + following
pattern = re.compile(f"\s?({'|'.join(all_words)}+)\s")

# grouped_info_dict = json.load(open(f"{COMMON_PATH}/grouped_info_dict.json"))
# locations = set([img["location"].lower()
#                  for img in grouped_info_dict.values()])
# regions = set([w.strip().lower() for img in grouped_info_dict.values()
#                for w in img["region"]])
# deeplab = set([w.replace('_', ' ') for img in grouped_info_dict.values()
#                for w in img["deeplab"]])
# coco = set([w.replace('_', ' ') for img in grouped_info_dict.values()
#             for w in img["coco"]])
# attributes = set([w.replace('_', ' ') for img in grouped_info_dict.values()
#                   for w in img["attributes"]])
# category = set([w.replace('_', ' ') for img in grouped_info_dict.values()
#                 for w in img["category"]])
# microsoft = set([w.replace('_', ' ') for img in grouped_info_dict.values()
#                  for w in img["microsoft_tags"] + img["microsoft_descriptions"]])

# all_keywords = regions | deeplab | coco | attributes | category | microsoft
# old_keywords = regions | deeplab | coco | attributes | category
# json.dump(list(keywords), open(f'{COMMON_PATH}/all_keywords.json', 'w'))
locations = json.load(open(f'{COMMON_PATH}/locations.json'))
regions = json.load(open(f'{COMMON_PATH}/regions.json'))
microsoft = json.load(open(f'{COMMON_PATH}/microsoft.json'))
coco = json.load(open(f'{COMMON_PATH}/coco.json'))

all_keywords = json.load(open(f'{COMMON_PATH}/all_keywords.json'))
all_address = '|'.join([re.escape(a) for a in locations])
activities = set(["walking", "airplane", "transport", "running"])
phrases = json.load(open(f'{COMMON_PATH}/phrases.json'))


def find_regex(regex, text, escape=False):
    regex = re.compile(regex, re.IGNORECASE + re.VERBOSE)
    for m in regex.finditer(text):
        result = m.group()
        start = m.start()
        while len(result) > 0 and result[0] == ' ':
            result = result[1:]
            start += 1
        while len(result) > 0 and result[-1] == ' ':
            result = result[:-1]
        yield (start, start + len(result), result)


def flatten_tree(t):
    return " ".join([l[0] for l in t.leaves()])


def flatten_tree_tags(t, pos):
    if isinstance(t, nltk.tree.Tree):
        if t.label() in pos:
            return [flatten_tree(t), t.label()]
        else:
            return [flatten_tree_tags(l, pos) for l in t]
    else:
        return t


def cache(_func=None, *, file_name=None, separator='_'):
    """
    if file_name is None, just cache it using memory, else save result to file
    """
    if file_name:
        d = shelve.open(file_name)
    else:
        d = {}

    def decorator(func):
        def new_func(*args, **kwargs):
            param = separator.join(
                [str(arg) for arg in args] + [str(v) for v in kwargs.values()])
            if param not in d:
                d[param] = func(*args, **kwargs)
            return d[param]
        return new_func

    if _func is None:
        return decorator
    else:
        return decorator(_func)


freq = json.load(open(f"{COMMON_PATH}/stats/all_tags.json"))
overlap = json.load(open(f"{COMMON_PATH}/stats/overlap_all.json"))


@cache
def intersect(word, keyword):
    if word == keyword:
        return True
    try:
        if word in keyword.split(' '):
            cofreq = overlap[word][keyword]
            return True
            return cofreq / freq[word] > 0.8
        elif keyword in word.split(' '):
            cofreq = overlap[keyword][word]
            return True
            return cofreq / freq[keyword] > 0.8
    except KeyError:
        pass
    return False