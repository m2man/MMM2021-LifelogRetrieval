import re

import nltk
from nltk.tokenize import MWETokenizer

from .common import *
from .info_objects import Location, Time, Action, Object
from .time import TimeTagger


##### TAGGER #####
class Tagger:
    def __init__(self, locations):
        self.tokenizer = MWETokenizer()
        self.time_tagger = TimeTagger()
        for a in locations:
            self.tokenizer.add_mwe(a.split())
        # Rules defined
        self.specials = {
            "ACTIVITY": activities.union(["driving", "flight"]),
            "REGION": regions,
            "KEYWORD": [word for word in all_keywords if ' ' in word],
            "LOCATION": locations,
            "QUANTITY": ["at least", "more than", "less than", "at most",
                         "not more than", "a number of"],
            "IN": ["in front of", "called"],
            "NN": [phrase.replace("_", " ") for phrase in list(phrases.keys())],
            "SPACE": ["living room", "fastfood restaurant", "restaurant kitchen", "restaurant",
                      "dining hall", "food court", "butchers shop", "restaurant patio",
                      "coffee shop", "room", "hotel room", "kitchen", "office",
                      "airport", "salon"],
            "POSITION": ["side", "foreground", "background", "right", "left",
                         "image"],
            "TOBE": ["am", "is", "are", "be", "is being", "am being", "are being", "being"],
            "WAS": ["was", "were", "had been", "have been"],
            "TIMEPREP": ["prior to", "then", "earlier than", "later than", "sooner than"],
            "POSITION_PREP": ["near", "distance to"],

        }
        for tag in self.specials:
            for keyword in self.specials[tag]:
                if ' ' in keyword:
                    self.tokenizer.add_mwe(keyword.split())

    def tag(self, sent):
        sent = sent.replace(',', ' , ')
        # tokenize places and address
        token = self.tokenizer.tokenize(sent.lower().split())
        sent = re.sub(r'\b(i)\b', 'I', ' '.join(token))  # replace i to I
        tags = self.time_tagger.tag(sent)
        new_tags = []
        keywords = []
        for word, tag in tags:
            if word in all_keywords:
                keywords.append((word, 'KEYWORD'))
            if '_' in word:
                new_tag = None
                word = word.replace('_', ' ')
                for t in self.specials:
                    if word in self.specials[t]:
                        new_tag = t
                        break
                if new_tag is None:
                    tag = 'LOCATION'
                else:
                    tag = new_tag
            else:
                for t in self.specials:
                    if word in self.specials[t]:
                        tag = t
                        break
            if tag in ['NN', 'NNS']:  # fix error NN after NN --> should be NN after VBG
                try:
                    t1, t2 = new_tags[-1]
                except:
                    t1, t2 = None, None
                if t2 in ['NN', 'NNS']:
                    new_tags[-1] = (t1, 'VBG')
            if tag == 'JJ' and tag in all_keywords:
                tag = 'NN'
            new_tags.append((word, tag))
        return new_tags + keywords


class ElementTagger:
    def __init__(self):
        grammar = r"""
                      WEEKDAY: {<IN><DT><WEEKDAY>}
                      POSITION_PREP: {<JJ>*<POSITION_PREP>}
                      TIME: {<DT>?<RB>?<JJ>*<TIMEOFDAY>}
                      PERIOD: {<QUANTITY>?<PERIOD>}
                      TIMEPREP: {(<RB>|<PERIOD>)?<TIMEPREP>}
                      SPACE: {<SPACE><LOCATION>}
                      LOCATION: {(<OBJECT><LOCATION>|<SPACE>|<NNP>)(<VBD>|<VBN>|<\,>)<DT>?(<OBJECT><LOCATION>|<SPACE>|<NNP>|<REGION>)}
                      LOCATION: {<RB>?<IN>?(<DT>|<PRP\$>)?(<LOCATION>|<SPACE>|<REGION>)+}
                      NN: {(<NN>|<NNS>)+(<IN>(<NNS>|<NN>))?}
                      OBJECT: {(<EX><TOBE>|<QUANTITY>?<CD>?|<DT>|<PRP\$>)<JJ>*<SPACE>?(<NN>|<NNS>|<KEYWORD>)+}
                      TIME: {<TIMEPREP>?(<IN>|<TO>|<RB>)?(<DT>|<PRP\$>)?(<TIMEOFDAY>|<DATE>|<TIME>)}
                      TIME: {(<IN>|<TO>|<RB>)?(<TIMEPREP>|<IN>)?<TIME>}
                      POSITION: {(<IN>|<TO>)?(<DT>|<PRP\$>)<JJ>?<POSITION>}
                                {(<IN>|<TO>)<PRP>}
                      OBnPOS: {(<OBJECT><IN>)?<OBJECT><TOBE>?<JJ>?<POSITION>}
                      VERB_ING: {<VBG><RP>?(<TO>|<IN>)?}
                      VERB_ING: {<VERB_ING>((<CC>|<\,>|<\,><CC>)<VERB_ING>)+}
                      ACTION_ING: {<TOBE>?(<VERB_ING>|<ACTIVITY>)}
                      ACTION_ING: {<TIMEPREP>?<ACTION_ING><RP>?(<TO>|<IN>)?<DT>?((<NN>|<OBJECT>)?(<LOCATION>|<SPACE>)|<OBnPOS>|<OBJECT>)}
                      ACTION_ING: {<TIMEPREP>?<ACTION_ING>(<CC>|<\,>)<ACTION_ING>}
                                  {<TIMEPREP><ACTION_ING>}
                      PAST_VERB: {<RB>?(<WAS><RB>?<VBG>|<VBD>|<VBN>|<VBD><VBG>)<RB>?(<TO>|<IN>)?}
                      PAST_VERB: {<TIMEPREP>?<PAST_VERB>((<CC>|<\,>|<\,><CC>)<PAST_VERB>)+}
                      PAST_ACTION: {<TIMEPREP>?(<CC>|<PRP>)?<PAST_VERB><DT>?((<NN>|<OBJECT>)?(<LOCATION>|<SPACE>)|<OBnPOS>|<OBJECT>)<ACTION_ING>?}
                                   {<PRP><WAS><IN>?(<LOCATION>|<SPACE>)<ACTION_ING>}
                                   {<WAS><ACTION_ING>}
                                   {<TIMEPREP>?(<CC>|<PRP>)?<PAST_VERB><DT>?(?!<PAST_VERB>)}
                      VERB: {(<VB>|<VBG>|<VBP>)<RP>?(<TO>|<IN>)?<DT>?<VB>?}
                      VERB: {<VERB>((<CC>|<\,>|<\,><CC>)<VERB>)+}
                      ACTION: {<TIMEPREP>?<PRP>?<VERB>((<NN>|<OBJECT>)?(<LOCATION>|<SPACE>)|<OBnPOS>|<OBJECT>)?}
                              {<TOBE><LOCATION>?<ACTION_ING>}
                              {<OBJECT><TOBE><VBN>}
                   """
        # OBJECTS: {<OBJECT>((<CC>|<\,>)<OBJECT>)+}
        # PERIOD_ACTION: {<TIMEPREP>(<ACTION_ING>|<PAST_ACTION>|<PAST_VERB>)}
        self.cp = nltk.RegexpParser(grammar)

    def express(self, t):
        if isinstance(t, nltk.tree.Tree):
            expressions = [self.express(l) for l in t]
            return f"({','.join(expressions)})"
        else:
            return t[0]

    def update(self, elements, t):
        if isinstance(t, nltk.tree.Tree):
            label = t.label()
        else:
            label = t[1]
        if label in ["LOCATION"]:
            elements["location"].append(Location(t))
        elif label in ["WEEKDAY", "TIMEOFDAY", "TIME", "TIMERANGE"]:
            elements["time"].append(Time(t))
        elif label in ["PERIOD", "TIMEPREP"]:
            elements["period"].append(Time(t))
        elif label in ["OBJECT", "OBnPOS"]:
            elements["object"].append(Object(t))
        elif label in ["PAST_ACTION", "ACTION_ING", "ACTION"]:
            action_process = Action(t)
            elements["action"].append(action_process)
            for idx in range(len(action_process.obj)):
                elements['object'].append(Object(action_process.obj[idx]))
            for idx in range(len(action_process.loc)):
                elements['location'].append(Location(action_process.loc[idx]))
        # else:
        #    print("UNUSED:", t)
        return elements

    def tag(self, tags):
        elements = {"location": [],
                    "action": [],
                    "object": [],
                    "period": [],
                    "time": []}
        # for key, val in tags:
        #     if key in ['dinner', 'lunch', 'breakfast']:
        #         elements['time'].append(key)

        for n in self.cp.parse(tags):
            self.update(elements, n)
            # print(n)

        # # Convert to string and Filter same result
        # for key, value in elements.items():
        #     for idx_val in range(len(value)):
        #         value[idx_val] = str(value[idx_val])
        #     elements[key] = list(set(value))

        # # Filter useless information
        # idx_obj = 0
        # useless_object = ['someone', 'of']
        # while idx_obj < len(elements['object']):
        #     temp = elements['object'][idx_obj].split(', ')
        #     # object only has 1 character --> useless for search
        #     if len(temp[1]) == 1 or temp[1] in useless_object:
        #         elements['object'].pop(idx_obj)
        #     else:
        #         temp_name = temp[1].split()
        #         for val in useless_object:
        #             try:
        #                 temp_name.remove(val)
        #             except ValueError:
        #                 pass
        #         temp_name = " ".join(temp_name)
        #         elements['object'][idx_obj] = ", ".join([temp[0], temp_name])
        #         idx_obj += 1

        return elements


class ElementTagger2:
    def __init__(self):
        grammar = r"""
                      WEEKDAY: {<IN><DT>?<WEEKDAY>}
                      POSITION_PREP: {<JJ>*<POSITION_PREP>}
                      TIMEOFDAY: {<DT>?<RB>?<JJ>*<TIMEOFDAY>}
                      PERIOD: {<QUANTITY>?<PERIOD>}
                      TIMEPREP: {(<RB>|<PERIOD>)?<TIMEPREP>}
                      SPACE: {<SPACE><LOCATION>}
                      LOCATION: {(<OBJECT><LOCATION>|<SPACE>|<NNP>)(<VBD>|<VBN>|<\,>)<DT>?(<OBJECT><LOCATION>|<SPACE>|<NNP>)}
                      LOCATION: {<RB>?<IN>?(<DT>|<PRP\$>)?(<LOCATION>|<SPACE>)+}
                      NN: {(<NN>|<NNS>)+(<IN>(<NNS>|<NN>))?}
                      OBJECT: {(<EX><TOBE>|<QUANTITY>?<CD>?|<DT>|<PRP\$>)<JJ>*<SPACE>?(<NN>|<NNS>)+}
                      TIME: {<TIMEPREP>?(<IN>|<TO>|<RB>)?(<DT>|<PRP\$>)?(<TIMEOFDAY>|<DATE>|<TIME>)}
                      TIME: {(<IN>|<TO>|<RB>)?(<TIMEPREP>|<IN>)?<TIME>}
                      POSITION: {(<IN>|<TO>)?(<DT>|<PRP\$>)<JJ>?<POSITION>}
                                {(<IN>|<TO>)<PRP>}
                      OBnPOS: {(<OBJECT><IN>)?<OBJECT><TOBE>?<JJ>?<POSITION>}
                      VERB_ING: {<VBG><RP>?(<TO>|<IN>)?}
                      VERB_ING: {<VERB_ING>((<CC>|<\,>|<\,><CC>)<VERB_ING>)+}
                      ACTION_ING: {<TOBE>?(<VERB_ING>|<ACTIVITY>)}
                      ACTION_ING: {<TIMEPREP>?<ACTION_ING><RP>?(<TO>|<IN>)?<DT>?((<NN>|<OBJECT>)?(<LOCATION>|<SPACE>)|<OBnPOS>|<OBJECT>)}
                      ACTION_ING: {<TIMEPREP>?<ACTION_ING>(<CC>|<\,>)<ACTION_ING>}
                                  {<TIMEPREP><ACTION_ING>}
                      PAST_VERB: {<RB>?(<WAS><RB>?<VBG>|<VBD>|<VBN>|<VBD><VBG>)<RB>?(<TO>|<IN>)?}
                      PAST_VERB: {<TIMEPREP>?<PAST_VERB>((<CC>|<\,>|<\,><CC>)<PAST_VERB>)+}
                      PAST_ACTION: {<TIMEPREP>?(<CC>|<PRP>)?<PAST_VERB><DT>?((<NN>|<OBJECT>)?(<LOCATION>|<SPACE>)|<OBnPOS>|<OBJECT>)<ACTION_ING>?}
                                   {<PRP><WAS><IN>?(<LOCATION>|<SPACE>)<ACTION_ING>}
                                   {<WAS><ACTION_ING>}
                                   {<TIMEPREP>?(<CC>|<PRP>)?<PAST_VERB><DT>?(?!<PAST_VERB>)}
                      VERB: {(<VB>|<VBG>|<VBP>)<RP>?(<TO>|<IN>)?<DT>?<VB>?}
                      VERB: {<VERB>((<CC>|<\,>|<\,><CC>)<VERB>)+}
                      ACTION: {<TIMEPREP>?<PRP>?<VERB>((<NN>|<OBJECT>)?(<LOCATION>|<SPACE>)|<OBnPOS>|<OBJECT>)?}
                              {<TOBE><LOCATION>?<ACTION_ING>}
                              {<OBJECT><TOBE><VBN>}
                   """
        # OBJECTS: {<OBJECT>((<CC>|<\,>)<OBJECT>)+}
        # PERIOD_ACTION: {<TIMEPREP>(<ACTION_ING>|<PAST_ACTION>|<PAST_VERB>)}
        self.cp = nltk.RegexpParser(grammar)

    def express(self, t):
        if isinstance(t, nltk.tree.Tree):
            expressions = [self.express(l) for l in t]
            return f"({','.join(expressions)})"
        else:
            return t[0]

    def update(self, elements, t):
        if isinstance(t, nltk.tree.Tree):
            label = t.label()
        else:
            label = t[1]
        if label in ["LOCATION"]:
            elements["location"].append(Location(t))
        elif label in ["WEEKDAY", "TIMEOFDAY", "TIME"]:
            elements["time"].append(Time(t))
        elif label in ["PERIOD", "TIMEPREP"]:
            elements["period"].append(Time(t))
        elif label in ["OBJECT", "OBnPOS"]:
            elements["object"].append(Object(t))
        elif label in ["PAST_ACTION", "ACTION_ING", "ACTION"]:
            action_process = Action(t)
            elements["action"].append(action_process)
            for idx in range(len(action_process.obj)):
                elements['object'].append(Object(action_process.obj[idx]))
            for idx in range(len(action_process.loc)):
                elements['location'].append(Location(action_process.loc[idx]))
        # else:
        #    print("UNUSED:", t)
        return elements

    def tag(self, tags):
        elements = {"location": [],
                    "action": [],
                    "object": [],
                    "period": [],
                    "time": []}
        for key, val in tags:
            if key in ['dinner', 'lunch', 'breakfast']:
                elements['time'].append(key)

        for n in self.cp.parse(tags):
            self.update(elements, n)
            # print(n)

        # Convert to string and Filter same result
        for key, value in elements.items():
            for idx_val in range(len(value)):
                value[idx_val] = str(value[idx_val])
            elements[key] = list(set(value))

        # Filter useless information
        idx_obj = 0
        useless_object = ['someone', 'of']
        while idx_obj < len(elements['object']):
            temp = elements['object'][idx_obj].split(', ')
            # object only has 1 character --> useless for search
            if len(temp[1]) == 1 or temp[1] in useless_object:
                elements['object'].pop(idx_obj)
            else:
                temp_name = temp[1].split()
                for val in useless_object:
                    try:
                        temp_name.remove(val)
                    except ValueError:
                        pass
                temp_name = " ".join(temp_name)
                elements['object'][idx_obj] = ", ".join([temp[0], temp_name])
                idx_obj += 1

        return elements