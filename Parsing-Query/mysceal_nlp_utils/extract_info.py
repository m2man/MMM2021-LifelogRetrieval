from autocorrect import Speller
from nltk import pos_tag
from collections import defaultdict
from .common import *
from .pos_tag import *
from .time import *


init_tagger = Tagger(locations)
e_tag = ElementTagger()


def process_time(time_info):
    weekdays = set()
    dates = set()
    start = (0, 0)
    end = (24, 0)
    for time in time_info:
        if time.info == "WEEKDAY":
            weekdays.add(" ".join(time.name))
        elif time.info == "TIMERANGE":
            s, e = " ".join(time.name).split("-")
            start = adjust_start_end("start", start, *am_pm_to_num(s))
            end = adjust_start_end("end", end, *am_pm_to_num(e))
        elif time.info == "TIME":
            if set(time.prep).intersection(["before", "earlier than", "sooner than", "to", "until"]):
                end = adjust_start_end(
                    "end", end, *am_pm_to_num(" ".join(time.name)))
            elif set(time.prep).intersection(["after", "later than", "from"]):
                start = adjust_start_end(
                    "start", start, *am_pm_to_num(" ".join(time.name)))
            else:
                h, m = am_pm_to_num(" ".join(time.name))
                start = adjust_start_end("start", start, h - 1, m)
                end = adjust_start_end("end", end, h + 1, m)
        elif time.info == "DATE":
            dates.add(get_day_month(" ".join(time.name)))
        elif time.info == "TIMEOFDAY":
            t = time.name[0]
            if "early" in time.prep:
                if "early; " + time.name[0] in timeofday:
                    t = "early; " + time.name[0]
            elif "late" in time.prep:
                if "late; " + time.name[0] in timeofday:
                    t = "late; " + time.name[0]
            if t in timeofday:
                s, e = timeofday[t].split("-")
                start = adjust_start_end("start", start, *am_pm_to_num(s))
                end = adjust_start_end("end", end, *am_pm_to_num(e))
            else:
                print(t, f"is not a registered time of day ({timeofday})")
    return list(weekdays), start, end, list(dates)


def extract_info_from_tag(tag_info):
    objects = set()
    verbs = set()
    locations = set()
    region = set()
    # loc, split_keywords, info, weekday, month, timeofday,
    for action in tag_info['action']:
        if action.name:
            verbs.add(" ".join(action.name))
        if action.in_obj:
            objects.add(" ".join(action.in_obj))
        if action.in_loc:
            locations.add(" ".join(action.in_loc))

    for obj in tag_info['object']:
        for name in obj.name:
            objects.add(name)

    for loc in tag_info['location']:
        for name, info in zip(loc.name, loc.info):
            if info == "REGION":
                region.add(name)
            locations.add(name)

    split_keywords = {"descriptions": {"exact": [], "expanded": []},
                      "coco": {"exact": [], "expanded": []},
                      "microsoft": {"exact": [], "expanded": []}}
    objects = objects.difference({""})
    new_objects = set()
    for keyword in objects:
        # if keyword not in all_keywords:
        #     corrected = speller(keyword)
        #     if corrected in all_keywords:
        #         print(keyword, '--->', corrected)
        #         keyword = corrected
        new_objects.add(keyword)
        for kw in microsoft:
            if kw == keyword:
                split_keywords["microsoft"]["exact"].append(kw)
            if intersect(kw, keyword):
                split_keywords["microsoft"]["expanded"].append(kw)
        for kw in coco:
            if kw == keyword:
                split_keywords["coco"]["exact"].append(kw)
            if intersect(kw, keyword):
                split_keywords["coco"]["expanded"].append(kw)
        for kw in all_keywords:
            if kw == keyword:
                split_keywords["descriptions"]["exact"].append(kw)
            if intersect(kw, keyword):
                split_keywords["descriptions"]["expanded"].append(kw)
    weekdays, start_time, end_time, dates = process_time(tag_info["time"])
    return list(new_objects), split_keywords, list(region), list(locations.difference({""})), list(weekdays), start_time, end_time, list(dates)


def extract_info_from_sentence(sent):
    sent = sent.replace(', ', ',')
    tense_sent = sent.split(',')

    past_sent = ''
    present_sent = ''
    future_sent = ''

    for current_sent in tense_sent:
        split_sent = current_sent.split()
        if split_sent[0] == 'after':
            past_sent += ' '.join(split_sent) + ', '
        elif split_sent[0] == 'then':
            future_sent += ' '.join(split_sent) + ', '
        else:
            present_sent += ' '.join(split_sent) + ', '

    past_sent = past_sent[0:-2]
    present_sent = present_sent[0:-2]
    future_sent = future_sent[0:-2]

    list_sent = [past_sent, present_sent, future_sent]

    info = {}
    info['past'] = {}
    info['present'] = {}
    info['future'] = {}

    for idx, tense_sent in enumerate(list_sent):
        tags = init_tagger.tag(tense_sent)
        obj = []
        loc = []
        period = []
        time = []
        timeofday = []
        for word, tag in tags:
            if word not in stop_words:
                if tag in ['NN', 'NNS']:
                    obj.append(word)
                if tag in ['SPACE', 'LOCATION']:
                    loc.append(word)
                if tag in ['PERIOD']:
                    period.append(word)
                if tag in ['TIMEOFDAY']:
                    timeofday.append(word)
                if tag in ['TIME', 'DATE', 'WEEKDAY']:
                    time.append(word)
        if idx == 0:
            info['past']['obj'] = obj
            info['past']['loc'] = loc
            info['past']['period'] = period
            info['past']['time'] = time
            info['past']['timeofday'] = timeofday
        if idx == 1:
            info['present']['obj'] = obj
            info['present']['loc'] = loc
            info['present']['period'] = period
            info['present']['time'] = time
            info['present']['timeofday'] = timeofday
        if idx == 2:
            info['future']['obj'] = obj
            info['future']['loc'] = loc
            info['future']['period'] = period
            info['future']['time'] = time
            info['future']['timeofday'] = timeofday

    return info


def extract_info_from_sentence_full_tag(sent):
    # sent = sent.replace(', ', ',')
    # tense_sent = sent.split(';')
    #
    # past_sent = ''
    # present_sent = ''
    # future_sent = ''
    #
    # for current_sent in tense_sent:
    #     split_sent = current_sent.split()
    #     if split_sent[0] == 'after':
    #         past_sent += ' '.join(split_sent) + ', '
    #     elif split_sent[0] == 'then':
    #         future_sent += ' '.join(split_sent) + ', '
    #     else:
    #         present_sent += ' '.join(split_sent) + ', '
    #
    # past_sent = past_sent[0:-2]
    # present_sent = present_sent[0:-2]
    # future_sent = future_sent[0:-2]
    #
    # list_sent = [past_sent, present_sent, future_sent]

    info = {}
    info['past'] = {}
    info['present'] = {}
    info['future'] = {}

    for idx, tense_sent in enumerate(["", sent]):
        if len(tense_sent) > 2:
            tags = init_tagger.tag(tense_sent)
            info_full = e_tag.tag(tags)
            obj = []
            loc = []
            period = []
            time = []
            timeofday = []

            if len(info_full['object']) != 0:
                for each_obj in info_full['object']:
                    split_term = each_obj.split(', ')
                    if len(split_term) == 2:
                        obj.append(split_term[1])

            if len(info_full['period']) != 0:
                for each_period in info_full['period']:
                    if each_period not in ['after', 'before', 'then', 'prior to']:
                        period.append(each_period)

            if len(info_full['location']) != 0:
                for each_loc in info_full['location']:
                    split_term = each_loc.split('> ')
                    if split_term[0][-3:] != 'not':
                        word_tag = pos_tag(split_term[1].split())
                        final_loc = []
                        for word, tag in word_tag:
                            if tag not in ['DT']:
                                final_loc.append(word)
                        final_loc = ' '.join(final_loc)
                        loc.append(final_loc)

            if len(info_full['time']) != 0:
                for each_time in info_full['time']:
                    if 'from' in each_time or 'to' in each_time:
                        timeofday.append(each_time)
                    else:
                        timetag = init_tagger.time_tagger.tag(each_time)
                        if timetag[-1][1] in ['TIME', 'TIMEOFDAY']:
                            timeofday.append(each_time)
                        elif timetag[-1][1] in ['WEEKDAY', 'DATE']:
                            time.append(timetag[-1][0])

            if idx == 0:
                info['past']['obj'] = obj
                info['past']['loc'] = loc
                info['past']['period'] = period
                info['past']['time'] = time
                info['past']['timeofday'] = timeofday
            if idx == 1:
                info['present']['obj'] = obj
                info['present']['loc'] = loc
                info['present']['period'] = period
                info['present']['time'] = time
                info['present']['timeofday'] = timeofday
            if idx == 2:
                info['future']['obj'] = obj
                info['future']['loc'] = loc
                info['future']['period'] = period
                info['future']['time'] = time
                info['future']['timeofday'] = timeofday

    return info


speller = Speller(lang='en')


def process_query(sent):
    must_not = re.findall(r"-\S+", sent)
    must_not_terms = []
    for word in must_not:
        sent = sent.replace(word, '')
        must_not_terms.append(word.strip('-'))

    tags = init_tagger.tag(sent)
    timeofday = []
    weekday = []
    loc = []
    info = []
    activity = []
    month = []
    region = []
    keywords = []
    for word, tag in tags:
        if word == "airport":
            activity.append("airplane")
        # if word == "candle":
            # keywords.append("lamp")
        if tag == 'TIMEOFDAY':
            timeofday.append(word)
        elif tag == "WEEKDAY":
            weekday.append(word)
        elif word in ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
                      "november", "december"]:
            month.append(word)
        elif tag == "ACTIVITY":
            if word == "driving":
                activity.append("transport")
                info.append("car")
            elif word == "flight":
                activity.append("airplane")
            else:
                activity.append(word)
            keywords.append(word)
        elif tag == "REGION":
            region.append(word)
        elif tag == "KEYWORDS":
            keywords.append(word)
        elif tag in ['NN', 'SPACE', "VBG", "NNS"]:
            if word in ["office", "meeting"]:
                loc.append("work")
            corrected = speller(word)
            if corrected in all_keywords:
                keywords.append(corrected)
            info.append(word)

    split_keywords = {"descriptions": {"exact": [], "expanded": []},
                      "coco": {"exact": [], "expanded": []},
                      "microsoft": {"exact": [], "expanded": []}}

    for keyword in keywords:
        for kw in microsoft:
            if kw == keyword:
                split_keywords["microsoft"]["exact"].append(kw)
            if kw in keyword or keyword in kw:
                split_keywords["microsoft"]["expanded"].append(kw)
        for kw in coco:
            if kw == keyword:
                split_keywords["coco"]["exact"].append(kw)
            if kw in keyword or keyword in kw:
                split_keywords["coco"]["expanded"].append(kw)
        for kw in all_keywords:
            if kw == keyword:
                split_keywords["descriptions"]["exact"].append(kw)
            if kw in keyword or keyword in kw:
                split_keywords["descriptions"]["expanded"].append(kw)

    return loc, split_keywords, info, weekday, month, timeofday, list(set(activity)), list(set(region)), must_not_terms


def process_query2(sent):
    tags = init_tagger.tag(sent)
    original_tags = tags 
    #print(tags)
    tags = e_tag.tag(tags)
    return (original_tags, extract_info_from_tag(tags))

def process_query3(sent):
    tags = init_tagger.tag(sent)
    timeofday = []
    weekdays = []
    locations = []
    info = []
    activity = []
    month = []
    region = []
    keywords = []
    for word, tag in tags:
        if word == "airport":
            activity.append("airplane")
        if tag == 'TIMEOFDAY':
            timeofday.append(word)
        elif tag == "WEEKDAY":
            weekdays.append(word)
        elif word in ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
                      "november", "december"]:
            month.append(word)
        elif tag == "ACTIVITY":
            if word == "driving":
                activity.append("transport")
                info.append("car")
            elif word == "flight":
                activity.append("airplane")
            else:
                activity.append(word)
            keywords.append(word)
        elif tag == "REGION":
            region.append(word)
        elif tag == "KEYWORDS":
            keywords.append(word)
        elif tag in ['NN', 'SPACE', "VBG", "NNS"]:
            if word in ["office", "meeting"]:
                locations.append("work")
            corrected = speller(word)
            if corrected in all_keywords:
                keywords.append(corrected)
            info.append(word)


    split_keywords = {"descriptions": {"exact": [], "expanded": []},
                      "coco": {"exact": [], "expanded": []},
                      "microsoft": {"exact": [], "expanded": []}}
    objects = set(keywords).union(info).difference({""})
    new_objects = set()
    for keyword in objects:
        new_objects.add(keyword)
        for kw in microsoft:
            if kw == keyword:
                split_keywords["microsoft"]["exact"].append(kw)
            if intersect(kw, keyword):
                split_keywords["microsoft"]["expanded"].append(kw)
        for kw in coco:
            if kw == keyword:
                split_keywords["coco"]["exact"].append(kw)
            if intersect(kw, keyword):
                split_keywords["coco"]["expanded"].append(kw)
        for kw in all_keywords:
            if kw == keyword:
                split_keywords["descriptions"]["exact"].append(kw)
            if intersect(kw, keyword):
                split_keywords["descriptions"]["expanded"].append(kw)
    return list(new_objects), split_keywords, list(region), [], list(weekdays), (0, 0), (24, 0), []