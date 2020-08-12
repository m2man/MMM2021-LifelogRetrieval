from nltk import pos_tag
from nltk.tokenize import WordPunctTokenizer
from parsedatetime import Constants, Calendar

from .common import *

more_timeofday = {"early; morning": ["dawn", "sunrise", "daybreak"],
                  "morning": ["breakfast"],
                  "evening": ["nightfall", "dusk", "dinner", "dinnertime", "sunset", "twilight"],
                  "noon": ["midday", "lunchtime", "lunch"],
                  "night": ["nighttime"],
                  "afternoon": ["supper", "suppertime", "teatime"]}

timeofday = {"early; morning": "5am-8am",
             "late; morning": "11am-12pm",
             "morning": "5am-12pm",
             "early; afternoon": "1pm-3pm",
             "late; afternoon": "4pm-5pm",
             "midafternoon": "2pm-3pm",
             "afternoon": "12pm-5pm",
             "early; evening": "5pm-7pm",
             "midevening": "7pm-9pm",
             "evening": "5pm-9pm",
             "night": "9pm-4am",
             "noon": "11am-1pm",
             "midday": "11am-1pm",
             "midnight": "11pm-1am",
             "bedtime": "8pm-1am",
             }

for t in more_timeofday:
    for synonym in more_timeofday[t]:
        timeofday[synonym] = timeofday[t]


class TimeTagger:
    def __init__(self):
        regex_lib = Constants()
        self.all_regexes = []
        for key, r in regex_lib.cre_source.items():
            # if key in ["CRE_MODIFIER"]:
            #     self.all_regexes.append(("TIMEPREP", r))
            if key in ["CRE_TIMEHMS", "CRE_TIMEHMS2",
                       "CRE_RTIMEHMS", "CRE_RTIMEHMS"]:
                # TIME (proper time oclock)
                self.all_regexes.append(("TIME", r))
            elif key in ["CRE_DATE", "CRE_DATE3", "CRE_DATE4", "CRE_MONTH", "CRE_DAY", "",
                         "CRE_RDATE", "CRE_RDATE2"]:
                self.all_regexes.append(("DATE", r))  # DATE (day in a month)
            elif key in ["CRE_TIMERNG1", "CRE_TIMERNG2", "CRE_TIMERNG3", "CRE_TIMERNG4",
                         "CRE_DATERNG1", "CRE_DATERNG2", "CRE_DATERNG3"]:
                self.all_regexes.append(("TIMERANGE", r))  # TIMERANGE
            elif key in ["CRE_UNITS", "CRE_QUNITS"]:
                self.all_regexes.append(("PERIOD", r))  # PERIOD
            elif key in ["CRE_UNITS_ONLY"]:
                self.all_regexes.append(("TIMEUNIT", r))  # TIMEUNIT
            elif key in ["CRE_WEEKDAY"]:
                self.all_regexes.append(("WEEKDAY", r))  # WEEKDAY
        # Added by myself
        timeofday_regex = set()
        for t in timeofday:
            if ';' in t:
                t = t.split('; ')[-1]
            timeofday_regex.add(t)

        timeofday_regex = "|".join(timeofday_regex)
        self.all_regexes.append(
            ("TIMEOFDAY", r"\b(" + timeofday_regex + r")\b"))

        # self.all_regexes.append(
        #     ("TIMEOFDAY", r"\b(|afternoon|noon|morning|evening|night|twilight)\b"))
        self.all_regexes.append(
            ("TIMEPREP", r"\b(before|after|while|late|early)\b"))
        self.all_regexes.append(
            ("DATE", r"\b(2015|2016|2018)\b"))
        self.tags = [t for t, r in self.all_regexes]

    def merge_interval(self, intervals):
        if intervals:
            intervals.sort(key=lambda interval: interval[0])
            merged = [intervals[0]]
            for current in intervals:
                previous = merged[-1]
                if current[0] <= previous[1] and current[-1] == previous[-1]:
                    if current[1] > previous[1]:
                        previous[1] = current[1]
                        previous[2] = current[2]
                else:
                    merged.append(current)
            return merged
        return []

    def find_time(self, sent):
        results = []
        for kind, r in self.all_regexes:
            for t in find_regex(r, sent):
                results.append([*t, kind])
        return self.merge_interval(results)

    def tag(self, sent):
        times = self.find_time(sent)
        intervals = dict([(time[0], time[1]) for time in times])
        tag_dict = dict([(time[2], time[3]) for time in times])
        tokenizer = WordPunctTokenizer()
        # for a in [time[2] for time in times]:
        #     tokenizer.add_mwe(a.split())

        # --- FIXED ---
        original_tokens = tokenizer.tokenize(sent)
        original_tags = pos_tag(original_tokens)
        # --- END FIXED ---

        tokens = []
        current = 0
        for span in tokenizer.span_tokenize(sent):
            if span[0] < current:
                continue
            if span[0] in intervals:
                tokens.append(f'__{sent[span[0]: intervals[span[0]]]}')
                current = intervals[span[0]]
            else:
                tokens.append(sent[span[0]:span[1]])
                current = span[1]

        tags = pos_tag(tokens)

        new_tags = []
        for word, tag in tags:
            if word[:2] == '__':
                new_tags.append((word[2:], tag_dict[word[2:]]))
            else:
                tag = [t[1] for t in original_tags if t[0] == word][0]  # FIXED
                new_tags.append((word, tag))
        return new_tags


cal = Calendar()
month2num = {"january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6, "july": 7,
             "august": 8,
             "september": 9, "october": 10, "november": 11, "december": 12}
num2month = dict([(n, m) for (m, n) in month2num.items()])


def get_day_month(date_string):
    if (date_string) in ["2015", "2016", "2018"]:
        return int(date_string), None, None
    today = cal.parse("today")[0]
    date = cal.parse(date_string)[0]
    date_string = date_string.lower()
    y, m, d = date.tm_year, date.tm_mon, date.tm_mday
    if str(y) not in date_string:
        y = None
    if m == today.tm_mon and (num2month[m] not in date_string or str(m) not in date_string):
        m = None
    if str(d) not in date_string:
        d = None
    return y, m, d


def am_pm_to_num(hour):
    minute = 0
    if ':' in hour:
        minute = re.compile(r'\d+(:\d+).*').findall(hour)[0]
        hour = hour.replace(minute, '')
        minute = int(minute[1:])
    if 'am' in hour:
        hour = int(hour.replace('am', ''))
        if hour == 12:
            hour = 0
    elif 'pm' in hour:
        hour = int(hour.replace('pm', '')) + 12
        if hour == 24:
            hour = 12
    return hour, minute


def adjust_start_end(mode, original, hour, minute):
    if mode == "start":
        if original[0] == hour:
            return hour, max(original[1], minute)
        elif hour > original[0]:
            return hour, minute
        else:
            return original
    if original[0] == hour:
        return hour, min(original[1], minute)
    elif hour < original[0]:
        return hour, minute
    else:
        return original