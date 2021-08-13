from transformers import pipeline
from nltk.corpus import stopwords
import re
import torch
import operator
from langdetect import detect

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

stopwords = set(stopwords.words('english') + stopwords.words('french'))
device = -1
if torch.cuda.is_available():
    device = 0
classifier1 = pipeline("zero-shot-classification",
                       model="facebook/bart-large-mnli", device=device)
classifier2 = pipeline("zero-shot-classification",
                       model="joeddav/xlm-roberta-large-xnli", device=device)


def importance(topics, scores, frequencies):
    """

    :param topics: list of topics
    :param scores: list of scores(freq + similarity)
    :param frequencies: list of freq
    :return: sorted dict according to freq and sorted dict according to scores in desc order
    """
    dict_ = {}
    for i in range(len(topics)):
        dict_[topics[i]] = scores[i]

    dict_freq = {}
    for i in range(len(topics)):
        dict_freq[topics[i]] = frequencies[i]

    dict_res = dict(sorted(dict_.items(), key=operator.itemgetter(1), reverse=True))
    dict_imp = dict(sorted(dict_freq.items(), key=operator.itemgetter(1), reverse=True))

    for key, value in dict_imp.items():
        if value >= 100:
            dict_imp[key] = 10

        elif value <= 10:
            dict_imp[key] = 1
        else:
            dict_imp[key] = int(value / 10) + 1

    return dict_imp, dict_res


def Scores(keywords, text):
    """

    :param keyword: Keyword
    :param text: piece of text for content grading
    :return: Bart score for keyword w.r.t text
    """
    if detect(text) == 'en':
        dict_ = classifier1(text, keywords, multi_label=True)
    else:
        dict_ = classifier2(text, keywords, multi_label=True)
    return dict_


def bart_scores(dict_keywords, text):
    """

    :param keywords: List of keywords
    :param text: Piece of text
    :return: Dict of bart scores for each keyword
    """
    results = {}
    keywords = [key for key, value in dict_keywords.items()]
    dict_bart = Scores(keywords, text)
    Mapping = {}
    for i in range(len(dict_bart['labels'])):
        Mapping[dict_bart['labels'][i]] = dict_bart['scores'][i]
    for keyword in keywords:
        results[keyword] = Mapping[keyword]

    return results


def grader(dict_imp, dict_bart, text, avg_count):
    """

    :param dict_imp: sorted dict of scores of each topics (freq+similarity) in desc order
    :param dict_bart: sorted dict of bart scores of each topic in desc order
    :param text: piece of text that user will enter
    :param avg_count: avg count of words of compitetors
    :return: gradings
    """
    gradings = ['E-', 'E', 'D-', 'D', 'C-', 'C', 'B-', 'B', 'A-', 'A']
    count = len(text.split())
    print(count)
    weighted_avg = 0
    sum_ = 0
    for key in dict_imp:
        weighted_avg = weighted_avg + dict_imp[key] * dict_bart[key]
        sum_ = sum_ + dict_imp[key]

    weighted_avg = weighted_avg / sum_
    print(weighted_avg)
    if count <= 0.1 * avg_count:
        grades = gradings[:2]
        print(grades)
        if weighted_avg < 0.5:
            return grades[0]
        else:
            return grades[1]

    elif 0.1 * avg_count < count <= 0.5 * avg_count:
        grades = gradings[:5]
        print(grades)
        return grades[int((weighted_avg / 2) * 10)]
    else:
        return gradings[int(weighted_avg * 10)]


def final_presence(text, dict_imp):
    """

    :param text: piece of text
    :param dict_imp: dict of scores of each topics in desc order
    :return: 0,1 labelling of topics with respect to presence
    """
    list_ = [key for key in dict_imp]
    list_res = []
    for item in list_:
        if len(re.findall(item, text)) > 0:
            list_res.append((item, 1))
        else:
            list_res.append((item, 0))

    return list_res
