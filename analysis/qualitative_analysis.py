#!/usr/bin/python3

import pickle
import operator
from collections import Counter

import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt


def load_scores(split):
    """Loading dictionaries containing rouge scores for individual articles.

    Scores are F-scores for ROUGE-1, ROUGE-2 and ROUGE-L, computed between reference summary and
        - lead (article beginning),
        - with penalty and
        - without penalty
    settings for each article of the selected data split.

    Args:
        split (str): data split for which scores have been computed; either dev or test

    Returns:
        rouge_scores_lead (dict): contains F-scores for ROUGE-1, ROUGE-2 and ROUGE-L computed between
            lead (article beginning) and reference summary for each article
        rouge_scores_wo_pen (dict): contains F-scores for ROUGE-1, ROUGE-2 and ROUGE-L computed between
            summaries generated without penalty and reference summary for each article
        rouge_scores_w_pen (dict): contains F-scores for ROUGE-1, ROUGE-2 and ROUGE-L computed between
            summaries generated with penalty and reference summary for each article

    """
    with open("analysis/scores/rouge_scores_{}_lead.bin".format(split), "rb") as score_file:
        rouge_scores_lead = pickle.load(score_file)

    with open("analysis/scores/rouge_scores_{}_wo_pen.bin".format(split), "rb") as score_file:
        rouge_scores_wo_pen = pickle.load(score_file)

    with open("analysis/scores/rouge_scores_{}_w_pen.bin".format(split), "rb") as score_file:
        rouge_scores_w_pen = pickle.load(score_file)

    return rouge_scores_lead, rouge_scores_wo_pen, rouge_scores_w_pen


def compute_and_plot_differences(score_name, scores1, scores2, scores1_name, scores2_name, filename):
    """Computes and plots article-based differences between ROUGE scores of two settings

    Args:
        score_name (str): type of score for which difference is supposed to be computed; rouge-1, rouge-2 or rouge-l
        scores1 (dict): contains ROUGE F-scores for each article of one setting
        scores2 (dict): contains ROUGE F-scores for each article of second setting
        scores1_name (str): setting of scores1
        scores2_name (str): setting of scores2
        filename (str): path to save plot to

    """
    if score_name == "rouge-1":
        pos = 0
    elif score_name == "rouge-2":
        pos = 1
    elif score_name == "rouge-l":
        pos = 2
    differences = {}
    for name, (value1, _, _) in scores1.items():
        diff = value1 - scores2[name][pos]
        differences[name] = diff*100

    differences = Counter(differences)
    high = differences.most_common(3)
    print("Dictionary with 3 highest values:")
    print("Articles: Values")
    for i in high:
        print(i[0], " :", i[1], " ")

    low = differences.most_common()[:-4:-1]
    print("Dictionary with 3 lowest values:")
    print("Articles: Values")
    for i in low:
        print(i[0], " :", i[1], " ")

    objects = np.array(sorted(list(differences.values())))
    y_pos = np.arange(len(objects))
    neg = objects < 0
    pos = objects >= 0
    plt.bar(y_pos[neg], objects[neg], align='center', alpha=0.5, color='red')
    plt.bar(y_pos[pos], objects[pos], align='center', alpha=0.5, color='green')
    plt.title("Diifferences between {} F-scores {} and {}".format(score_name.upper(), scores1_name, scores2_name))
    plt.ylabel("Differences in % points")
    plt.xlabel("Articles")
    plt.savefig(filename)
    plt.clf()


for split in ["dev", "test"]:
    lead, wo_pen, w_pen = load_scores(split)
    print("Difference between with and without penalty")
    compute_and_plot_differences('rouge-1', w_pen, wo_pen, "with", "without penalty", "analysis/plots/distribution_{}_w_wo_pen".format(split))
    print("Difference between lead and without penalty")
    compute_and_plot_differences('rouge-1',wo_pen, lead, "without penalty", "lead",
                                 "analysis/plots/distribution_{}_wo_pen_lead".format(split))
    print("Difference between lead and with penalty")
    compute_and_plot_differences('rouge-1', w_pen, lead, "with penalty", "lead", "analysis/plots/distribution_{}_w_pen_lead".format(split))




