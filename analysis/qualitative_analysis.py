import pickle
import operator

import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

with open("scoring/scores/rouge_scores_dev_lead.bin", "rb") as score_file:
    rouge_scores_dev_lead = pickle.load(score_file)

with open("scoring/scores/rouge_scores_dev_wo_pen.bin", "rb") as score_file:
    rouge_scores_dev_wo_pen = pickle.load(score_file)

with open("scoring/scores/rouge_scores_dev_w_pen.bin", "rb") as score_file:
    rouge_scores_dev_w_pen = pickle.load(score_file)

diff_w_wo_pen = {}
for name, (value1, _, _) in rouge_scores_dev_w_pen.items():
    diff = value1 - rouge_scores_dev_wo_pen[name][0]
    diff_w_wo_pen[name] = diff*100

max_name, max_diff = max(diff_w_wo_pen.items(), key=operator.itemgetter(1))
print(max_name, max_diff)
min_name, min_diff = min(diff_w_wo_pen.items(), key=operator.itemgetter(1))
print(min_name, min_diff)

objects = sorted(list(diff_w_wo_pen.values()))
y_pos = np.arange(len(objects))
plt.bar(y_pos, objects, align='center', alpha=0.5)
plt.title("Diifferences between ROUGE-1 F-scores with and without penalty")
plt.ylabel("Differences in % points")
plt.xlabel("Articles")
plt.savefig("scoring/plots/diff_dev_w_wo_pen")

diff_wo_pen_lead = {}
for name, (value1, _, _) in rouge_scores_dev_wo_pen.items():
    diff = value1 - rouge_scores_dev_lead[name][0]
    diff_wo_pen_lead[name] = diff

max_name, max_diff = max(diff_wo_pen_lead.items(), key=operator.itemgetter(1))
print(max_name, max_diff)
print(rouge_scores_dev_wo_pen[max_name])
min_name, min_diff = min(diff_wo_pen_lead.items(), key=operator.itemgetter(1))
print(min_name, min_diff)
print(rouge_scores_dev_wo_pen[min_name])


diff_w_pen_lead = {}
for name, (value1, _, _) in rouge_scores_dev_w_pen.items():
    diff = value1 - rouge_scores_dev_lead[name][0]
    diff_w_pen_lead[name] = diff

max_name, max_diff = max(diff_w_pen_lead.items(), key=operator.itemgetter(1))
print(max_name, max_diff)
min_name, min_diff = min(diff_w_pen_lead.items(), key=operator.itemgetter(1))
print(min_name, min_diff)




