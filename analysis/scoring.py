import argparse
import os
import re
import pickle

from pytorch_transformers import XLNetTokenizer
from rouge_score import rouge_scorer
from utils_summarisation import CNNDailyMailDataset


def load_data(folder_prefix, chunks):
    summary_dict = {}
    story_dict = {}
    t = XLNetTokenizer.from_pretrained("xlnet-base-cased")
    for i in range(chunks):
        folder_path = "{}_{}".format(folder_prefix, i+5)
        dataset = CNNDailyMailDataset(tokenizer=t, data_dir=folder_path)
        for j in range(len(dataset.stories_path)):
            name, story, summary = dataset[j]
            if name not in ['cnn_13abd3e35628071686b33a3b9201cd09da4e1a01', 'cnn_7e94c09d00811e544d2d87deacc98b11de685cda']:
                summary_dict[name] = " ".join(summary)
                story_dict[name] = " ".join(story)
    print(len(summary_dict.values()))
    return summary_dict, story_dict


def load_sequences(folder_prefix, chunks, mode):
    seq_dict = {}
    for i in range(chunks):
        folder_path = "{}_{}_{}".format(folder_prefix, i+5, mode)
        for file in os.listdir(folder_path):
            with open(os.path.join(folder_path, file), "r") as source:
                filename = re.match(r"(.*)_generated.txt", file).group(1)
                seq = source.read()
                seq_dict[filename] = seq
    return seq_dict


def compute_rouge_score(refs, hyps, setting, dat):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = {}
    for name, ref in refs.items():
        if name not in ['cnn_13abd3e35628071686b33a3b9201cd09da4e1a01', 'cnn_7e94c09d00811e544d2d87deacc98b11de685cda',]:
            hyp = hyps[name]
            score = scorer.score(hyp, ref)
            rouge_scores[name] = (score['rouge1'][2], score['rouge2'][2], score['rougeL'][2])
    with open("scoring/scores/rouge_scores_{}_{}.bin".format(dat, setting), "wb") as score_file:
        pickle.dump(rouge_scores, score_file)
    return rouge_scores


parser = argparse.ArgumentParser()
parser.add_argument("--data_pref", type=str)
parser.add_argument("--summaries_folder_pref", type=str)
parser.add_argument("--chunks", type=int)
parser.add_argument("--summary_tokens", type=int)

args = parser.parse_args()

datset = re.match(r"(.*)_data.*\/.*", args.data_pref).group(1)

refs, story_dict = load_data(args.data_pref, args.chunks)

# rouge score for lead-67 (first 67 tokens of article)
# cut story to first 67 tokens
t = XLNetTokenizer.from_pretrained("xlnet-base-cased")
stories_encoded = {name: t.encode(story)[:args.summary_tokens] for name, story in story_dict.items()}
hyps_lead = {name: t.decode(story, clean_up_tokenization_spaces=True) for name, story in stories_encoded.items()}
scores_lead = compute_rouge_score(hyps_lead, refs, "lead", datset)

# rouge score without penalty
mode = "wo_penalty"
hyps_wo_pen = load_sequences(args.summaries_folder_pref, args.chunks, mode)
scores_wo_pen = compute_rouge_score(hyps_wo_pen, refs, "wo_pen", datset)

# rouge score with penalty
mode = "w_penalty"
hyps_w_pen = load_sequences(args.summaries_folder_pref, args.chunks, mode)
scores_w_pen = compute_rouge_score(hyps_w_pen, refs, "w_pen", datset)

for scores in [scores_lead, scores_wo_pen, scores_w_pen]:
    avg_rouge_1 = sum([rouge_1 for name, (rouge_1, _, _) in scores.items()])/len(scores)
    avg_rouge_2 = sum([rouge_2 for name, (_, rouge_2, _) in scores.items()])/len(scores)
    avg_rouge_l = sum([rouge_l for name, (_, _, rouge_l) in scores.items()])/len(scores)
    print(avg_rouge_1, avg_rouge_2, avg_rouge_l)

