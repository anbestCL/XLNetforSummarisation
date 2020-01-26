#!/usr/bin/python3

import argparse
import os
import re
import pickle

from pytorch_transformers import XLNetTokenizer
from rouge_score import rouge_scorer
from utils_summarisation import CNNDailyMailDataset


def load_data(folder_prefix, chunks):
    """Loading summaries and stories of articles, excluding zero-length stories or summaries.

    Args:
        folder_prefix (str): prefix of folder path
        chunks (int): number of data chunks to load data from

   Returns:
       summary_dict (dict): key: article_name, value: summary text
       story_dict (dict): key: article_name, value: story text

    """
    summary_dict = {}
    story_dict = {}
    t = XLNetTokenizer.from_pretrained("xlnet-base-cased")
    for i in range(chunks):
        folder_path = "{}_{}".format(folder_prefix, i)
        dataset = CNNDailyMailDataset(tokenizer=t, data_dir=folder_path)
        for j in range(len(dataset.stories_path)):
            name, story, summary = dataset[j]
            if name not in ['cnn_13abd3e35628071686b33a3b9201cd09da4e1a01', 'cnn_7e94c09d00811e544d2d87deacc98b11de685cda']:
                summary_dict[name] = " ".join(summary)
                story_dict[name] = " ".join(story)
    return summary_dict, story_dict


def load_sequences(folder_prefix, chunks, mode, prompt=None):
    """Loads generated summaries.

    Args:
        folder_prefix (str): prefix of folder path where summaries are saved
        chunks (int): number of data chunks to load data from
        mode (str): setting the summaries have been generated in, 'w_penalty' or 'wo_penalty'

    Returns:
        seq_dict (dict): key: article_name, value: summary text

    """
    t = XLNetTokenizer.from_pretrained("xlnet-base-cased")
    seq_dict = {}
    for i in range(chunks):
        folder_path = "{}_{}_{}".format(folder_prefix, i, mode)
        for file in os.listdir(folder_path):
            with open(os.path.join(folder_path, file), "r") as source:
                filename = re.match(r"(.*)_generated.txt", file).group(1)
                seq = source.read()
                if mode == "w_penalty_prompt":
                    seq_split = t.encode(seq)[prompt:]
                    seq = t.decode(seq_split)
                seq_dict[filename] = seq
    return seq_dict


def compute_rouge_score(refs, hyps, filename):
    """Computes ROUGE scores between reference and generated summaries

    Args:
        refs (dict): key: article name, value: reference summary
        hyps (dict): key: article_name, value: generated summary
        filename (str): path where to save scores to

    Returns:
        rouge_scores (dict): key: article_name, value: (ROUGE-1 F-score, ROUGE-2 F-score, ROUGE-L F-score)

    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = {}
    for name, ref in refs.items():
        if name not in ['cnn_13abd3e35628071686b33a3b9201cd09da4e1a01', 'cnn_7e94c09d00811e544d2d87deacc98b11de685cda',]:
            hyp = hyps[name]
            score = scorer.score(hyp, ref)
            rouge_scores[name] = (score['rouge1'][2], score['rouge2'][2], score['rougeL'][2])
    with open(filename, "wb") as score_file:
        pickle.dump(rouge_scores, score_file)
    return rouge_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_pref", type=str, help="Prefix of data folder")
    parser.add_argument("--summaries_folder_pref", type=str, help="Prefix of folder containing summaries")
    parser.add_argument("--chunks", type=int, help="Number of data chunks to choose")
    parser.add_argument("--summary_tokens", type=int, help="Length of generated summary")
    parser.add_argument("--prompt", type=int, help="Number of tokens used as prompt")


    args = parser.parse_args()

    datset = re.match(r"data\/(.*)_data.*\/.*", args.data_pref).group(1)

    refs, story_dict = load_data(args.data_pref, args.chunks)

    # rouge score for lead-67 (first 67 tokens of article)
    # cut story to first 67 tokens
    t = XLNetTokenizer.from_pretrained("xlnet-base-cased")
    stories_encoded = {name: t.encode(story)[:args.summary_tokens] for name, story in story_dict.items()}
    hyps_lead = {name: t.decode(story, clean_up_tokenization_spaces=True) for name, story in stories_encoded.items()}
    scores_lead = compute_rouge_score(hyps_lead, refs, "analysis/scores/rouge_scores_{}_lead.bin".format(datset))

    # rouge score without penalty
    mode = "wo_penalty"
    hyps_wo_pen = load_sequences(args.summaries_folder_pref, args.chunks, mode)
    scores_wo_pen = compute_rouge_score(hyps_wo_pen, refs, "analysis/scores/rouge_scores_{}_wo_pen.bin".format(datset))

    # rouge score with penalty
    mode = "w_penalty"
    hyps_w_pen = load_sequences(args.summaries_folder_pref, args.chunks, mode)
    scores_w_pen = compute_rouge_score(hyps_w_pen, refs, "analysis/scores/rouge_scores_{}_w_pen.bin".format(datset))

    for scores, setting in zip([scores_lead, scores_wo_pen, scores_w_pen], ["lead", "without penalty", "with penalty"]):
        avg_rouge_1 = sum([rouge_1 for name, (rouge_1, _, _) in scores.items()])/len(scores)
        avg_rouge_2 = sum([rouge_2 for name, (_, rouge_2, _) in scores.items()])/len(scores)
        avg_rouge_l = sum([rouge_l for name, (_, _, rouge_l) in scores.items()])/len(scores)
        print("Average ROUGE-1 score on {} set: {} \n"
              "Average ROUGE-2 score on {} set: {} \n"
              "Average ROUGE-L score on {} set: {} \n"
              .format(datset, avg_rouge_1, datset, avg_rouge_2, datset, avg_rouge_l))