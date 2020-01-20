#!/usr/bin/python3

import pickle
import argparse
import re

from pytorch_transformers import XLNetTokenizer

from utils_summarisation import CNNDailyMailDataset


def load_paths(filename):
    """Reading in the split of paths into train, dev and test paths.

    Args:
        filename (str): path to file containing the split

    Returns:
        train_paths (list): lists articles' paths belonging to train set
        dev_paths (list): lists articles' paths belonging to dev set
        test_paths (list): lists articles' paths belonging to test set

    """
    with open(filename, "rb") as log:
        train_paths = pickle.load(log)
        dev_paths = pickle.load(log)
        test_paths = pickle.load(log)
    return train_paths, dev_paths, test_paths


def get_statistics(dataset, paths, tokenizer, filename):
    """Computes and saves summary and story length of articles in data set.

    Args:
        dataset (class instance): instance of class CNNDailyMailDataset containing paths to stories as property
        paths (list): lists paths to articles of a specific data split
        tokenizer: instance of XLNetTokenizer, used to calculate token lengths of summary and story
        filename: file path for saving dictionaries : key: article, value: summary or story length

    """
    dataset.stories_path = paths

    def _get_seq_lengths(lengths, lines, name):
        text = tokenizer.tokenize(" ".join(lines))
        lengths[name] = len(text)

    story_lengths = {}
    summary_lengths = {}
    for (story_name, story_lines, summary_lines) in dataset:
        _get_seq_lengths(story_lengths, story_lines, story_name)
        _get_seq_lengths(summary_lengths, summary_lines, story_name)
    with open(filename, "wb") as file:
        pickle.dump(story_lengths, file)
        pickle.dump(summary_lengths, file)


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, help="Directory to load data from")
args = parser.parse_args()

tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
dataset = CNNDailyMailDataset(tokenizer=tokenizer, data_dir=args.data_dir)

for i in ("data/split_cnn.p", "data/split_dailymail.p"):
    paths = load_paths(i)
    cat = re.match("split_(.*).p", i).group(1)
    for j, name in enumerate(["descriptives/param_files/descriptives_{}_train".format(cat),
                              "descriptives/param_files/descriptives_{}_dev".format(cat),
                              "descriptives/param_files/descriptives_{}_test".format(cat)]):
        get_statistics(dataset, paths[j], tokenizer, name)




