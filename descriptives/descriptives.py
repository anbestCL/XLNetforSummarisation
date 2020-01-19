from pytorch_transformers import XLNetTokenizer

import pickle
import argparse
import re

from utils_summarisation import CNNDailyMailDataset


def load_paths(filename):
    with open(filename, "rb") as log:
        train_paths = pickle.load(log)
        dev_paths = pickle.load(log)
        test_paths = pickle.load(log)
    return train_paths, dev_paths, test_paths


def get_statistics(dataset, paths, tokenizer, prefix):
    dataset.stories_path = paths

    def _get_seq_lengths(lengths, lines, name):
        text = tokenizer.tokenize(" ".join(lines))
        lengths[name] = len(text)

    story_lengths = {}
    summary_lengths = {}
    for (story_name, story_lines, summary_lines) in dataset:
        _get_seq_lengths(story_lengths, story_lines, story_name)
        _get_seq_lengths(summary_lengths, summary_lines, story_name)
    with open("descriptives_{}.p".format(prefix), "wb") as file:
        pickle.dump(story_lengths, file)
        pickle.dump(summary_lengths, file)


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, help="Directory to load data from")
args = parser.parse_args()

tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
dataset = CNNDailyMailDataset(tokenizer=tokenizer, data_dir=args.data_dir)

for i in ("split_cnn.p", "split_dailymail.p"):
    paths = load_paths(i)
    cat = re.match("split_(.*).p", i).group(1)
    for j, pref in enumerate(["{}_train".format(cat), "{}_dev".format(cat), "{}_test".format(cat)]):
        get_statistics(dataset, paths[j], tokenizer, pref)




