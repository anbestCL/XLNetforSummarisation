#!/usr/bin/python3
# coding=utf-8

import os
import re
import argparse
import pickle
import random

from pytorch_transformers import XLNetTokenizer

from utils_summarisation import CNNDailyMailDataset


def train_dev_test_split(dataset):
    """Splits data into train, dev, test split for CNN and DailyMail articles according to standard splits
    by Herman et al. (2015).
    for CNN: (90.154/1.218/1.093)
    for DailyMail: (196.961/12.148/10.397)

    Split is documented for future purposes.

    Args:
        dataset: instance of CNNDailyMailDataset class, property self.stories_path lists all article paths of data

    Returns:
        train_paths (list): lists all paths to articles belonging to train set (not used for this thesis)
        dev_paths (list): lists all paths to articles belonging to development set
        test_paths (list): lists all paths to articles belonging to test set

    """
    cnn_paths = []
    dailymail_paths = []
    for path in dataset.stories_path:
        if "cnn" in path:
            cnn_paths.append(path)
        elif "dailymail" in path:
            dailymail_paths.append(path)
    assert len(cnn_paths) + len(dailymail_paths) == len(dataset.stories_path)
    random.shuffle(cnn_paths)
    random.shuffle(dailymail_paths)
    train_cnn = cnn_paths[:90266]
    dev_cnn = cnn_paths[90266:90266 + 1220]
    test_cnn = cnn_paths[90266 + 1220:90266 + 1220 + 1093]
    train_dm = dailymail_paths[:196961]
    dev_dm = dailymail_paths[196961:196961 + 12148]
    test_dm = dailymail_paths[196961 + 12148:196961 + 12148 + 10397]
    with open("data/split_cnn.p", "wb") as cnn_log:
        pickle.dump(train_cnn, cnn_log)
        pickle.dump(dev_cnn, cnn_log)
        pickle.dump(test_cnn, cnn_log)
    with open("data/split_dailymail.p", "wb") as dm_log:
        pickle.dump(train_dm, dm_log)
        pickle.dump(dev_dm, dm_log)
        pickle.dump(test_dm, dm_log)

    train_paths = train_cnn + train_dm
    random.shuffle(train_paths)
    dev_paths = dev_cnn + dev_dm
    random.shuffle(dev_paths)
    test_paths = test_cnn + test_dm
    random.shuffle(test_paths)
    return train_paths, dev_paths, test_paths


def make_chunks(paths, chunk_size, prefix=""):
    """Chunks data into chunk_size many chunks of smaller size for parallel evaluation.

    Args:
        paths (list): lists paths to articles of specific data split
        chunk_size (int): number of articles in one chunk
        prefix (str): indicates data split, 'dev' or 'test'

    """
    for i, chunk in enumerate(range(0, len(paths), chunk_size)):
        path = "data/{}_data/0-9/{}_data_{}".format(prefix, i)
        os.makedirs(path, exist_ok=True)
        article_paths = paths[chunk:chunk+chunk_size]
        for article_path in article_paths:
            with open(article_path, encoding="utf-8") as source:
                raw_story = source.read()
                print(article_path)
                article_path_part = re.match(r"original_data/(.*)", article_path).group(1)
                new_path = os.path.join(path, article_path_part)
                os.makedirs(os.path.dirname(new_path), exist_ok=True)
                with open(new_path, "w") as destination:
                    destination.write(raw_story)


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, help="Data directory whose data is supposed to be split")
parser.add_argument("--chunks", type=int, help="Number of smaller data packages to be produced")
args = parser.parse_args()

tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
dataset = CNNDailyMailDataset(tokenizer=tokenizer, data_dir=args.data_dir)
_, dev_paths, test_paths = train_dev_test_split(dataset)

chunk_size_dev = round(len(dev_paths)/args.chunks)
chunk_size_test = round(len(test_paths)/args.chunks)

make_chunks(dev_paths, chunk_size_dev, "dev")
make_chunks(test_paths, chunk_size_test, "test")


