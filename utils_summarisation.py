from collections import deque
import os
import re

import torch
from torch.utils.data import Dataset
from pytorch_transformers import XLNetTokenizer

# ------------
# Data loading
# ------------


class CNNDailyMailDataset(Dataset):
    """ Abstracts the dataset used to train seq2seq models.
    CNN/Daily News:
    The CNN/Daily News raw datasets are downloaded from [1]. The stories are
    stored in different files; the summary appears at the end of the story as
    sentences that are prefixed by the special `@highlight` line. To process
    the data, pass the path to this
    folder as the "data_dir argument. The formatting code was inspired by [2].
    [1] https://cs.nyu.edu/~kcho/
    [2] https://github.com/abisee/cnn-dailymail/
    """

    def __init__(self, tokenizer, prefix="train", data_dir=""):
        assert os.path.isdir(data_dir)
        self.tokenizer = tokenizer

        # We initialize the class by listing all the files that contain
        # stories and summaries. Files are not read in memory given
        # the size of the corpus.
        self.stories_path = []
        datasets = ("dailymail", "cnn")
        for dataset in datasets:
            path_to_stories = os.path.join(data_dir, dataset, "stories")
            story_filenames_list = os.listdir(path_to_stories)
            for story_filename in story_filenames_list:
                path_to_story = os.path.join(path_to_stories, story_filename)
                if not os.path.isfile(path_to_story):
                    continue
                self.stories_path.append(path_to_story)

    def __len__(self):
        return len(self.stories_path)

    def __getitem__(self, idx):
        story_path = self.stories_path[idx]
        story_name = re.match(r"(^.*\/)(.*)(\..*$)", story_path).group(2)
        with open(story_path, encoding="utf-8") as source:
            raw_story = source.read()
            story_lines, summary_lines = process_story(raw_story)
        return story_name, story_lines, summary_lines


def process_story(raw_story):
    """ Extract the story and summary from a story file.
    Attributes:
        raw_story (str): content of the story file as an utf-8 encoded string.
    Raises:
        IndexError: If the story is empty or contains no highlights.
    """
    nonempty_lines = list(
        filter(lambda x: len(x) != 0, [line.strip() for line in raw_story.split("\n")])
    )

    # for some unknown reason some lines miss a period, add it
    nonempty_lines = [_add_missing_period(line) for line in nonempty_lines]

    # gather article lines
    story_lines = []
    lines = deque(nonempty_lines)
    while True:
        try:
            element = lines.popleft()
            if element.startswith("@highlight"):
                break
            story_lines.append(element)
        except IndexError:
            # if "@highlight" is absent from the file we pop
            # all elements until there is None.
            return story_lines, []

    # gather summary lines
    summary_lines = list(filter(lambda t: not t.startswith("@highlight"), lines))

    return story_lines, summary_lines


def _add_missing_period(line):
    END_TOKENS = [".", "!", "?", "...", "'", "`", '"', u"\u2019", u"\u2019", ")"]
    if line.startswith("@highlight"):
        return line
    if line[-1] in END_TOKENS:
        return line
    return line + "."

# --------------------------
# Encoding and preprocessing
# --------------------------
def set_max_seqlen(stories, summaries, tokenizer):
    stories = [tokenizer.encode(" ".join(story) +"<sep>" + " ".join(summary), add_special_tokens=True) for story, summary in zip(stories, summaries)]
    max_seq_len = len(max(stories, key=len))
    return max_seq_len


def build_target_mapping(seq_len, pad_sum_len=None, predict_pos=None, sum_len=None):
    """ Mask to indicate which tokens to predict
    For train mode: for each token of summary to be predicted cannot attend to token's position
    in sequence (diagonal matrix)
    For eval mode: one token to predict in each loop at position to be predicted"""
    if predict_pos is None:
        target_mapping = torch.zeros((1, pad_sum_len, seq_len), dtype=torch.float)
        for i in range(sum_len):
            for j in range(seq_len):
                if j <= sum_len:
                    if i == j:
                        target_mapping[0, i, j] = 1
        assert target_mapping[-1, 1, sum_len + 1] == 0

    if predict_pos is not None:
        target_mapping = torch.zeros((1, 1, seq_len), dtype=torch.float)
        target_mapping[:, :, predict_pos] = 1
    return target_mapping


def build_perm_mask(sum_len, seq_len):
    """ Mask that sets permutation order to be used
    First token of summary can only attend to article,
    second token of summary can attend to first token of summary and whole article,
    and so on"""
    perm_mask = torch.zeros(1, seq_len, seq_len)
    for i in range(seq_len):
        for j in range(seq_len):
            if i <= sum_len and j <= sum_len:
                if i == j:
                    perm_mask[0, i, j] = 1
                elif i < j:
                    perm_mask[0, i, j] = 1
    assert perm_mask[-1, sum_len + 1, sum_len + 1] == 0
    return perm_mask


def build_attention_mask(seq_len, max_seq_len):
    """ Mask to avoid performing attention on padded tokens
    1 for real tokens, 0 for padded
    """
    mask = torch.ones(max_seq_len)
    if max_seq_len - seq_len > 0:
        mask[seq_len:] = 0
    mask = mask.unsqueeze(0)
    return mask


def pad_target_mapping(sum_lens, case, pad_sum_len, max_seqlen, predict_pos, sum_len):
    if case == 1:
        target_mappings = torch.cat(
            [build_target_mapping(max_seqlen, pad_sum_len=pad_sum_len, sum_len=sum_len) for
             sum_len in sum_lens], dim=0)
    if case == 2:
        target_mappings = torch.zeros((len(sum_lens)))

    if sum_len is not None:
        target_mappings = torch.cat(
            [build_target_mapping(max_seqlen, predict_pos=predict_pos) for
             _ in sum_lens], dim=0)

    return target_mappings


def pad_summaries_ids(summaries_ids, case, pad_sum_len, max_seqlen, sum_len):
    if case == 1:
        summaries_ids = torch.cat([pad_summary(summary, pad_sum_len) for summary in summaries_ids], dim=0)
    if case == 2:
        summaries_ids = torch.cat([pad_summary(summary, max_seqlen) for summary in summaries_ids], dim=0)

    if sum_len is not None:
        summaries = torch.zeros((len(summaries_ids), sum_len))
        for i, summary in enumerate(summaries_ids):
            if summary.shape[1] > sum_len:
                summary = summary[:, :sum_len]
            else:
                pad_len = sum_len - summary.shape[1]
                pad = torch.tensor([-1] * pad_len).unsqueeze(0)
                summary = torch.cat([summary, pad], dim=1)
            summaries[i] = summary
        summaries_ids = summaries
    return summaries_ids

def pad_summary(tokens, pad_seqlen):
    if pad_seqlen > tokens.shape[1]:
        padding = torch.tensor([-1] *(pad_seqlen-tokens.shape[1])).unsqueeze(0)
        tokens = torch.cat([tokens, padding], dim=1)
    else:
        tokens = tokens[:pad_seqlen]
    return tokens

def encode_for_summarization(story_lines, summary_lines, story_name, tokenizer, max_seqlen, sum_len = None):
    """ Encode the story and summary lines, and join them
    as specified in [1] by using `[SEP] [CLS]` tokens to separate
    sentences.
    """
    story = " ".join(story_lines)
    with open("stories/{}_story.txt".format(story_name), "w") as story_file:
        story_file.write(story)
    summary = " ".join(summary_lines)
    with open("summaries/{}_summary.txt".format(story_name), "w") as summary_file:
        summary_file.write(story)

    assert tokenizer.mask_token == "<mask>"

    # mode : EVAL
    if sum_len is not None:
        summary_mask = " ".join(["<mask>"] * sum_len)
        summary_mask = tokenizer.encode(summary_mask, add_special_tokens=True)
        story_token_ids = tokenizer.encode("." + story, add_special_tokens=True)
        input_ids = summary_mask[:-1] + story_token_ids
        summary_token_ids = tokenizer.encode(summary, add_special_tokens=True)


    #mode : TRAIN
    else:
        summary_token_ids = tokenizer.encode(summary, add_special_tokens=True)
        sum_len = len(summary_token_ids)
        story_token_ids = tokenizer.encode(story, add_special_tokens=True)
        # summary has CLS token at the end -> remove for training ?
        input_ids = summary_token_ids[:-1] + story_token_ids


    def _pad_input(tokens, max_seqlen, tokenizer):
        if max_seqlen > len(tokens):
            pad_token_id = tokenizer.encode(tokenizer.pad_token)
            padding = pad_token_id *(max_seqlen-len(tokens))
            tokens += padding
        else:
            tokens = tokens[:max_seqlen]
        return tokens

    input_ids = torch.tensor(_pad_input(input_ids, max_seqlen, tokenizer)).unsqueeze(0)
    summary_token_ids = torch.tensor(summary_token_ids).unsqueeze(0)
    story_token_ids = torch.tensor(story_token_ids).unsqueeze(0)
    return input_ids, story_token_ids, summary_token_ids, sum_len


if __name__ == "__main__":
    tokenizer = XLNetTokenizer.from_pretrained('xlnet-large-cased')
    test = CNNDailyMailDataset(tokenizer, data_dir="testing")
    test_path = test.stories_path[0]
    print(encode_for_summarization(test[0][0], test[0][1], test_path, tokenizer, sum_len = 10))