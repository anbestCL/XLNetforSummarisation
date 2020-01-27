from collections import deque
import os
import re

import torch
from torch.utils.data import Dataset
from pytorch_transformers import XLNetTokenizer

# ------------
# Data loading
# Functions inspired by
# https://github.com/huggingface/transformers/blob/master/examples/summarization/utils_summarization.py
# ------------

class CNNDailyMailDataset(Dataset):
    """ Caches the paths to stories located in specified folder for future preprocessing
    CNN/Daily News:
    The CNN/Daily News raw datasets are downloaded from [1]. The stories are
    stored in separate folders for CNN and DailyMail stories;
    the summary appears at the end of the story as sentences that are prefixed b
    y the special `@highlight` line. The formatting code was inspired by [2].
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
            if os.path.exists(os.path.join(data_dir, dataset)):
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
        story_dir = re.match(r".*data.*\/(.*)\/stories", story_path).group(1)
        story_name = story_dir + "_" + story_name
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

def build_target_mapping(seq_len, predict_pos=0):
    """Mask to indicate which tokens to predict

        For eval mode: one token to predict in each loop at position to be predicted
        Resulting shape (1, 1, seq_len)

        Args:
            seq_len (int): maximal sequence length
            predict_pos (int): position to be predicted

        Returns:
            target_mapping: tensor of shape (1, 1, seq_len) indicating position to be predicted
    """
    if predict_pos is not None:
        target_mapping = torch.zeros((1, 1, seq_len), dtype=torch.float)
        target_mapping[:, :, predict_pos] = 1
    return target_mapping


def build_perm_mask(sum_len, seq_len):
    """Mask that sets permutation order to be used

    First token of summary can only attend to article, second token of summary can attend to first token of
    summary and whole article, and so on
    If prompt is not None, the mask needs to be shifted to the right, since model should attend to prompt

    Args:
        sum_len (int): length of generated summary
        seq_len (int): maximal sequence length

    Returns:
        perm_mask: tensor of shape (1, seq_len, seq_len)

    """
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
    """Mask to avoid performing attention on padded tokens

    1 for real tokens, 0 for padded

    Args:
        seq_len (int): actual length of input (masked summary + story + <sep> + <cls>
        max_seq_len (int): maximal sequence length (hyperparameter)

    Returns:
        mask: attention_mask of shape (1, max_seq_len)
    """
    mask = torch.ones(max_seq_len)
    if max_seq_len - seq_len > 0:
        mask[seq_len:] = 0
    mask = mask.unsqueeze(0)
    return mask


def pad_summaries_ids(mode, summaries_ids, max_seqlen, sum_len):
    """Pads summaries for batching

    In train mode, the summaries are padded to length of longest summary in batch
    In eval mode, the summaries are truncated or padded to generated summary length specified by user

    Args:
        mode (str): "train" or "eval" mode
        summaries_ids: list containing summaries
        max_seqlen: maximal summary length (needed for padding)
        sum_len: length of generated summary (needed for eval mode)

    Returns:
        summaries_ids: tensor containing all summaries
    """
    if mode == "train":
        summaries_ids = torch.cat([pad_summary(summary, max_seqlen) for summary in summaries_ids], dim=0)

    if mode == "eval":
        summaries = torch.zeros((len(summaries_ids), sum_len))
        for i, summary in enumerate(summaries_ids):
            if summary.shape[1] > sum_len:
                summary = summary[:, :sum_len]
            else:
                pad_len = sum_len - summary.shape[1]
                pad = torch.tensor([-1.0] * pad_len, dtype=torch.float64).unsqueeze(0)
                summary = torch.cat([summary, pad], dim=1)
            summaries[i] = summary
        summaries_ids = summaries
    return summaries_ids


def pad_summary(tokens, pad_seqlen):
    """Pads summary to pad_seqlen"""
    if pad_seqlen > tokens.shape[1]:
        padding = torch.tensor([-1] *(pad_seqlen-tokens.shape[1])).unsqueeze(0)
        tokens = torch.cat([tokens, padding], dim=1)
    else:
        tokens = tokens[:pad_seqlen]
    return tokens


def encode_for_summarization(mode, story_lines, summary_lines, tokenizer, max_seqlen, sum_len=None):
    """Create input sequence by encoding summary and story

    Args:
        mode (str): "train" or "eval" mode
        story_lines (list): contains lines from story
        summary_lines (list): contains lines from summary
        tokenizer: instance of XLNetTokenizer loaded from XLNet-Base model, needed to encode sequences
        max_seqlen (int): sequence length (hyperparameter) for which padding is performed
        sum_len (int): generated summary length (hyperparameter)

    Returns:
        input_ids: tensor of shape (1, max_seqlen) representing input sequence of one article
        summaries_ids: tensor of shape (1, max_sumlen) or (1, sum_len) for train and eval case, respectively
         representing summary of one article
        input_len (int): length of input_ids before padding
        sum_len (int): actual length of summary (only for train mode)

    """
    story = " ".join(story_lines)
    summary = " ".join(summary_lines)

    assert tokenizer.mask_token == "<mask>"

    # mode : EVAL
    if mode == "eval":
        summary_mask = " ".join(["<mask>"] * sum_len)
        summary_mask = tokenizer.encode(summary_mask, add_special_tokens=False)
        story_token_ids = tokenizer.encode(story, add_special_tokens=False)
        input_ids = summary_mask + story_token_ids + tokenizer.encode("<sep> <cls>")
        summary_token_ids = tokenizer.encode(summary, add_special_tokens=False)


    #mode : TRAIN
    else:
        summary_token_ids = tokenizer.encode(summary, add_special_tokens=False)
        sum_len = len(summary_token_ids)
        story_token_ids = tokenizer.encode(story, add_special_tokens=False)
        input_ids = summary_token_ids + story_token_ids + tokenizer.encode("<sep> <cls>")

    def _pad_input(tokens, max_seqlen, tokenizer):
        if max_seqlen > len(tokens):
            pad_token_id = tokenizer.encode(tokenizer.pad_token)
            padding = pad_token_id * (max_seqlen-len(tokens))
            tokens += padding
        else:
            tokens = tokens[:max_seqlen-2]
            tokens += tokenizer.encode("<sep> <cls>")
        return tokens

    input_len = len(input_ids)
    input_ids = torch.tensor(_pad_input(input_ids, max_seqlen, tokenizer)).unsqueeze(0)
    summary_token_ids = torch.tensor(summary_token_ids, dtype=torch.float64).unsqueeze(0)
    return input_ids, summary_token_ids, input_len, sum_len


if __name__ == "__main__":
    tokenizer = XLNetTokenizer.from_pretrained('xlnet-large-cased')
    test = CNNDailyMailDataset(tokenizer, data_dir="testing")
    test_path = test.stories_path[0]
    print(encode_for_summarization(tokenizer, test[0][1], test[0][2], test[0][0], tokenizer, max_seqlen = 2500, sum_len = 10))
