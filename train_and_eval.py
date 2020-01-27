#!/usr/bin/python3
# coding=utf-8

from rouge import Rouge
import argparse
import functools
import logging
import os
import random
import sys
import pickle
import time

from sklearn.model_selection import train_test_split
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from pytorch_transformers import XLNetTokenizer, XLNetLMHeadModel, AdamW

from train_and_eval_utils import CNNDailyMailDataset, set_max_seqlen, encode_for_summarization, build_attention_mask, build_perm_mask, build_target_mapping, pad_summary, pad_summaries_ids, pad_target_mapping

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def set_seed(args):
    """Function set seeds for replication purposes

    Args:
        args: dictionary containing seed passed to by user

    """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


# ------------
# Load dataset
# ------------


def load_and_cache_examples(data_dir, tokenizer):
    """Loads paths to data to instance of CNNDailyMailDataset

    Args:
        data_dir (str): path to directory containing data
        tokenizer: instance of XLNetTokenizer loaded from XLNet-Base model

    Returns:
        dataset: instance of CNNDailyMailDataset, holding paths to every story in set in property stories_path
    """
    dataset = CNNDailyMailDataset(tokenizer, data_dir=data_dir)
    return dataset


def train_dev_test_split(dataset):
    """ Splits the list of paths into paths for training, development and test

    Args:
        dataset: instance of CNNDailyMailDataset, dataset.stories_path contains all paths to stories

    Returns:
        train_stories (list): paths to train stories
        dev_stories (list): paths to development stories
        test_stories (list): paths to test stories
    """
    train_dev_stories, test_stories = train_test_split(dataset.stories_path, train_size=0.8)
    train_stories, dev_stories = train_test_split(train_dev_stories, train_size=0.8)

    return train_stories, dev_stories, test_stories


def collate(data, tokenizer, mode, args):
    """Function used to transform a batch to model input

    Transforms story and summary to input_ids, generates permutation and attention masks and target mapping.

    Args:
        data (list): contains tuples (story name, story and summary), length of list=batch_size
        tokenizer: instance of XLNetTokenizer loaded from XLNet-Base model, needed to encode sequences
        args: ditionary containing arguments passed by user, including the maximal sequence length (max_seqlen) and
        length of generated summary (sum_len)

    Returns:
        input_ids: tensor containing input sequence (args.batch_size x args.max_seqlen)
        attention_masks: tensor containing attention mask (args.batch_size x args.max_seqlen)
        perm_masks: tensor containing permutation mask (args.batch_size x args.max_seqlen x args.max_seqlen)
        target_mappings: tensor containing target mapping (args.batch_size x 1 x args.max_seqlen)
        storynames: list with story names of batch
    """

    # remove the files with empty an story/summary
    data = filter(lambda x: not (len(x[1]) == 0 or len(x[2]) == 0), data)
    story_names, stories, summaries = zip(*list(data))

    if mode == "eval":
        # create input_ids of shape (batch_size, seq_len)
        input_ids, input_lens, summaries_ids, sum_lens = \
            zip(*list([encode_for_summarization(mode, story_lines=story, tokenizer=tokenizer, max_seqlen=args.max_seqlen,
                                                sum_len=args.sum_len, prompt=args.prompt)
                       for (_, story, _) in zip(story_names, stories, summaries)]))
        pad_sum_len = max(sum_lens)
        summaries_ids = pad_summaries_ids(mode, summaries_ids, pad_sum_len, args.max_seqlen, args.sum_len)

        # create perm_masks of shape (batch_size, seq_len, seq_len)
        perm_masks = \
            torch.cat([build_perm_mask(sum_len=args.sum_len, seq_len=args.max_seqlen, prompt=args.prompt)
                       for _ in input_ids], dim=0)

        # create target_mappings (batch_size, num_predict, seq_len) for first position to be predicted
        # num_predict=1 for evaluation (one token predicted at a time)
        target_mappings = torch.cat([build_target_mapping(args.max_seqlen, prompt=args.prompt) for _ in input_ids],
                                    dim=0)

        # create attention_masks (batch_size, seq_len)
        attention_masks = torch.cat([build_attention_mask(input_len, args.max_seqlen)
                                     for input_len in input_lens], dim=0)

        input_ids = torch.cat(input_ids, dim=0)

        return (
            input_ids,
            summaries_ids,
            attention_masks,
            perm_masks,
            target_mappings,
            story_names
        )
    elif mode == "train":
        input_ids, input_lens, summaries_ids, sum_lens = zip(*list([encode_for_summarization(mode, story, summary, story_name, tokenizer, args.max_seqlen, args.sum_len) for (story_name, story, summary) in zip(story_names, stories, summaries)]))
        pad_sum_len = max(sum_lens)
        perm_masks = torch.cat([build_perm_mask(sum_len, args.max_seqlen) for sum_len in sum_lens], dim=0)
        summaries_ids = pad_summaries_ids(mode, summaries_ids, pad_sum_len, args.max_seqlen, args.sum_len)

        attention_masks = torch.cat([build_attention_mask(input_len, args.max_seqlen)
        for input_len in input_lens], dim=0)

        input_ids = torch.cat(input_ids, dim=0)

        return (
            input_ids,
            summaries_ids,
            attention_masks,
            perm_masks
        )


# ------------
# Train
# ------------


def train(args, model, tokenizer, train_paths, dev_paths):
    """Loads data in batches, performs training for each batch, calls evaluation loop after
     every epoch and saves generated summaries

    For a batch, training is performed for number of epochs in args.num_epochs

    Args:
        args: ditionary containing arguments passed by user, including the path to the data folder (args. data_dir),
        batch size (args.batch_size), the device (args.device) and more
        model: instance of XLNetLMHeadModel, loaded from pretrained XLNet-Base
        tokenizer: instance of XLNetTokenizer loaded from XLNet-Base model, needed to encode sequences
        train_paths: list of path to stories selected for training
        dev_paths: list of path to stories selected for evaluation

    Returns:
        global_loss (list): tracking loss of training routine
        dev_scores (list): tracking ROUGE-scores on development set
        best_dev_score (float): best ROUGE-1 score achieved on dev set
    """
    set_seed(args)

    # Load the data
    train_dataset = load_and_cache_examples(tokenizer=tokenizer, data_dir=args.data_dir)
    train_dataset.stories_path = train_paths

    train_sampler = RandomSampler(train_dataset)
    model_collate_fn = functools.partial(collate, tokenizer=tokenizer, mode="train", args=args)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.batch_size,
        collate_fn=model_collate_fn,
        num_workers=args.num_workers
    )

    optimizer = AdamW(model.parameters())

    # Train
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num epochs = %d", args.num_epochs)
    logger.info(
        "  Batch size = %d", args.batch_size
    )

    model.zero_grad()
    model = model.train().to(args.device)

    global_loss = [] #keep track of train loss
    dev_scores = [] #keep track of dev scores

    def train_fn(loader):
        tr_loss = 0.0
        for step, batch in enumerate(loader):
            input_ids, summaries, attention_mask, perm_mask = batch

            input_ids = input_ids.to(args.device)
            perm_mask = perm_mask.to(args.device)
            attention_mask = attention_mask.to(args.device)
            summaries = summaries.to(args.device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                perm_mask=perm_mask,
                labels=summaries
            )

            loss = outputs[0]
            logger.info("Case {}: Loss = {}".format(args.case, loss))
            tr_loss += loss
            loss.backward()
            optimizer.step()

        tr_loss /= step
        global_loss.append(tr_loss)

    best_dev_score = 0.0 # rouge-1 f-score
    for epoch in range(args.num_epochs):
        train_fn(train_dataloader)

        curr_rouge_score = evaluate(args, model, tokenizer, dev_paths)

        curr_dev_score = curr_rouge_score['rouge-1']['f']
        dev_scores.append(curr_dev_score)
        if curr_dev_score > best_dev_score:
            best_dev_score = curr_dev_score
            logger.info("Saving model checkpoint to %s", args.output_dir)

            # Save a trained model, configuration and tokenizer using `save_pretrained()`.
            # They can then be reloaded using `from_pretrained()`
            model.save_pretrained(args.output_dir)
            torch.save(args, os.path.join(args.output_dir, "training_arguments.bin"))
    return global_loss, dev_scores, best_dev_score


# ------------
# Evaluate
# ------------

def compute_rouge_score(tokenizer, predicted_sequences, references, article_names):
    """Computes ROUGE-scores between generated  and reference summaries for a batch

    Args:
        tokenizer: instance of XLNetTokenizer loaded from XLNet-Base model
        predicted_sequences: tensor of generated summaries
        references: tensor of reference summaries
        article_names: list of article names, used to write generated summary to file

    Returns:
        scores (dict): containing ROUGE scores averaged over batch

    """
    r = Rouge()

    def _decode_seq(seq):
        sent = tokenizer.decode(seq, clean_up_tokenization_spaces=True, skip_special_tokens=True)
        if sent.startswith("."):
            sent = sent[1:]
        return sent

    hyps = []
    refs = []
    if not isinstance(references, list):
        references = references.tolist()
    for (hyp, ref) in zip(predicted_sequences.tolist(), references):
        hyps.append(_decode_seq(hyp))
        ref = [int(val) for val in ref]
        refs.append(_decode_seq(ref))

    scores = r.get_scores(hyps, refs, avg=True) #avg=True to get mean values
    for i, article in enumerate(article_names):
        with open("summaries/{}_generated.txt".format(article), "w") as summary_file:
            summary_file.write(" ".join(hyps[i]))
    return scores


def evaluate(args, model, tokenizer, dev_paths):
    """Loads data in batches, performs evaluation for each batch and saves generated summaries

    For a batch, evaluation is performed by looping over the provided summary length and generating one token at a time.
    The generated token replaces the <mask> token in the input_ids and then prediction for next token is performed.

    Args:
        args: ditionary containing arguments passed by user, including the path to the data folder (args. data_dir),
        batch size (args.batch_size), the device (args.device) and more
        model: instance of XLNetLMHeadModel, loaded from pretrained XLNet-Base
        tokenizer: instance of XLNetTokenizer loaded from XLNet-Base model, needed to encode sequences
        dev_paths (list): contains path to stories used for evaluation

    Returns:
        rouge_dict (dict): containing averaged ROUGE-1, ROUGE-2 and ROUGE-L F-scores
    """
    set_seed(args)

    eval_dataset = load_and_cache_examples(tokenizer=tokenizer, data_dir=args.data_dir)
    eval_dataset.stories_path = dev_paths

    eval_sampler = SequentialSampler(eval_dataset)
    model_collate_fn = functools.partial(collate, tokenizer=tokenizer, mode="eval", args=args)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=args.batch_size,
        collate_fn=model_collate_fn,
        num_workers=args.num_workers
    )

    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.batch_size)

    if args.lead:
        rouge_score_article = []
    rouge_score = []

    model.eval().to(args.device)

    for step, batch in enumerate(eval_dataloader):
        tic = time.perf_counter()
        print("Batch {}".format(step))
        input_ids, summaries_ids, attention_mask, perm_mask, target_mapping, article_names = batch

        # To keep track of the generated sequence
        predicted_sequence = torch.zeros((input_ids.shape[0], args.sum_len), dtype=torch.int32)

        input_ids = input_ids.to(args.device)
        perm_mask = perm_mask.to(args.device)
        attention_mask = attention_mask.to(args.device)

        for predict_pos in range(args.sum_len):
            print("Predicting position {}".format(predict_pos))
            if predict_pos > 0:
                # for each position a new target_mapping has to be created
                target_mapping = torch.cat([build_target_mapping(seq_len=input_ids.shape[1], predict_pos=predict_pos,
                                                                 prompt=args.prompt)
                                            for _ in range(input_ids.shape[0])], dim=0)

            target_mapping = target_mapping.to(args.device)

            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    perm_mask=perm_mask,
                    target_mapping=target_mapping,
                )

            # Output has shape [batch_size, num_predict, config.vocab_size],
            # num_predict is number of tokens to be predicted, for evaluation: num_predict=1
            next_token_logits = outputs[0]
            # slightly modified multiplicative repetition penalty from CTRL paper (https://arxiv.org/abs/1909.05858)
            if args.repetition_penalty != 1.0:
                for i in range(input_ids.shape[0]):
                    # loop through previously generated tokens
                    # generated tokens replace <mask> token in input
                    if args.prompt:
                        generated_tokens = set(input_ids[i].tolist()[args.prompt:args.prompt + predict_pos])
                    else:
                        generated_tokens = set(input_ids[i].tolist()[:predict_pos])
                    for previous_tokens in generated_tokens:
                        # if score < 0,
                        # then repetition penalty has to be multiplied to reduce the previous token probability
                        if next_token_logits[i, 0, previous_tokens] < 0:
                            next_token_logits[i, 0, previous_tokens] *= args.repetition_penalty
                        else:
                            next_token_logits[i, 0, previous_tokens] /= args.repetition_penalty

            # alternative additive penalty
            # if args.repetition_penalty != 0.0:
            # for i in range(input_ids.shape[0]):
            # for previous_tokens in set(input_ids[i].tolist()[:predict_pos]):
            # next_token_logits[i, 0, previous_tokens] -= args.repetition_penalty

            # choosing candidate token (no beam search)
            _, predicted_indices = torch.max(next_token_logits.view(input_ids.shape[0], -1), dim=1, keepdim=True)

            for i in range(predicted_indices.shape[0]):
                # keep track of prediction
                predicted_sequence[i, predict_pos] = int(predicted_indices[i].item())

                # replace prediction in input
                if args.prompt:
                    input_ids[i, args.prompt + predict_pos] = predicted_indices[i].item()
                else:
                    input_ids[i, predict_pos] = predicted_indices[i].item()

        # rouge score per predicted sentence
        sent_score = compute_rouge_score(tokenizer, predicted_sequence, summaries, article_names)
        rouge_score.append(sent_score)

    def _average_rouge_score(score):
        """Compute overall scores as average over batch-level scores"""
        rouge_dict = {rouge_type: {key: 0 for key in ['f', 'p', 'r']} for rouge_type in ['rouge-1', 'rouge-2', 'rouge-l']}
        for batch in score:
            for rouge_type, values in batch.items():
                for measure, value in values.items():
                    rouge_dict[rouge_type][measure] += value
        rouge_dict = {rouge_type: {measure: value / len(score) for measure, value in values.items()} for
                      rouge_type, values in rouge_dict.items()}
        return rouge_dict

    rouge_dict = _average_rouge_score(rouge_score)

    # Save the evaluation's results
    output_eval_file = os.path.join(args.output_dir, "eval_results.txt")

    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results *****")
        logger.info("***** Rouge scores with summary *****")
        for key in sorted(rouge_dict.keys()):
            logger.info("  %s = %s", key, str(rouge_dict[key]))
            writer.write("%s = %s\n" % (key, str(rouge_dict[key])))
    return rouge_dict


def main():
    """ Main function calling train function or eval function depending on parameters given by user"""
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input training data file (a text file).",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # Optional parameters
    parser.add_argument(
        "--do_evaluate",
        type=bool,
        default=False,
        help="Run model evaluation on out-of-sample data.",
    )
    parser.add_argument("--do_train", type=bool, default=False, help="Run training.")
    parser.add_argument(
        "--model_name_or_path",
        default="xlnet-base-cased",
        type=str,
        help="The model checkpoint to initialize the encoder and decoder's weights with.",
    )
    parser.add_argument(
        "--num_layers",
        default=3,
        type=int,
        help="Number of layers of model to use",
    )
    parser.add_argument(
        "--num_epochs",
        default=1,
        type=int,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--batch_size",
        default=1,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--num_workers",
        default=4,
        type=int,
        help="Number of workers for data loading",
    )
    parser.add_argument(
        "--max_seqlen",
        type=int,
        help="Maximal sequence length, longer sentences are truncated",
    )
    parser.add_argument(
        "--sum_len",
        type=int,
        help="Only for eval mode: length of summary to be generated",
    )
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.0,
        help="The parameter for repetition penalty. Between 1.0 and + infinity. 1.0 means no penalty. Default to 1.",
    )
    parser.add_argument("--is_cpu", type=bool, help="Set training to cpu")
    parser.add_argument("--is_cuda", type=bool, help="Set training to gpu")

    args = parser.parse_args()

    # Set up training device
    if args.is_cuda:
        if torch.cuda.is_available():
            args.device = torch.device("cuda")
    else:
        args.device = torch.device("cpu")

    # Load pretrained model and tokenizer
    tokenizer = XLNetTokenizer.from_pretrained(args.model_name_or_path)
    model = XLNetLMHeadModel.from_pretrained(args.model_name_or_path, n_layer=args.num_layers)

    # Make train-test-dev-split
    dataset = load_and_cache_examples(tokenizer=tokenizer, data_dir=args.data_dir)
    train_paths, dev_paths, test_paths = train_dev_test_split(dataset)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.warning(
        "Device: %s, ",
        args.device
    )

    logger.info("Training/evaluation parameters %s", args)

    # Train the model
    if args.do_train:
        logger.info("***** Running training *****")
        tr_losses, dev_scores, best_dev_score = train(args, model, tokenizer, train_paths)
        logger.info(" train loss history = %s, \n, dev score history = %s, \n best Rouge F1 score on dev set = %s",
                    tr_losses, dev_scores, best_dev_score)
        metrics_file = os.path.join(args.output_dir, "metrics.bin")
        with open(metrics_file, "wb") as metrics:
            pickle.dump(tr_losses, metrics)
            pickle.dump(dev_scores, metrics)
            pickle.dump(best_dev_score, metrics)

    # Evaluate the model
    if args.do_evaluate:
        # test check
        logger.info("***** Running testing *****")
        evaluate(args, model, tokenizer, test_paths)


if __name__ == "__main__":
    main()