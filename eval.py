#!/usr/bin/python3
# coding=utf-8

import argparse
import functools
import logging
import os
import random
import sys
import re
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler

from pytorch_transformers import XLNetTokenizer, XLNetLMHeadModel

from eval_utils import CNNDailyMailDataset, \
    encode_for_summarization, build_attention_mask, build_perm_mask, build_target_mapping

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


def collate(data, tokenizer, args):
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

    # create input_ids of shape (batch_size, seq_len)
    input_ids, input_lens = \
        zip(*list([encode_for_summarization(story_lines=story, tokenizer=tokenizer, max_seqlen=args.max_seqlen,
                                            sum_len=args.sum_len, prompt=args.prompt)
                   for (_, story, _) in zip(story_names, stories, summaries)]))

    # create perm_masks of shape (batch_size, seq_len, seq_len)
    perm_masks = \
        torch.cat([build_perm_mask(sum_len=args.sum_len, seq_len=args.max_seqlen, prompt=args.prompt)
                   for _ in input_ids], dim=0)

    # create target_mappings (batch_size, num_predict, seq_len) for first position to be predicted
    # num_predict=1 for evaluation (one token predicted at a time)
    target_mappings = torch.cat([build_target_mapping(args.max_seqlen, prompt=args.prompt) for _ in input_ids], dim=0)

    # create attention_masks (batch_size, seq_len)
    attention_masks = torch.cat([build_attention_mask(input_len, args.max_seqlen)
    for input_len in input_lens], dim=0)

    input_ids = torch.cat(input_ids, dim=0)

    return (
        input_ids,
        attention_masks,
        perm_masks,
        target_mappings,
        story_names
    )


# ------------
# Evaluate
# ------------

def evaluate(args, model, tokenizer):
    """Loads data in batches, performs evaluation for each batch and saves generated summaries

    For a batch, evaluation is performed by looping over the provided summary length and generating one token at a time.
    The generated token replaces the <mask> token in the input_ids and then prediction for next token is performed.

    Args:
        args: ditionary containing arguments passed by user, including the path to the data folder (args. data_dir),
        batch size (args.batch_size), the device (args.device) and more
        model: instance of XLNetLMHeadModel, loaded from pretrained XLNet-Base
        tokenizer: instance of XLNetTokenizer loaded from XLNet-Base model, needed to encode sequences

    """
    set_seed(args)

    eval_dataset = load_and_cache_examples(tokenizer=tokenizer, data_dir=args.data_dir)
    eval_sampler = SequentialSampler(eval_dataset)
    model_collate_fn = functools.partial(collate, tokenizer=tokenizer, args=args)
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

    model.eval().to(args.device)

    for step, batch in enumerate(eval_dataloader):
        tic = time.perf_counter()
        print("Batch {}".format(step))
        input_ids, attention_mask, perm_mask, target_mapping, article_names = batch

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
            #if args.repetition_penalty != 0.0:
                #for i in range(input_ids.shape[0]):
                    #for previous_tokens in set(input_ids[i].tolist()[:predict_pos]):
                        #next_token_logits[i, 0, previous_tokens] -= args.repetition_penalty

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

        # saving predicted sequence to file
        for i, seq in enumerate(predicted_sequence.tolist()):
            if args.prompt:
                pref = input_ids.tolist()[i][:args.prompt]
                pref_decoded = tokenizer.decode(pref, clean_up_tokenization_spaces=True, skip_special_tokens=True)
            seq_decoded = tokenizer.decode(seq, clean_up_tokenization_spaces=True, skip_special_tokens=True)
            logger.info("***** Writing prediction for article {}".format(article_names[i]))
            chunk = re.match(r".*_data_(.*)", args.data_dir).group(1)
            path = os.path.join("test_summaries_{}_wo_penalty".format(chunk), "{}_generated.txt".format(article_names[i]))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as summary_file:
                if args.prompt:
                    summary_file.write(pref_decoded)
                summary_file.write(seq_decoded)
        toc = time.perf_counter()
        print("Prediction of batch {} took {} seconds".format(step, (toc-tic)))


def main():
    """Main routine, loading model, setting up logging and calling evaluate function"""

    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input training data file (a text file).",
    )

    # Optional parameters
    parser.add_argument(
        "--do_evaluate",
        type=bool,
        default=False,
        help="Run model evaluation on out-of-sample data.",
    )
    parser.add_argument(
        "--model_name_or_path",
        default="xlnet-base-cased",
        type=str,
        help="The model checkpoint to initialize the encoder and decoder's weights with.",
    )
    parser.add_argument(
        "--num_layers",
        default=12,
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
        default=0,
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
    parser.add_argument(
        "--prompt",
        type=int,
        help="Number of tokens to use as prompt",
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

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.warning(
        "Device: %s, ",
        args.device
    )

    logger.info("Training/evaluation parameters %s", args)

    # Evaluate the model
    if args.do_evaluate:
        # baseline
        logger.info("***** Running Evaluation *****")
        if args.model_name_or_path == "xlnet-base-cased":
            logger.info("***** Running baseline *****")
        evaluate(args, model, tokenizer)
        logger.info("***** Evaluation finished *****")


if __name__ == "__main__":
    main()
