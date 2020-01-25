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

from utils_summarisation import CNNDailyMailDataset, set_max_seqlen, encode_for_summarization, build_attention_mask, build_perm_mask, build_target_mapping, pad_summary, pad_summaries_ids, pad_target_mapping

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


# ------------
# Load dataset
# ------------


def load_and_cache_examples(data_dir, tokenizer):
    dataset = CNNDailyMailDataset(tokenizer, data_dir=data_dir)
    return dataset

def collate(data, tokenizer, mode, args, sum_len=None, predict_pos=None):
    """ storyname, stories, summaries per article in batch as input """

    # remove the files with empty an story/summary
    data = filter(lambda x: not (len(x[1]) == 0 or len(x[2]) == 0), data)
    story_names, stories, summaries = zip(*list(data))

    # max_seqlen set as parameter for training on GPU
    if args.max_seqlen is None:
        args.max_seqlen = set_max_seqlen(stories, summaries, tokenizer)

    input_ids, stories_ids, summaries_ids, sum_lens = zip(*list([encode_for_summarization(mode, story, summary, tokenizer, args.max_seqlen, sum_len) for (_, story, summary) in zip(story_names, stories, summaries)]))
    perm_masks = torch.cat([build_perm_mask(sum_len, args.max_seqlen) for sum_len in sum_lens], dim=0)
    target_mappings = pad_target_mapping(mode, sum_lens, args.max_seqlen, predict_pos)
    summaries_ids = pad_summaries_ids(mode, summaries_ids, args.max_seqlen, sum_len)

    attention_masks = torch.cat([build_attention_mask(story.shape[1], args.max_seqlen)
    for story in stories_ids], dim=0)

    input_ids = torch.cat(input_ids, dim=0)

    if mode == "eval":
        return (
            input_ids,
            stories_ids,
            summaries_ids,
            attention_masks,
            perm_masks,
            target_mappings,
            story_names
        )
    else:
        return (
            input_ids,
            stories_ids,
            summaries_ids,
            attention_masks,
            perm_masks,
            target_mappings
        )


# ------------
# Evaluate
# ------------

def evaluate(args, model, tokenizer):
    set_seed(args)

    eval_dataset = load_and_cache_examples(tokenizer=tokenizer, data_dir=args.data_dir)

    eval_sampler = SequentialSampler(eval_dataset)
    model_collate_fn = functools.partial(collate, tokenizer=tokenizer, mode="eval", args=args, sum_len=args.sum_len, predict_pos=0)
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
        input_ids, stories, summaries, attention_mask, perm_mask, target_mapping, article_names = batch
        predicted_sequence = torch.zeros((input_ids.shape[0], args.sum_len), dtype=torch.int32)

        input_ids = input_ids.to(args.device)
        perm_mask = perm_mask.to(args.device)
        attention_mask = attention_mask.to(args.device)

        for predict_pos in range(args.sum_len):
            print("Predicting position {}".format(predict_pos))
            if predict_pos > 0:
                target_mapping = torch.cat([build_target_mapping(seq_len=input_ids.shape[1], predict_pos=predict_pos) for _ in range(input_ids.shape[0])], dim=0)

            target_mapping = target_mapping.to(args.device)

            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    perm_mask=perm_mask,
                    target_mapping=target_mapping,
                )

            # Output has shape [target_mapping.size(0), target_mapping.size(1), config.vocab_size]
            next_token_logits = outputs[0]

            if args.repetition_penalty != 0.0:
                for i in range(input_ids.shape[0]):
                    for previous_tokens in set(input_ids[i].tolist()[:predict_pos]):
                        next_token_logits[i, 0, previous_tokens] -= args.repetition_penalty
            _, predicted_indices = torch.max(next_token_logits.view(input_ids.shape[0], -1), dim=1, keepdim=True)

            for i in range(predicted_indices.shape[0]):
                predicted_sequence[i, predict_pos] = int(predicted_indices[i].item())

                # replace prediction in input
                input_ids[i, predict_pos] = predicted_indices[i].item()

        for i, seq in enumerate(predicted_sequence.tolist()):
            seq_decoded = tokenizer.decode(seq, clean_up_tokenization_spaces=True, skip_special_tokens=True)
            logger.info("***** Writing prediction for article {}".format(article_names[i]))
            chunk = re.match(r".*_data_(.*)", args.data_dir).group(1)
            path = os.path.join("dev_summaries_{}_wo_penalty2".format(chunk), "{}_generated.txt".format(article_names[i]))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as summary_file:
                summary_file.write(seq_decoded)
        toc = time.perf_counter()
        print("Prediction of batch {} took {} seconds".format(step, (toc-tic)))


def main():
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
    parser.add_argument("--do_train", type=bool, default=False, help="Run training.")
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
        # baseline on dev-set
        logger.info("***** Running Evaluation *****")
        if args.model_name_or_path == "xlnet-base-cased":
            logger.info("***** Running baseline *****")
        else:
            logger.info("***** Running test *****")
        evaluate(args, model, tokenizer)
        logger.info("***** Evaluation finished *****")

if __name__ == "__main__":
    main()
