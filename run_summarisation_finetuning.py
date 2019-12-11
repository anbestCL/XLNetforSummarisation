#!/usr/bin/python3


# coding=utf-8
# Copyright 2019 The HuggingFace Inc. team.
# Copyright (c) 2019 The HuggingFace Inc.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Finetuning seq2seq models for sequence generation.#

import argparse
import functools
import logging
import os
import random
import sys

import numpy as np
from tqdm import tqdm, trange
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from pytorch_transformers import XLNetTokenizer, XLNetLMHeadModel, XLNetConfig, AdamW

from utils_summarisation import CNNDailyMailDataset, set_max_seqlen, encode_for_summarization, build_attention_mask, build_perm_mask, build_target_mapping, pad_summary

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


# ------------
# Load dataset
# ------------


def load_and_cache_examples(args, tokenizer):
    dataset = CNNDailyMailDataset(tokenizer, data_dir=args.data_dir)
    return dataset


def collate(data, tokenizer, max_seqlen, sum_len = None):
    """ List of tuple as an input. """

    #TODO: implement padding with max_seq_len and predict pos

    # remove the files with empty an story/summary
    data = filter(lambda x: not (len(x[1]) == 0 or len(x[2]) == 0), data)
    story_names, stories, summaries = zip(*list(data))
    if max_seqlen is None:
        max_seqlen = set_max_seqlen(stories, summaries, tokenizer)
    input_ids, stories_ids, summaries_ids, sum_lens = zip(*list([encode_for_summarization(story, summary, story_name, tokenizer, max_seqlen, sum_len) for (story_name, story, summary) in zip(story_names, stories, summaries)]))
    pad_sum_len = max(sum_lens)
    attention_masks = torch.cat([build_attention_mask(story.shape[1], max_seqlen)
    for story in stories_ids], dim=0)
    perm_masks = torch.cat([build_perm_mask(sum_len, max_seqlen) for sum_len in sum_lens], dim=0)
    target_mappings = torch.cat([build_target_mapping(sum_len, max_seqlen, pad_sum_len=pad_sum_len) for sum_len in sum_lens], dim=0)

    input_ids = torch.cat(input_ids, dim=0)
    summaries_ids = torch.cat([pad_summary(summary, pad_sum_len) for summary in summaries_ids], dim=0)

    return (
        input_ids,
        stories_ids,
        summaries_ids,
        attention_masks,
        perm_masks,
        target_mappings,
    )


# ------------
# Train
# ------------


def train(args, model, tokenizer):
    """ Fine-tune the pretrained model on the corpus. """
    set_seed(args)

    # Load the data
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_dataset = load_and_cache_examples(args, tokenizer)

    train_sampler = RandomSampler(train_dataset)
    model_collate_fn = functools.partial(collate, tokenizer=tokenizer, max_seqlen=args.max_seqlen)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.train_batch_size,
        collate_fn=model_collate_fn,
    )

    # Training schedule
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = t_total // (
                len(train_dataloader) // args.gradient_accumulation_steps + 1
        )
    else:
        t_total = (
                len(train_dataloader)
                // args.gradient_accumulation_steps
                * args.num_train_epochs
        )

    # TODO: set optimizerxlnet
    #Optimizer
    #no_decay = ['bias', 'LayerNorm.weight']
    #optimizer_grouped_parameters = [
        #{'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         #'weight_decay': args['weight_decay']},
        #{'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    #]

    #warmup_steps = math.ceil(t_total * args['warmup_ratio'])
    #args['warmup_steps'] = warmup_steps if args['warmup_steps'] == 0 else args['warmup_steps']

    #optimizer = AdamW(optimizer_grouped_parameters, lr=args['learning_rate'], eps=args['adam_epsilon'])
    optimizer = AdamW(model.parameters())

    # Train
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info(
        "  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size
    )
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size * args.gradient_accumulation_steps
        # * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    model.zero_grad()
    train_iterator = trange(args.num_train_epochs, desc="Epoch", disable=True)

    global_step = 0
    tr_loss = 0.0
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=True)
        for step, batch in enumerate(epoch_iterator):
            input_ids, stories, summaries, attention_mask, perm_mask, target_mapping = batch

            input_ids = input_ids.to(args.device)
            target_mapping = target_mapping.to(args.device)
            perm_mask = perm_mask.to(args.device)
            attention_mask = attention_mask.to(args.device)
            summaries = summaries.to(args.device)

            model.train()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                perm_mask=perm_mask,
                labels=input_ids
            )

            loss = outputs[0]
            print(loss)
            if args.gradient_accumulation_steps > 1:
                loss /= args.gradient_accumulation_steps

            loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                model.zero_grad()
                global_step += 1

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    return global_step, tr_loss / global_step


# ------------
# Evaluate
# ------------


def evaluate(args, model, tokenizer, prefix=""):
    set_seed(args)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=True)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size
    )

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        source, target, encoder_token_type_ids, encoder_mask, decoder_mask, lm_labels = batch

        source = source.to(args.device)
        target = target.to(args.device)
        encoder_token_type_ids = encoder_token_type_ids.to(args.device)
        encoder_mask = encoder_mask.to(args.device)
        decoder_mask = decoder_mask.to(args.device)
        lm_labels = lm_labels.to(args.device)

        with torch.no_grad():
            outputs = model(
                source,
                target,
                encoder_token_type_ids=encoder_token_type_ids,
                encoder_attention_mask=encoder_mask,
                decoder_attention_mask=decoder_mask,
                decoder_lm_labels=lm_labels,
            )
            lm_loss = outputs[0]
            eval_loss += lm_loss.mean().item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))

    result = {"perplexity": perplexity}

    # Save the evaluation's results
    output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    return result


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
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # Optional parameters
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--do_evaluate",
        type=bool,
        default=False,
        help="Run model evaluation on out-of-sample data.",
    )
    parser.add_argument("--do_train", type=bool, default=False, help="Run training.")
    parser.add_argument(
        "--do_overwrite_output_dir",
        type=bool,
        default=False,
        help="Whether to overwrite the output dir.",
    )
    parser.add_argument(
        "--model_name_or_path",
        default="xlnet-base-cased",
        type=str,
        help="The model checkpoint to initialize the encoder and decoder's weights with.",
    )

    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument(
        "--to_cpu", default=False, type=bool, help="Whether to force training on CPU."
    )
    parser.add_argument(
        "--num_train_epochs",
        default=1,
        type=int,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--per_gpu_train_batch_size",
        default=1,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--max_seqlen",
        type=int,
        help="Maximal sequence length, longer sentences are truncated",
    )
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    if (
            os.path.exists(args.output_dir)
            and os.listdir(args.output_dir)
            and args.do_train
            and not args.do_overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --do_overwrite_output_dir to overwrite.".format(
                args.output_dir
            )
        )

    # Set up training device
    if args.to_cpu or not torch.cuda.is_available():
        args.device = torch.device("cpu")
        args.n_gpu = 0
    else:
        args.device = torch.device("cuda")
        args.n_gpu = torch.cuda.device_count()

    # Load pretrained model and tokenizer
    tokenizer = XLNetTokenizer.from_pretrained(args.model_name_or_path)
    #config = XLNetConfig.from_pretrained(args.model_name_or_path)
    model = XLNetLMHeadModel.from_pretrained(args.model_name_or_path)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        0,
        args.device,
        args.n_gpu,
        False,
        False,
    )

    logger.info("Training/evaluation parameters %s", args)

    # Train the model
    model.to(args.device)
    if args.do_train:
        global_step, tr_loss = train(args, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)

        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        torch.save(args, os.path.join(args.output_dir, "training_arguments.bin"))

    # Evaluate the model
    results = {}
    if args.do_evaluate:
        checkpoints = []
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            encoder_checkpoint = os.path.join(checkpoint, "encoder")
            decoder_checkpoint = os.path.join(checkpoint, "decoder")
            model = PreTrainedEncoderDecoder.from_pretrained(
                encoder_checkpoint, decoder_checkpoint
            )
            model.to(args.device)
            results = "placeholder"

    return results


if __name__ == "__main__":
    main()