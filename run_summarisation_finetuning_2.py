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

from rouge import Rouge
import argparse
import functools
import logging
import os
import random
import sys
from collections import defaultdict
import re

from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm, trange
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from pytorch_transformers import XLNetTokenizer, XLNetLMHeadModel, XLNetConfig, AdamW

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


def train_dev_test_split(dataset):
    train_dev_stories, test_stories = train_test_split(dataset.stories_path, train_size=0.8)
    train_stories, dev_stories = train_test_split(train_dev_stories, train_size=0.8)

    def _write_stories_to_set(stories, type):
        dir_cnn = "{}/cnn/stories".format(type)
        dir_dailymail = "{}/dailymail/stories".format(type)
        if not os.path.exists(dir_cnn):
            os.makedirs(dir_cnn)
        if not os.path.exists(dir_dailymail):
            os.makedirs(dir_dailymail)
        for story in stories:
            if "cnn" in story:
                dir = dir_cnn
            else:
                dir = dir_dailymail
            story_name = re.search('(stories\/)(.*)', story).group(2)
            with open("{}/{}".format(dir, story_name), "w") as copy_story_file:
                with open(story, "r") as story_file:
                    raw_story = story_file.read()
                    copy_story_file.write(raw_story)

    _write_stories_to_set(train_stories, "train")
    _write_stories_to_set(dev_stories, "dev")
    _write_stories_to_set(test_stories, "test")



def collate(data, tokenizer, max_seqlen, case, sum_len=None, predict_pos=None):
    """ storyname, stories, summaries per article in batch as input """

    # remove the files with empty an story/summary
    data = filter(lambda x: not (len(x[1]) == 0 or len(x[2]) == 0), data)
    story_names, stories, summaries = zip(*list(data))

    # max_seqlen set as parameter for training on GPU
    if max_seqlen is None:
        max_seqlen = set_max_seqlen(stories, summaries, tokenizer)

    # sum_len set for eval case
    input_ids, stories_ids, summaries_ids, sum_lens = zip(*list([encode_for_summarization(story, summary, story_name, tokenizer, max_seqlen, sum_len) for (story_name, story, summary) in zip(story_names, stories, summaries)]))

    pad_sum_len = max(sum_lens)

    attention_masks = torch.cat([build_attention_mask(story.shape[1], max_seqlen)
    for story in stories_ids], dim=0)

    perm_masks = torch.cat([build_perm_mask(sum_len, max_seqlen) for sum_len in sum_lens], dim=0)

    input_ids = torch.cat(input_ids, dim=0)

    summaries_ids = pad_summaries_ids(summaries_ids, case, pad_sum_len, max_seqlen, sum_len)

    target_mappings = pad_target_mapping(sum_lens, case, pad_sum_len, max_seqlen, predict_pos, sum_len)

    if sum_len is not None:
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
# Train
# ------------


def train(args, model, tokenizer, case):
    """ Fine-tune the pretrained model on the corpus. """
    set_seed(args)

    # Load the data
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_dataset = load_and_cache_examples("train", tokenizer)

    train_sampler = RandomSampler(train_dataset)
    model_collate_fn = functools.partial(collate, tokenizer=tokenizer, max_seqlen=args.max_seqlen, case=case)
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

    global_loss = {'train':0, 'dev':0} #loss
    best_dev_score = 0.0 # rouge-1 f-score
    for _ in train_iterator:
        tr_loss = 0.0
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=True)
        for step, batch in enumerate(epoch_iterator):
            input_ids, stories, summaries, attention_mask, perm_mask, target_mapping = batch

            input_ids = input_ids.to(args.device)
            perm_mask = perm_mask.to(args.device)
            attention_mask = attention_mask.to(args.device)
            target_mapping = target_mapping.to(args.device)
            summaries = summaries.to(args.device)

            model.train()

            if case == 2:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    perm_mask=perm_mask,
                    labels=summaries
                )

            if case == 1:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    perm_mask=perm_mask,
                    target_mapping = target_mapping,
                    labels=summaries
                )

            loss = outputs[0]
            print("Case {}: Loss = {}".format(case, loss))
            tr_loss += loss
            loss.backward()
            optimizer.step()

        tr_loss /= step
        global_loss['train'] = tr_loss

        curr_dev_loss, curr_rouge_score = evaluate(args, model, tokenizer)
        global_loss['dev'] = curr_dev_loss

        curr_dev_score = curr_rouge_score['rouge-1']['f']
        if curr_dev_score > best_dev_score:
            best_dev_score = curr_dev_score
            logger.info("Saving model checkpoint to %s", args.output_dir)

            # Save a trained model, configuration and tokenizer using `save_pretrained()`.
            # They can then be reloaded using `from_pretrained()`
            model_to_save = (
                model.module if hasattr(model, "module") else model
            )  # Take care of distributed/parallel training
            model_to_save.save_pretrained(args.output_dir)
            torch.save(args, os.path.join(args.output_dir, "training_arguments.bin"))

    return global_loss, best_dev_score


# ------------
# Evaluate
# ------------

def calculate_rouge_score(tokenizer, predicted_sequences, summaries, article_names):
    r = Rouge()
    results = defaultdict(dict)
    hyps = [tokenizer.decode(seq.toList(), clean_up_tokenization_spaces=True, skip_special_tokens=True) for seq in predicted_sequences]
    refs = [tokenizer.decode(summary.numpy().toList(), clean_up_tokenization_spaces=True, skip_special_tokens=True) for summary in summaries]
    scores = r.get_scores(hyps, refs, avg=True) #avg=True to get mean values
    logger.info("***** Rouge scores {} *****".format(scores))
    for i, article in enumerate(article_names):
        with open("summaries/{}_summary.txt".format(article), "w") as summary_file:
            summary_file.write(" ".join(hyps[i]))
    return results

def evaluate(args, model, tokenizer):
    set_seed(args)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_dataset = load_and_cache_examples("dev", tokenizer)
    model_collate_fn = functools.partial(collate, tokenizer=tokenizer, max_seqlen=args.max_seqlen, case=args.case, sum_len=args.sum_len, predict_pos=0)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=model_collate_fn
    )

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    logger.info("***** Running evaluation {} *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)


    eval_loss = 0.0
    rouge_score = []
    nb_eval_steps = 0

    model.eval()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):

        input_ids, stories, summaries, attention_mask, perm_mask, target_mapping, article_names = batch
        predicted_sequence = torch.zeros((input_ids.shape[0], args.sum_len))
        seq_loss = 0.0

        for predict_pos in range(args.sum_len):
            if predict_pos > 0:
                target_mapping = build_target_mapping(seq_len=input_ids.shape[1], predict_pos=predict_pos)
            input_ids = input_ids.to(args.device)
            target_mapping = target_mapping.to(args.device)
            perm_mask = perm_mask.to(args.device)
            attention_mask = attention_mask.to(args.device)
            label = summaries[:, predict_pos].long()
            label = label.to(args.device)

            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    perm_mask=perm_mask,
                    target_mapping=target_mapping,
                    labels=label
                )

                lm_loss = outputs[0]
                seq_loss += lm_loss.item()

                # Output has shape [target_mapping.size(0), target_mapping.size(1), config.vocab_size]
                next_token_logits = outputs[1]

                _, predicted_indices = torch.max(next_token_logits[0].view(2, -1), dim=1, keepdim=True)

                for i in range(len(predicted_indices)):
                    predicted_sequence[i, predict_pos] = predicted_indices[i].item()

                    # replace prediction in input
                    input_ids[i, predict_pos] = predicted_indices[i].item()

        eval_loss += seq_loss/args.sum_len

        rouge_score.append(calculate_rouge_score(tokenizer, predicted_sequence, summaries, article_names))
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    rouge_dict = {rouge_type: {key: 0 for key in ['f', 'p', 'r']} for rouge_type in ['rouge-1', 'rouge-2', 'rouge-l']}
    for batch in rouge_score:
        for rouge_type, values in batch.items():
            for measure, value in values.items():
                rouge_dict[rouge_type][measure] += value
    rouge_dict = {rouge_type: {measure: value / len(rouge_score) for measure, value in values.items()} for
                  rouge_type, values in rouge_dict.items()}

    # Save the evaluation's results
    output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results *****")
        logger.info(" eval loss = %s", eval_loss)
        for key in sorted(rouge_dict.keys()):
            logger.info("  %s = %s", key, str(rouge_dict[key]))
            writer.write("%s = %s\n" % (key, str(rouge_dict[key])))

    return eval_loss, rouge_dict


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
        "--per_gpu_eval_batch_size",
        default=2,
        type=int,
        help="Batch size per GPU/CPU for training.",
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
        "--case",
        type=int,
        help="case=1 : no target_mapping, labels=summaries padded with -1, case=2: target_mapping, labels=summaries padded to max_sum_len",
    )


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
    model = XLNetLMHeadModel.from_pretrained(args.model_name_or_path)

    # Make train-test-dev-split
    dataset = load_and_cache_examples(args.data_dir, tokenizer)
    train_dev_test_split(dataset)

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
        global_loss, best_dev_score = train(args, model, tokenizer, args.case)
        logger.info(" train loss history = %s, \n, dev loss history = %s, \n best rouge-1 f-score on dev set = %s", global_loss['train'], global_loss['dev'], best_dev_score)

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

    # Evaluate the model
    if args.do_evaluate:
        evaluate(args, model, tokenizer)


if __name__ == "__main__":
    main()