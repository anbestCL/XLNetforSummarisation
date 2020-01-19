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

from sklearn.model_selection import train_test_split
import numpy as np
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

    return train_stories, dev_stories, test_stories


def collate(data, tokenizer, mode, args, sum_len=None, predict_pos=None):
    """ storyname, stories, summaries per article in batch as input """

    # remove the files with empty an story/summary
    data = filter(lambda x: not (len(x[1]) == 0 or len(x[2]) == 0), data)
    story_names, stories, summaries = zip(*list(data))

    # max_seqlen set as parameter for training on GPU
    if args.max_seqlen is None:
        args.max_seqlen = set_max_seqlen(stories, summaries, tokenizer)

    input_ids, stories_ids, summaries_ids, sum_lens = zip(*list([encode_for_summarization(mode, story, summary, story_name, tokenizer, args.max_seqlen, sum_len) for (story_name, story, summary) in zip(story_names, stories, summaries)]))
    pad_sum_len = max(sum_lens)
    perm_masks = torch.cat([build_perm_mask(sum_len, args.max_seqlen) for sum_len in sum_lens], dim=0)
    target_mappings = pad_target_mapping(mode, sum_lens, args.case, pad_sum_len, args.max_seqlen, predict_pos)
    summaries_ids = pad_summaries_ids(mode, summaries_ids, args.case, pad_sum_len, args.max_seqlen, sum_len)

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
# Train
# ------------


def train(args, model, tokenizer, train_paths):
    """ Fine-tune the pretrained model on the corpus. """
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
            input_ids, stories, summaries, attention_mask, perm_mask, target_mapping = batch

            input_ids = input_ids.to(args.device)
            perm_mask = perm_mask.to(args.device)
            attention_mask = attention_mask.to(args.device)
            target_mapping = target_mapping.to(args.device)
            summaries = summaries.to(args.device)

            if args.case == 2:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    perm_mask=perm_mask,
                    labels=summaries
                )

            if args.case == 1:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    perm_mask=perm_mask,
                    target_mapping = target_mapping,
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

        curr_rouge_score = evaluate(args, model, tokenizer)

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
    set_seed(args)

    eval_dataset = load_and_cache_examples(tokenizer=tokenizer, data_dir=args.data_dir)
    eval_dataset.stories_path = dev_paths

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

    if args.lead:
        rouge_score_article = []
    rouge_score = []

    model.eval()
    for step, batch in enumerate(eval_dataloader):
        print(eval_dataloader)
        print("Batch {}".format(step))
        input_ids, stories, summaries, attention_mask, perm_mask, target_mapping, article_names = batch
        print("Article {}".format(article_names))
        predicted_sequence = torch.zeros((input_ids.shape[0], args.sum_len), dtype=torch.int32)

        input_ids = input_ids.to(args.device)
        perm_mask = perm_mask.to(args.device)
        attention_mask = attention_mask.to(args.device)

        for predict_pos in range(args.sum_len):
            print("Prediction position {}".format(predict_pos))
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
            # repetition penalty from CTRL paper (https://arxiv.org/abs/1909.05858)
            if args.repetition_penalty != 1.0:
                for i in range(args.batch_size):
                    for previous_tokens in set(input_ids[i].tolist()):
                        # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
                        if next_token_logits[i, -1, previous_tokens] < 0:
                            next_token_logits[i, -1, previous_tokens] *= args.repetition_penalty
                        else:
                            next_token_logits[i, -1, previous_tokens] /= args.repetition_penalty
            _, predicted_indices = torch.max(next_token_logits.view(input_ids.shape[0], -1), dim=1, keepdim=True)

            for i in range(predicted_indices.shape[0]):
                predicted_sequence[i, predict_pos] = int(predicted_indices[i].item())

                # replace prediction in input
                input_ids[i, predict_pos] = predicted_indices[i].item()

        # rouge score per predicted sentence
        sent_score = compute_rouge_score(tokenizer, predicted_sequence, summaries, article_names)
        rouge_score.append(sent_score)

        if args.lead:
            article_beginning = [story[0].tolist()[:args.sum_len] for story in stories]
            sent_score_article = compute_rouge_score(tokenizer, predicted_sequence, article_beginning, article_names)
            rouge_score_article.append(sent_score_article)

    def _average_rouge_score(score):
        rouge_dict = {rouge_type: {key: 0 for key in ['f', 'p', 'r']} for rouge_type in ['rouge-1', 'rouge-2', 'rouge-l']}
        for batch in score:
            for rouge_type, values in batch.items():
                for measure, value in values.items():
                    rouge_dict[rouge_type][measure] += value
        rouge_dict = {rouge_type: {measure: value / len(score) for measure, value in values.items()} for
                      rouge_type, values in rouge_dict.items()}
        return rouge_dict

    rouge_dict = _average_rouge_score(rouge_score)
    if args.lead:
        rouge_article_dict = _average_rouge_score(rouge_score_article)

    # Save the evaluation's results
    output_eval_file = os.path.join(args.output_dir, "eval_results.txt")

    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results *****")
        logger.info("***** Rouge scores with summary *****")
        for key in sorted(rouge_dict.keys()):
            logger.info("  %s = %s", key, str(rouge_dict[key]))
            writer.write("%s = %s\n" % (key, str(rouge_dict[key])))
        if args.lead:
            logger.info("***** Rouge scores with first %d words in article *****", args.sum_len)
            for key in sorted(rouge_article_dict.keys()):
                logger.info("  %s = %s", key, str(rouge_article_dict[key]))
                writer.write("%s = %s\n" % (key, str(rouge_article_dict[key])))

    return rouge_dict


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
        "--case",
        type=int,
        help="case=2 : no target_mapping, labels=summaries padded with -1, case=1: target_mapping, labels=summaries padded to max_sum_len",
    )
    parser.add_argument(
        "--lead",
        type=bool,
        default=False,
        help="flag whether lead baseline or not",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.0,
        help="The parameter for repetition penalty. Between 1.0 and + infinity. 1.0 means no penalty. Default to 1.",
    )

    args = parser.parse_args()

    # Set up training device
    if torch.cuda.is_available():
        args.device = torch.device("cpu")
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
        # baseline on dev-set
        if args.lead:
            logger.info("***** Running lead baseline *****")
            evaluate(args, model, tokenizer, dev_paths)
        # test set check
        else:
            logger.info("***** Running testing *****")
            evaluate(args, model, tokenizer, test_paths)


if __name__ == "__main__":
    main()