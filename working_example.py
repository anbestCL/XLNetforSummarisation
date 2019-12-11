from pytorch_transformers import XLNetTokenizer, XLNetLMHeadModel
import torch
import time
from rouge import Rouge
import argparse
import os
import json
import re


def _load_and_prepare_input(path, article_name, tokenizer, sum_len):
    with open(path) as file:
        text = [line.strip() for line in file]

    article = []
    summary = []
    highlight_flag = False

    for sent in text:
        if '@highlight' in sent:
            highlight_flag = True
        elif '@highlight' not in sent:
            if not highlight_flag:
                article.append(sent)
            else:
                summary.append(sent)

    article_tokens = " ".join(article)
    with open("{}_article.txt".format(article_name), "w") as article_file:
        article_file.write(article_tokens)
    summary_tokens = " ".join(summary)

    assert tokenizer.mask_token == "<mask>"
    if sum_len is None:
        sum_len = len(summary_tokens)
    summary_mask = " ".join(["<mask>"] * sum_len)

    # batch_size x seq_len
    input_ids = torch.tensor(tokenizer.encode(summary_mask + article_tokens)).unsqueeze(0)
    return input_ids, summary_tokens, sum_len


def main(path, sum_len, version, tokenizer, model):
    article_name = re.match(r"(^.*\/)(.*)(\..*$)", path).group(2)
    with open("{}_log_{}.txt".format(article_name, version), "w") as log:
        log.write("article:{} \nsummary_length:{} \nversion_extended:{}\n \n".format(article_name, sum_len, version))
        input_ids, summary_tokens, sum_len = _load_and_prepare_input(path, article_name, tokenizer, sum_len)

        seq_len = input_ids.shape[1]

        def _create_perm_mask(predict_pos, version):
            perm_mask = torch.zeros(1, seq_len, seq_len)
            if version == "extended":
                for i in range(seq_len):
                    for j in range(seq_len):
                        if i <= sum_len and j <= sum_len:
                            if i == j:
                                perm_mask[0, i, j] = 1
                            elif i < j:
                                perm_mask[0, i, j] = 1
            else:
                perm_mask[:, 0:sum_len, predict_pos:sum_len] = 1
            assert perm_mask[-1, sum_len + 1, sum_len + 1] == 0
            return perm_mask

        def _create_target_mapping(predict_pos):
            target_mapping = torch.zeros((1, 1, seq_len), dtype=torch.float)
            target_mapping[:, :, predict_pos] = 1
            return target_mapping

        tic = time.perf_counter()
        predicted_words = []
        predicted_logits = []
        for predict_pos in range(sum_len):
            tic2 = time.perf_counter()
            perm_mask = _create_perm_mask(predict_pos, version)
            target_mapping = _create_target_mapping(predict_pos)

            outputs = model(input_ids, perm_mask=perm_mask, target_mapping=target_mapping)
            next_token_logits = outputs[0]  # Output has shape [target_mapping.size(0), target_mapping.size(1), config.vocab_size]

            predicted_logit, predicted_index = torch.topk(next_token_logits[0], k=1)

            predicted_words.append(tokenizer.decode(predicted_index[0][0].item()))
            predicted_logits.append(predicted_logit[0][0].item())
            with open("{}_summary_{}.txt".format(article_name, version), "w") as summary_file:
                summary_file.write(" ".join(predicted_words))

            #replace prediction in input
            input_ids[0, predict_pos] = predicted_index[0][0]
            toc2 = time.perf_counter()
            log.write("Predicting {}. position took {} seconds \n".format(predict_pos, toc2-tic2))

        toc = time.perf_counter()

        r = Rouge()
        scores = r.get_scores([" ".join(predicted_words)], [summary_tokens])
        log.write(json.dumps(scores))
        log.write("Article {} took {} minutes".format(article_name, (toc-tic)/60))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="path to folder contaning raw text files")
    parser.add_argument("sum_len", type=int, help="number of words to be predicted for summary")
    parser.add_argument("version", type=str, help="extended or normal perm_mask")

    args = parser.parse_args()
    tokenizer = XLNetTokenizer.from_pretrained('xlnet-large-cased')
    model = XLNetLMHeadModel.from_pretrained("xlnet-large-cased")
    #for file in os.listdir(args.input_dir):
        #filename = os.fsdecode(file)
        #path = os.path.join(args.input_dir, filename)
    path = "testing/0a0a1a0e94aac65f2b50815050dfaccf521dda35.story"
    main(path, args.sum_len, args.version, tokenizer, model)
