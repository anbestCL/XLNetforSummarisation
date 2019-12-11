from pytorch_transformers import XLNetTokenizer, XLNetLMHeadModel
import torch
import nltk
import time
from rouge import Rouge

#path = "test/0a00d5b9c9fce638cd7d7bd010ecc2cf09c01f5f.story"
path = "test/0a0a1a0e94aac65f2b50815050dfaccf521dda35.story"
with open(path) as file:
    text = [line.strip() for line in file]
    #text = [sent for line in file for sent in nltk.sent_tokenize(line.strip())]

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
tokenizer = XLNetTokenizer.from_pretrained('xlnet-large-cased')
assert tokenizer.mask_token == "<mask>"
model = XLNetLMHeadModel.from_pretrained('xlnet-large-cased')
# We show how to setup inputs to predict a next token using a bi-directional context.
article_tokens = " ".join(article)
summary_tokens = " ".join(summary)
#sum_len = len(summary_tokens)
sum_len = 10
summary_mask = " ".join(["<mask>"]*sum_len)
art_len = len(article_tokens)

# batch_size x seq_len
input_ids = torch.tensor(tokenizer.encode(summary_mask + article_tokens)).unsqueeze(0)
seq_len = input_ids.shape[1]


def top10_predictions(next_token_logits):
    for pred_pos in range(next_token_logits.shape[1]):
        predicted_logits_list, predicted_indexes_list = torch.topk(next_token_logits[:, pred_pos, :], k=10)

        print("predicted <masked> words:")
        for i, item in enumerate(predicted_indexes_list[0]):
            the_index = predicted_indexes_list[0][i].item()
            print("word and logits", tokenizer.decode(the_index), predicted_logits_list[0][i].item())


def top1_predictions(next_token_logits):
    predicted_logits = []
    predicted_words = []
    print("predicted <masked> words:")
    for pred_pos in range(next_token_logits.shape[1]):
        predicted_logit, predicted_index = torch.topk(next_token_logits[:, pred_pos, :], k=1)

        the_index = predicted_index[0][0].item()
        predicted_words.append(tokenizer.decode(the_index))
        predicted_logits.append(predicted_logit[0][0].item())

    print(" ".join(predicted_words))
    print(predicted_logits)



def test_1():
    ##### VERSION 1 #####
    perm_mask = torch.zeros(1, seq_len, seq_len)
    for i in range(seq_len):
        for j in range(seq_len):
            if i <= sum_len and j <= sum_len:
                if i == j:
                    perm_mask[0, i, j] = 1
                elif i < j:
                    perm_mask[0, i, j] = 1
    assert perm_mask[-1, sum_len+1, sum_len+1] == 0

    target_mapping = torch.zeros((1, sum_len, seq_len), dtype=torch.float)
    for i in range(sum_len):
        for j in range(seq_len):
            if j <= sum_len:
                if i == j:
                    target_mapping[0, i, j] = 1
    assert target_mapping[-1, 1, sum_len+1] == 0


    outputs = model(input_ids, perm_mask=perm_mask, target_mapping=target_mapping)
    next_token_logits = outputs[0]  # Output has shape [target_mapping.size(0), target_mapping.size(1), config.vocab_size]

    top10_predictions(next_token_logits)
    top1_predictions(next_token_logits)


def test_2():

    def create_perm_mask(predict_pos):
        perm_mask = torch.zeros(1, seq_len, seq_len)
        perm_mask[:, 0:sum_len, predict_pos:sum_len] = 1
        assert perm_mask[-1, sum_len + 1, sum_len + 1] == 0
        return perm_mask

    def create_target_mapping(predict_pos):
        target_mapping = torch.zeros((1, 1, seq_len), dtype=torch.float)
        target_mapping[:, :, predict_pos] = 1
        return target_mapping

    tic = time.perf_counter()
    predicted_words = []
    predicted_logits = []
    for predict_pos in range(sum_len):
        tic2 = time.perf_counter()
        perm_mask = create_perm_mask(predict_pos)
        target_mapping = create_target_mapping(predict_pos)

        outputs = model(input_ids, perm_mask=perm_mask, target_mapping=target_mapping)
        next_token_logits = outputs[0]  # Output has shape [target_mapping.size(0), target_mapping.size(1), config.vocab_size]

        predicted_logit, predicted_index = torch.topk(next_token_logits[0], k=1)

        predicted_words.append(tokenizer.decode(predicted_index[0][0].item()))
        predicted_logits.append(predicted_logit[0][0].item())
        #print(predicted_logit[0][0].item())
        #print(tokenizer.decode(predicted_index[0][0].item()))
        #replace prediction in input
        input_ids[0, predict_pos] = predicted_index[0][0]
        toc2 = time.perf_counter()
        print(toc2-tic2)
        print(" ".join(predicted_words))
        print(predicted_logits)
    toc = time.perf_counter()
    print(" ".join(predicted_words))
    print(predicted_logits)

    r = Rouge()

    scores = r.get_scores([" ".join(predicted_words)], [summary_tokens])

    print(scores)

    print(toc-tic)

test_2()