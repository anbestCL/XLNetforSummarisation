from ArticlePrep import ArticlePrep
import os
import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from pytorch_transformers import XLNetLMHeadModel, XLNetTokenizer
from pytorch_transformers import AdamW

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#n_gpu = torch.cuda.device_count()
#torch.cuda.get_device_name(0)

'''
# for future with batch size
batch_size = 32
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
'''


# Load Model and specify parameters
tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased', do_lower_case=True)
model = XLNetLMHeadModel.from_pretrained("xlnet-base-cased")
if torch.cuda.is_available(): model.to('cuda')

param_optimizer = list(model.named_parameters())

'''
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters,
                     lr=2e-5)
'''


# load data
directory = "test/"
# text consists of articles
for filename in os.listdir("test/"):
    path = os.path.join(directory, filename)
    article = ArticlePrep(path, max_length=128)


    # Eval mode
    model.eval()

    # Tracking variables
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    # Evaluate data for one epoch
    #for batch in validation_dataloader:
        # Add batch to GPU
        #batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        #b_input_ids, b_input_mask, b_labels = batch
        # Telling the model not to compute or store gradients, saving memory and speeding up validation
    with torch.no_grad():
        # Forward pass, calculate logit predictions
        output = model(article.inputs, token_type_ids=None, attention_mask=article.attention_mask, perm_mask=article.perm_mask)

    print(output)

    predicted_k_indexes = torch.topk(output[0][0][0], k=10)
    predicted_logits_list = predicted_k_indexes[0]
    predicted_indexes_list = predicted_k_indexes[1]

    print("predicted word:", tokenizer.decode(article.inputs[0][0].item()))
    for i, item in enumerate(predicted_indexes_list):
        the_index = predicted_indexes_list[i].item()
        print("word and logits", tokenizer.decode(the_index), predicted_logits_list[i].item())

    # Move logits and labels to CPU
    #logits = logits.detach().cpu().numpy()
    #label_ids = b_labels.to('cpu').numpy()


