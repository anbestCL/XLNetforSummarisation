import nltk
from pytorch_transformers import XLNetTokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import torch


class ArticlePrep:

    def __init__(self, path, max_length):
        with open(path) as file:
            self.article = [sent for line in file for sent in nltk.sent_tokenize(line.strip())]

        self.max_length = max_length
        input_ids, output_ids = self._create_inputs_outputs()

        self.attention_mask, self.perm_mask = self._create_masks(input_ids)
        self.inputs = self.convert_to_tensor(input_ids)
        self.outputs = self.convert_to_tensor(output_ids)

    def _create_inputs_outputs(self):
        sentences = [sent + " [SEP] [CLS]" for sent in self.article]

        tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased', do_lower_case=True)

        tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]

        # create input and output
        # input must be sequence excluding first sentence
        # output will be first sentence
        inputs = [sent for i, sent in enumerate(tokenized_texts) if i >= 0]
        outputs = [sent for i, sent in enumerate(tokenized_texts) if i == 0]

        max_input_length = 128
        max_output_length = 128
        # max_input_length = len(max(inputs, key = lambda sent: len(sent)))
        # max_output_length = len(max(outputs, key = lambda sent: len(sent)))

        # Use the XLNet tokenizer to convert the tokens to their index numbers in the XLNet vocabulary
        input_ids = [tokenizer.convert_tokens_to_ids(x) for x in inputs]
        output_ids = [tokenizer.convert_tokens_to_ids(x) for x in outputs]

        # Pad our input tokens
        input_ids = pad_sequences(input_ids, maxlen=max_input_length, dtype="long", truncating="post", padding="post")
        output_ids = pad_sequences(output_ids, maxlen=max_output_length, dtype="long", truncating="post", padding="post")

        return input_ids, output_ids

    def _create_masks(self, input_ids):
        # Create attention masks
        attention_masks = []

        # Create a mask of 1s for each token followed by 0s for padding
        for seq in input_ids:
            seq_mask = [float(i > 0) for i in seq]
            attention_masks.append(seq_mask)

        # Create permutation mask (attend only to 2nd and later sentences
        # Mask to indicate the attention pattern for each input token with values selected in [0, 1]:
        # If perm_mask[k, i, j] = 0, i attend to j in batch k; if perm_mask[k, i, j] = 1, i does not attend to j in batch k
        perm_masks = np.zeros((len(input_ids), self.max_length, self.max_length))
        for i, sent in enumerate(input_ids):
            for j, word in enumerate(sent):
                for k, word2 in enumerate(sent):
                    if i == 0:
                        perm_masks[i, j, k] = 1.0
                    else:
                        perm_masks[i, j, k] = 0.0

        return torch.tensor(attention_masks), torch.tensor(perm_masks)

    def convert_to_tensor(self, list_input):
        return torch.tensor(list_input)




if __name__ == "__main__":
    #test path
    path = "test/0a0a1a0e94aac65f2b50815050dfaccf521dda35.story"
    test = ArticlePrep(path, max_length=128)


