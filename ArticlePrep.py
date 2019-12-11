import nltk
from pytorch_transformers import XLNetTokenizer
from keras.preprocessing.sequence import pad_sequences
import torch


class ArticlePrep:

    def __init__(self, path, max_length):
        with open(path) as file:
            text = [sent for line in file for sent in nltk.sent_tokenize(line.strip())]

        self.max_length = max_length
        input_ids, output_ids, summary_length = self._create_inputs_outputs(text)

        self.attention_mask, self.perm_mask = self._create_masks(input_ids, summary_length)
        self.inputs = self.convert_to_tensor(input_ids)
        self.outputs = self.convert_to_tensor(output_ids)

    def _create_inputs_outputs(self, text):
        """Splitting of summary from article and converting both to ids and padding"""
        article = []
        summary = []
        highlight_flag = False
        for sent in text:
            if '@highlight' in sent:
                highlight_flag= True
            elif '@highlight' not in sent:
                if not highlight_flag:
                    article.append(sent)
                else:
                    summary.append(sent)

        article = summary + article
        sentences = [sent + " [SEP] [CLS]" for sent in article]

        tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased', do_lower_case=True)

        tokenized_article = [tokenizer.tokenize(sent) for sent in sentences]

        # Use the XLNet tokenizer to convert the tokens to their index numbers in the XLNet vocabulary
        input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_article]
        output_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_article[:len(summary)]]

        # Before padding create batches of articles -> flatten  inputs and outputs
        inputs = [id for sent in input_ids for id in sent]
        outputs = [id for sent in output_ids for id in sent]

        # Pad our input tokens
        input_ids = pad_sequences([inputs], maxlen=self.max_length, dtype="long", truncating="post", padding="post")
        output_ids = pad_sequences([outputs], maxlen=self.max_length, dtype="long", truncating="post", padding="post")

        return input_ids, output_ids, len(outputs)

    def _create_masks(self, input_ids, summary_length):
        """Create attention mask to mask padded tokens and perm_mask to create attention pattern for generation"""

        # Create a mask of 1s for each token followed by 0s for each padded token
        attention_masks = torch.ones(len(input_ids[0]))
        for i, seq in enumerate(input_ids[0]):
            if seq == 0:
                attention_masks[i] = 0

        # Create a mask such that
        # w_1 of input attends to all article sentences, w_2 attends to w_1 and all article sentences etc.
        # 1 for attention, 0 for no attention
        # If perm_mask[k, i, j] = 0, i attend to j in batch k;
        # if perm_mask[k, i, j] = 1, i does not attend to j in batch k
        perm_masks = torch.zeros(1, self.max_length, self.max_length)
        for i in range(summary_length):
            for j in range(summary_length):
                if i == j:
                    perm_masks[0, i, j] = 1
                elif i < j:
                    perm_masks[0, i, j] = 1

        return attention_masks, perm_masks

    def convert_to_tensor(self, list_input):
        return torch.tensor(list_input)




if __name__ == "__main__":
    #testing path
    path = "testing/0a0a1a0e94aac65f2b50815050dfaccf521dda35.story"
    test = ArticlePrep(path, max_length=2000)


