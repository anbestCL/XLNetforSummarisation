from utils_summarisation import CNNDailyMailDataset
from pytorch_transformers import XLNetTokenizer
import os

tokenizer = XLNetTokenizer.from_pretrained("xlnet-large-cased")
dataset = CNNDailyMailDataset(tokenizer=tokenizer, data_dir="data")

for i in range(len(dataset.stories_path)):
    story_name, story_lines, summary_lines = dataset[i]
    story = " ".join(story_lines)
    summary = " ".join(summary_lines)
    story_path = os.path.join("stories", "{}_story.txt".format(story_name))
    if not os.path.exists(story_path):
        with open(story_path, "w") as story_file:
            story_file.write(story)
    summary_path = os.path.join("summaries", "{}_summary.txt".format(story_name))
    if not os.path.exists(summary_path):
        with open(summary_path, "w") as summary_file:
            summary_file.write(summary)
