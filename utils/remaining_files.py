import argparse
import os
import re

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str)
parser.add_argument("--summary_dir", type=str)

args = parser.parse_args()

for subdir in os.listdir(args.data_dir):
    if subdir != '.DS_Store':
        for file in os.listdir(os.path.join(args.data_dir, subdir, "stories")):
            story_name = re.match(r'(.*).story', file).group(1)
            summary_files = os.listdir(args.summary_dir)
            found = next((summary for summary in summary_files if summary == "{}_{}_generated.txt".format(subdir, story_name)), None)
            if found is None:
                with open(os.path.join(args.data_dir, subdir, "stories", file), "r") as source:
                    text = source.read()
                new_path = os.path.join("{}_rest".format(args.data_dir), subdir, "stories", file)
                os.makedirs(os.path.dirname(new_path), exist_ok=True)
                with open(new_path, "w") as destination:
                    destination.write(text)