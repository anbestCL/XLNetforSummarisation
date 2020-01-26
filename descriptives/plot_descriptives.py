#!/usr/bin/python3

import pickle
import re
import statistics
import matplotlib.pyplot as plt
import operator


def get_stats(filename):
    """Reads in summary and story lengths and computes longest, shortest and average summary and story length.

    Args:
        filename (str): path to file containing summary and story length for each article of data split

    Returns:
        story_lengths (list): contains story lengths of all articles in specific data split
        avg_story_length (float): average story length
        summary_lengths (list): contains summary lengths of all articles in specific data split
        avg_summary_length (float): average summary length

    """
    origin = re.match(r".*param_files\/descriptives_(.*)_(.*).p", filename).group(1).upper()
    set = re.match(r".*param_files\/descriptives_(.*)_(.*).p", filename).group(2).upper()
    with open(filename, "rb") as lengths:
        story_dict = pickle.load(lengths)
        summary_dict = pickle.load(lengths)

    print("*** {} : {} ***".format(origin, set))

    # exclude zero-length summaries and stories
    zero_lengths_stories = [name for name, length in story_dict.items() if length == 0]
    zero_lengths_summaries = [name for name, length in summary_dict.items() if length == 0]
    print(zero_lengths_stories)
    print(zero_lengths_summaries)
    story_lengths = {name: length for name, length in story_dict.items() if name not in zero_lengths_stories and name not in zero_lengths_summaries}
    summary_lengths = {name: length for name, length in summary_dict.items() if name not in zero_lengths_stories and name not in zero_lengths_summaries}
    assert len(story_lengths.values()) == len(summary_lengths.values())
    print(len(story_lengths.values()))

    # longest/shortest/average story
    longest_story, longest_length = max(story_lengths.items(), key=operator.itemgetter(1))
    shortest_story, shortest_length = min(story_lengths.items(), key=operator.itemgetter(1))
    avg_story_length = statistics.mean(story_lengths.values())
    print("Longest story {} with length: {}".format(longest_story, longest_length))
    print("Shortest story {} with length: {}".format(shortest_story, shortest_length))
    print("Average story length: {}".format(avg_story_length))

    # longest/shortest/average summary
    longest_summary, longest_length_sum = max(summary_lengths.items(), key=operator.itemgetter(1))
    shortest_summary, shortest_length_sum = min(summary_lengths.items(), key=operator.itemgetter(1))
    avg_summary_length = statistics.mean(summary_lengths.values())
    print("Longest summary {} length: {}".format(longest_summary, longest_length_sum))
    print("Shortest summary {} length: {}".format(shortest_summary, shortest_length_sum))
    print("Average summary length: {}".format(avg_summary_length))
    return list(story_lengths.values()), avg_story_length, list(summary_lengths.values()), avg_summary_length


def plot_distribution(lengths_cnn, lengths_dm, dataset, text):
    """Plots distribution of story/summary lengths and saves images.

    Args:
        lengths_cnn (list): lists lengths of articles with origin CNN
        lengths_dm (list): lists lengths of articles with origin DailyMail
        dataset (str): indicates data split ('train', 'dev', 'test')
        text (str): 'summaries' or 'stories', used for plot title

    """
    fig, axs = plt.subplots(3,1, tight_layout=True)
    axs[0].set_title("Distribution of {} lengths in {} dataset".format(text, dataset))

    hist, bins, _ = axs[0].hist(lengths_dm, bins=50, alpha=0.75, facecolor='blue', edgecolor='black', linewidth=1.2,
                label='DailyMail {}'.format(text))
    axs[0].set_ylabel("{} count".format(text))
    axs[0].legend()

    axs[1].hist(lengths_cnn, bins=bins, alpha=0.75, facecolor='green', edgecolor='black', linewidth=1.2, label='CNN {}'.format(text))
    axs[1].set_ylabel("{} count".format(text))
    axs[1].legend()

    lengths = lengths_cnn + lengths_dm
    axs[2].hist(lengths, bins=bins, alpha=0.75, facecolor='red', edgecolor='black', linewidth=1.2, label="CNN + DailyMail {}".format(text))
    axs[2].set_xlabel("Token count")
    axs[2].set_ylabel("{} count".format(text))
    axs[2].legend()
    plt.savefig('../descriptives/plots/{}_{}_distribution'.format(dataset, text))


def compute_average(avg1, len1, avg2, len2):
    return (avg1 * len1 + avg2 * len2)/(len1 + len2)


for datset in ["train", "dev", "test"]:
    for origin in ["cnn", "dailymail"]:
        filename_cnn = "../descriptives/param_files/descriptives_{}_{}.p".format(origin, datset)
        story_lengths_cnn, avg_story_cnn, summary_lengths_cnn, avg_summary_cnn = get_stats(filename_cnn)

        filename_dm = "../descriptives/param_files/descriptives_dailymail_dev.p"
        story_lengths_dm, avg_story_dm, summary_lengths_dm, avg_summary_dm = get_stats(filename_dm)

        origin = re.match(r".*\/param_files\/descriptives_(.*)_(.*).p", filename_cnn).group(2).upper()
        print("*** {} ***".format(origin))
        avg_story_overall = compute_average(avg_story_cnn, len(story_lengths_cnn), avg_story_dm, len(story_lengths_dm))
        avg_summary_overall = compute_average(avg_summary_cnn, len(summary_lengths_cnn), avg_summary_dm, len(summary_lengths_dm))
        print("Average story length: {}".format(avg_story_overall))
        print("Average summary length: {}".format(avg_summary_overall))

        plot_distribution(summary_lengths_cnn, summary_lengths_dm, datset, "summaries")
        plot_distribution(story_lengths_cnn, story_lengths_dm, datset, "stories")

