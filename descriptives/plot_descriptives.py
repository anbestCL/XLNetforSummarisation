import pickle
import re
import statistics
import matplotlib.pyplot as plt
import operator


def get_stats(filename):
    origin = re.match(r".*\/descriptives_(.*)_(.*).p", filename).group(1).upper()
    set = re.match(r".*\/descriptives_(.*)_(.*).p", filename).group(2).upper()
    with open(filename, "rb") as lengths:
        story_dict = pickle.load(lengths)
        summary_dict = pickle.load(lengths)
    print("*** {} : {} ***".format(origin, set))
    # longest/shortest story
    zero_lengths_stories = [name for name, length in story_dict.items() if length == 0]
    zero_lengths_summaries = [name for name, length in summary_dict.items() if length == 0]
    print(zero_lengths_stories)
    print(zero_lengths_summaries)
    story_lengths = {name: length for name, length in story_dict.items() if name not in zero_lengths_stories and name not in zero_lengths_summaries}
    summary_lengths = {name: length for name, length in summary_dict.items() if name not in zero_lengths_stories and name not in zero_lengths_summaries}
    assert len(story_lengths.values()) == len(summary_lengths.values())
    print(len(story_lengths.values()))

    longest_story, longest_length = max(story_lengths.items(), key=operator.itemgetter(1))
    shortest_story, shortest_length = min(story_lengths.items(), key=operator.itemgetter(1))
    avg_story_length = statistics.mean(story_lengths.values())
    print("Longest story {} with length: {}".format(longest_story, longest_length))
    print("Shortest story {} with length: {}".format(shortest_story, shortest_length))
    print("Average story length: {}".format(avg_story_length))
    # longest/shortest summary
    longest_summary, longest_length_sum = max(summary_lengths.items(), key=operator.itemgetter(1))
    shortest_summary, shortest_length_sum = min(summary_lengths.items(), key=operator.itemgetter(1))
    avg_summary_length = statistics.mean(summary_lengths.values())
    print("Longest summary {} length: {}".format(longest_summary, longest_length_sum))
    print("Shortest summary {} length: {}".format(shortest_summary, shortest_length_sum))
    print("Average summary length: {}".format(avg_summary_length))
    return (list(story_lengths.values()), avg_story_length, len(story_lengths.values())), (list(summary_lengths.values()), avg_summary_length, len(summary_lengths.values()))


def plot_distribution(lengths_cnn, lengths_dm, dataset, text):
    fig, axs = plt.subplots(3,1, tight_layout=True)
    axs[0].set_title("Distribution of {} lengths in {} dataset".format(text, dataset))

    hist, bins, _ = axs[0].hist(lengths_dm, bins=50, alpha=0.75, facecolor='blue', edgecolor='black', linewidth=1.2,
                label='Dailymail {}'.format(text))
    axs[0].set_ylabel("{} count".format(text))
    axs[0].legend()

    axs[1].hist(lengths_cnn, bins=bins, alpha=0.75, facecolor='green', edgecolor='black', linewidth=1.2, label='CNN {}'.format(text))
    axs[1].set_ylabel("{} count".format(text))
    axs[1].legend()

    lengths = lengths_cnn + lengths_dm
    axs[2].hist(lengths, bins=bins, alpha=0.75, facecolor='red', edgecolor='black', linewidth=1.2, label="CNN + Dailymail {}".format(text))
    axs[2].set_xlabel("Token count")
    axs[2].set_ylabel("{} count".format(text))
    axs[2].legend()
    plt.savefig('descriptives/plots/{}_{}_distribution'.format(dataset,text))

filename = "descriptives/param_files/descriptives_cnn_dev.p"
(story_lengths_cnn, avg_story_cnn, stories_count_cnn), (summary_lengths_cnn, avg_summary_cnn, summaries_count_cnn) = get_stats(filename)

filename2 = "descriptives/param_files/descriptives_dailymail_dev.p"
(story_lengths_dm, avg_story_dm, stories_count_dm), (summary_lengths_dm, avg_summary_dm, summaries_count_dm) = get_stats(filename2)

origin = re.match(r".*\/param_files\/descriptives_(.*)_(.*).p", filename).group(2).upper()
print("*** {} ***".format(origin))
avg_story_overall = (avg_story_cnn*stories_count_cnn+avg_story_dm*stories_count_dm)/(stories_count_cnn+stories_count_dm)
avg_summary_overall = (avg_summary_cnn*summaries_count_cnn+avg_summary_dm*summaries_count_dm)/(summaries_count_cnn+summaries_count_dm)
print("Average story length: {}".format(avg_story_overall))
print("Average summary length: {}".format(avg_summary_overall))

plot_distribution(summary_lengths_cnn, summary_lengths_dm, "dev", "summaries")
plot_distribution(story_lengths_cnn, story_lengths_dm, "dev", "stories")

filename = "descriptives/param_files/descriptives_cnn_train.p"
(story_lengths_cnn, avg_story_cnn, stories_count_cnn), (summary_lengths_cnn, avg_summary_cnn, summaries_count_cnn) = get_stats(filename)

filename2 = "descriptives/param_files/descriptives_dailymail_train.p"
(story_lengths_dm, avg_story_dm, stories_count_dm), (summary_lengths_dm, avg_summary_dm, summaries_count_dm) = get_stats(filename2)

origin = re.match(r".*\/param_files\/descriptives_(.*)_(.*).p", filename).group(2).upper()
print("*** {} ***".format(origin))
avg_story_overall = (avg_story_cnn*stories_count_cnn+avg_story_dm*stories_count_dm)/(stories_count_cnn+stories_count_dm)
avg_summary_overall = (avg_summary_cnn*summaries_count_cnn+avg_summary_dm*summaries_count_dm)/(summaries_count_cnn+summaries_count_dm)
print("Average story length: {}".format(avg_story_overall))
print("Average summary length: {}".format(avg_summary_overall))

plot_distribution(summary_lengths_cnn, summary_lengths_dm, "train", "summaries")
plot_distribution(story_lengths_cnn, story_lengths_dm, "train", "stories")

filename = "descriptives/param_files/descriptives_cnn_test.p"
(story_lengths_cnn, avg_story_cnn, stories_count_cnn), (summary_lengths_cnn, avg_summary_cnn, summaries_count_cnn) = get_stats(filename)

filename2 = "descriptives/param_files/descriptives_dailymail_test.p"
(story_lengths_dm, avg_story_dm, stories_count_dm), (summary_lengths_dm, avg_summary_dm, summaries_count_dm) = get_stats(filename2)

origin = re.match(r".*\/param_files\/descriptives_(.*)_(.*).p", filename).group(2).upper()
print("*** {} ***".format(origin))
avg_story_overall = (avg_story_cnn*stories_count_cnn+avg_story_dm*stories_count_dm)/(stories_count_cnn+stories_count_dm)
avg_summary_overall = (avg_summary_cnn*summaries_count_cnn+avg_summary_dm*summaries_count_dm)/(summaries_count_cnn+summaries_count_dm)
print("Average story length: {}".format(avg_story_overall))
print("Average summary length: {}".format(avg_summary_overall))

plot_distribution(summary_lengths_cnn, summary_lengths_dm, "test", "summaries")
plot_distribution(story_lengths_cnn, story_lengths_dm, "test", "stories")