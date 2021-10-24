"""
This module reads the data, records the locations of capitalized words, and visualize them on the heat map
"""
import re
import string

import seaborn as sb
import matplotlib.pyplot as plt
import language_tool_python
import nltk
from nltk.tokenize import word_tokenize


def analyze_domain(paths):
    print("Analyze", paths)
    is_stack_overflow = "stack" in paths[0]
    grammar_analyzer = language_tool_python.LanguageTool("en-US")
    matches_num = 0
    sentence_capital_start_num = 0
    sentences = []
    max_sentence_len = 0
    sentence_max_len = ""  # for debugging purpose
    for data_path in paths:
        with open(data_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:  # not the empty line
                    sentences.append(line)
                    sentence = line.split()
                    if len(sentence) > max_sentence_len:
                        max_sentence_len = len(sentence)
                        sentence_max_len = sentence
    counter_capital_word_location = [0] * max_sentence_len  # initialize the record of locations
    counter_word_location = [0] * max_sentence_len
    for sentence in sentences:
        if is_stack_overflow:
            tokens = word_tokenize(sentence)
            print(tokens)

        matches = grammar_analyzer.check(sentence)
        matches_num += len(matches)
        # if matches:
        #     for match in matches:
        #         print(match)

        tokens = word_tokenize(sentence)
        # print(tokens)

        processed_sentence = sentence.translate(str.maketrans('', '', string.punctuation))
        if processed_sentence[0].isupper():
            sentence_capital_start_num += 1
        # record the capitalization
        for index, word in enumerate(sentence.split()):
            if word.isupper():
                counter_capital_word_location[index] += 1
    # visualization
    sb.set(rc={'figure.figsize': (10, 3)})
    fig = sb.heatmap([counter_capital_word_location], annot=True)
    fig.set(xlabel='Location of word', ylabel='occurrence')
    plt.show()
    print("Probability of Sentences Starting with Capital Letter = {} / {} = {}".format(sentence_capital_start_num,
                                                                                        len(sentences),
                                                                                        sentence_capital_start_num / len(
                                                                                                sentences)))


# analyze_domain(["data/hardware_zone_1.txt", "data/hardware_zone_2.txt"])
analyze_domain(["data/stack_overflow_1.txt", "data/stack_overflow_2.txt"])
# analyze_domain(["data/cna_1.txt", "data/cna_2.txt"])
