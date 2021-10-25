import argparse
import json
import re
import string

import nltk
import numpy as np
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
from wordcloud import WordCloud

from gsdmm import MovieGroupProcess


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--business", type=str, help="The id of business of interests", default="")
    parser.add_argument("--topic_num", type=int, help="The number of topic numbers", default=10)
    parser.add_argument("--is_visualize", type=bool, help="Whether to export the visualization of word clouds",
                        default=True)
    arguments_parsed = parser.parse_args()
    return arguments_parsed


def read_dataset(data_path='data/reviewSelected100.json'):
    """
    Reads dataset from the path
    :return: reviewList, processed
    """
    reviewList = []
    stop_words = set(stopwords.words('english'))
    stemmer = nltk.PorterStemmer()

    print("Started Reading JSON file which contains multiple JSON objects")
    with open(data_path, encoding="ISO-8859-1") as datafile:
        data = datafile.readlines()
        progress_bar = tqdm(total=len(data), desc="Load dataset")
        for jsonObj in data:
            reviewDict = json.loads(jsonObj)
            # preprocess data
            sentence = reviewDict["text"].translate(str.maketrans('', '', string.punctuation)).lower()
            sentence = re.sub(r'\d', '', sentence)  # remove digits
            sentence = word_tokenize(sentence)
            sentence = [x for x in sentence if x not in stop_words]  # remove stop words
            # sentence = [stemmer.stem(x) for x in sentence]
            reviewDict["text"] = sentence
            reviewList.append(reviewDict)
            progress_bar.update(1)
        progress_bar.close()
    return reviewList


def get_reviews_of_business(business_id, reviews):
    business_reviews = []
    for review in reviews:
        if review["business_id"] == business_id:
            business_reviews.append(review)
            # print(review["text"])
    return business_reviews


arguments = parse_arguments()
reviews = read_dataset()

if arguments.business:
    reviews = get_reviews_of_business(arguments.business, reviews)

reviews_text = [review["text"] for review in reviews]


def model_topic_LDA(is_visualize=False):
    print("Modelling topic with LDA...")
    count_vectorizer = CountVectorizer()
    data = count_vectorizer.fit_transform([' '.join(x) for x in reviews_text])
    topic_modeler = LDA(n_components=arguments.topic_num, n_jobs=-1, random_state=21)
    topic_modeler.fit(data)

    words = count_vectorizer.get_feature_names_out()

    words_in_topic = {}
    for topic_id, topic in enumerate(topic_modeler.components_):
        this_topic_words = [words[i] for i in topic.argsort()[-arguments.topic_num:]]
        print("Topic {}: ".format(topic_id), this_topic_words)
        words_in_topic[topic_id] = this_topic_words
    if is_visualize:
        visualize_word_cloud(words_in_topic, file_prefix="LDA")
    print("=======================Modelling topic with LDA Completes=================")


def visualize_word_cloud(words_in_topic, file_prefix):
    for topic_id, words in words_in_topic.items():
        text = []
        for index, word in enumerate(words):
            text.extend([word] * (index + 1))
        if not text:
            continue
        text = " ".join(text)
        word_cloud = WordCloud(collocations=False, background_color='white').generate(text)
        plt.imshow(word_cloud, interpolation='bilinear')
        plt.axis("off")
        plt.savefig("{}_{}.png".format(file_prefix, topic_id))


model_topic_LDA(is_visualize=arguments.is_visualize)


def model_topic_GSDMM(is_visualize=False):
    print("Modelling topic with GSDMM...")
    vocabulary = set([x for review in reviews for x in review["text"]])
    topic_modeler = MovieGroupProcess(K=30, alpha=0.1, beta=0.1, n_iters=10)
    topic_modeler.fit(reviews_text, len(vocabulary))
    # get the top 15 clusters with most reviews
    cluster_review_count = np.array(topic_modeler.cluster_doc_count)
    top_clusters = cluster_review_count.argsort()[-arguments.topic_num:]
    # print the words in top clusters
    words_in_topic = {}
    for cluster in top_clusters:
        sort_dict = sorted(topic_modeler.cluster_word_distribution[cluster].items(), key=lambda x: x[1], reverse=True)[
                    :10]
        words_sorted = [x[0] for x in sort_dict][::-1]
        print("Topic {}: ".format(cluster), words_sorted)
        words_in_topic[cluster] = words_sorted
    if is_visualize:
        visualize_word_cloud(words_in_topic, file_prefix="GSDMM")
    print("==================Modelling topic with GSDMM completes====================")


model_topic_GSDMM(is_visualize=arguments.is_visualize)
