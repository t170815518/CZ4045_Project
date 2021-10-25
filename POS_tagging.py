import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import brown
import json
import ssl
import random


def sentence_token_nltk(text):
    sent_tokenize_list = sent_tokenize(text)
    return sent_tokenize_list


def nltk_unigram_tagger(sentence):
    text = nltk.word_tokenize(sentence)
    brown_tagged_sents = brown.tagged_sents(categories='news')
    unigram_tagger = nltk.UnigramTagger(brown_tagged_sents)
    result = unigram_tagger.tag(text)
    print("POS tagging using unigram: \n", result)


def pos_tagging_nltk(sentence):
    text = nltk.word_tokenize(sentence)
    text_tagged = nltk.pos_tag(text)
    print("POS tagging using nltk:\n", text_tagged)


def main():
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    # nltk.download()

    data = []
    with open('data/reviewSelected100.json', encoding="ISO-8859-1") as datafile:
        for jsonObj in datafile:
            reviewDict = json.loads(jsonObj)
            data.append(reviewDict)

    sentence = []
    for item in data:
        text = item["text"]
        sentence_sliced = sentence_token_nltk(text)
        for item in sentence_sliced:
            sentence.append(item)

    for item in random.sample(sentence, 5):
        print("----------------------------------------")
        print("The sentence chosen is: \n", item)
        print("----------------------------------------")

        nltk_unigram_tagger(item)
        pos_tagging_nltk(item)


if __name__ == '__main__':
    main()
