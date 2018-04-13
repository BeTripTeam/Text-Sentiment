from SentimentClassifier import SentimentClassifier
from helpers.text.TextProcessor import TextProcessor
from helpers.data.DataManager import DataManager


class SentimentAnalyzer:
    _positive_tags = []
    _negative_tags = []

    def __init__(self):
        self.data_menager = DataManager()
        self.text_processor = TextProcessor()
        self.classifier = SentimentClassifier(self.data_menager)
        self._import_tags_tables()

    def get_text_sentiment(self, text):
        text = self.text_processor.prepare_for_sentiment(text)
        c, p = self.classifier.get_class_with_prob(text)
        return c * p
    
    def get_tags_cluster_sentiment(self, tags_list):
        """
        Counts number of places_tags of each sentiment_words group.
        :param tags_list: list of strings (places_tags)
        :return: list tuples of a form (number of positive places_tags, number of negative places_tags)
        """
        sentiment = []
        for tags in tags_list:
            pos = 0
            neg = 0
            for tag in tags:
                if tag in self._negative_tags:
                    neg += 1
                if tag in self._positive_tags:
                    pos += 1
            sentiment.append((pos, neg))
        return sentiment

    def get_texts_cluster_sentiment(self, texts):
        """
        Classifies each text_helpers according to its sentiment_words.
        :param texts: list of strings
        :return: list of tuples of a form (sentiment_words, confidence in right classification)
        """
        dicts = [self.text_processor.prepare_for_sentiment(text) for text in texts]
        return self.classifier.get_classes_with_prob(dicts)

    def _import_tags_tables(self):
        """
        Loads lists of places_tags of different categories.
        """
        self._positive_tags = self.data_menager.positive_tags_en
        self._negative_tags = self.data_menager.negative_tags_en
