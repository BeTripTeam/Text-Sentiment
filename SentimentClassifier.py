from helpers.text.LanguageClassifier import is_en


class SentimentClassifier:
    """
    Sentiment class labels:
    1: positive sentiment_words
    -1: negative
    """
    
    def __init__(self, data_menager):
        self._classifier_en = data_menager.sentiment_classifier_en
        self._classifier_ru = data_menager.sentiment_classifier_ru
    
    # def get_classes(self, items):
    #     """Returns classes of each item in the set.
    #     :param items: list of dictionaries like {"word" : True}
    #     :return list class labels with highest probability
    #     """
    #     return self.classifier_en.classify_many(items)
    #
    # def get_class(self, item):
    #     """Returns class for the given item.
    #     :param items: dictionary like {"word" : True}
    #     :return class label with highest probability
    #     """
    #     return self.classifier_en.classify(item)
    
    def get_class_with_prob(self, text):
        """
        Evaluates probabilities of a text_helpers to be with positive and negative sentiment_words.
        :param text: dictionary like {"word" : True}
        :return a tuple of a form (class label, probability of that class)
        """
        if self._is_en(text):
            dist = self._classifier_en.prob_classify(text)
        else:
            dist = self._classifier_ru.prob_classify(text)
        return dist.max(), dist.prob(dist.max())
    
    def get_classes_with_prob(self, items):
        """
        Evaluates probabilities of each text_helpers to be with positive and negative sentiment_words.
        :param items: list of dictionaries like {"word" : True}
        :return a list of tuples of a form (class, probability of that class)
        """
        ans = list()
        for text in items:
            ans.append(self.get_class_with_prob(text))
        return ans

    def _is_en(self, text):
        for t in text:
            if not is_en(t):
                return False
        return True