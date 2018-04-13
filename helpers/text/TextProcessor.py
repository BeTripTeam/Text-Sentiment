from nltk import TweetTokenizer
from nltk.stem import WordNetLemmatizer
# from social_analytics.analytics.places_analytics.text_helpers.TextCorrector import TextCorrector


class TextProcessor:
    """
    Works with text_helpers and prepares it for use of other methods.
    """

    def __init__(self):
        self.tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True)
        # self.corrector = TextCorrector()
        self.wnl = WordNetLemmatizer()

    def tokenize(self, text):
        """
        Splits text_helpers into the list of tokens. Token is a separate meaningful unit (word, punctuation mark, emoticon)
        :param text: string of text_helpers
        :return: list of tokens (string objects)
        """
        return self.tokenizer.tokenize(text)

    # def correct(self, token):
    #     """
    #     Corrects mistakes in the given word. It finds statistically most probable real word from its dictionary for
    #     the word not in dictionary.
    #     :param token: word
    #     :return: corrected word
    #     """
    #     return self.corrector.correct(token)

    def make_dict(self, words):
        """
        Makes a dictionary out of list. Needed for sentiment_words classification.
        :param words: list
        :return: dictionary of a form ("item" : True)
        """
        return dict([(word, True) for word in words])

    def prepare_for_sentiment(self, text):
        """
        Prepares a text_helpers for sentiment_words classification.
        :param text: string
        :return: dict object
        """
        tokens = self.tokenize(text.lower())
        #tokens = [self.correct(t) for t in tokens]
        data = self.make_dict(tokens)
        return data

    def to_singular(self, words):
        return [self.wnl.lemmatize(i) for i in words]

    def lowercase(self, words):
        return [word.lower() for word in words]