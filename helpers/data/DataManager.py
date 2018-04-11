import pandas as pd
import pickle
import os

from .Paths import Paths


class DataManager:
    # EN
    positive_tags_en = []
    negative_tags_en = []
    
    sentiment_classifier_en = ''

    # RU
    positive_tags_ru = []
    negative_tags_ru = []
    
    sentiment_classifier_ru = ''
    
    def __init__(self):
        self._basepath = os.path.dirname(__file__)
        if len(self.positive_tags_en) == 0:
            paths = Paths()
            self._import_tags_tables(paths)
            self._import_sentiment_classifier(paths)

    def _import_tags_tables(self, paths):
        """Loads lists of places_tags of different categories.
        """
        # EN
        self.positive_tags_en = self._get_table(paths.positive_tags_en, 'word')
        self.negative_tags_en = self._get_table(paths.negative_tags_en, 'word')
        
        # RU
        self.positive_tags_ru = self._get_table(paths.positive_tags_ru, '1')
        self.negative_tags_ru = self._get_table(paths.negative_tags_ru, '1')
        
    def _get_table(self, path, name):
        table = pd.read_csv(
            self._get_path(path), skip_blank_lines=True, keep_default_na=False, dtype=str, error_bad_lines=False)[name]
        s = set(table)
        if '' in s:
            s.remove('')
        return s

    def _get_path(self, path):
        return os.path.abspath(os.path.join(self._basepath, "..", "..", "..", "..", path))

    def _import_sentiment_classifier(self, paths):
        filepath = self._get_path(paths.sentiment_classifier)
        f = open(filepath, 'rb')
        self.sentiment_classifier_en = pickle.load(f)
        f.close()

        filepath = self._get_path(paths.sentiment_classifier_ru)
        f = open(filepath, 'rb')
        self.sentiment_classifier_ru = pickle.load(f)
        f.close()